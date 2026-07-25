//! MiniCPM5 provider (capabilities layer) — bridges MiniCPM5-1B's inline
//! XML tool-call format into the kernel's `StreamEvent::ToolCall`.
//!
//! Mirrors [`super::ollama`] over the same `/api/chat` wire protocol, but
//! where Ollama's `tool_calls` field is empty for MiniCPM5 the model emits
//! its calls as inline `<function>…</function>` XML inside `content`. This
//! decoder intercepts `content` text, peels off those XML blocks, and emits
//! them as proper [`StreamEvent::ToolCall`]s so the agent loop drives tools
//! exactly as it does for native OpenAI/Claude tool calls.
//!
//! The XML format handled here (`<function name="…"><arguments><param …>…`
//! with `<functionname=` tokenizer-collapse normalisation, CDATA unwrapping,
//! and `\u0120` space fixing) is ported from OpenBMB's reference parser
//! `tool_parsers/minicpm5xml_tool_parser.py` (Apache-2.0).

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde_json::{Map, Value};

use atomcode_kernel::message::Message;
use atomcode_kernel::provider::{ChatOptions, LlmProvider};
use atomcode_kernel::stream::{ProviderError, StreamEvent, TokenUsage};
use atomcode_kernel::tool::{ToolCall, ToolDef};

use super::ollama::{open_stream, OllamaConfig};

pub struct MinicpmProvider {
    cfg: OllamaConfig,
    client: reqwest::Client,
    url: String,
    session_id: std::sync::OnceLock<String>,
}

impl MinicpmProvider {
    pub fn new(cfg: OllamaConfig) -> Result<Self, ProviderError> {
        let mut builder = crate::proxy::apply_async_proxy_policy(reqwest::Client::builder())
            .connect_timeout(cfg.connect_timeout)
            .pool_idle_timeout(super::retry::POOL_IDLE_TIMEOUT)
            .user_agent(cfg.user_agent.as_deref().unwrap_or(super::DEFAULT_USER_AGENT));
        if cfg.skip_tls_verify {
            builder = builder.danger_accept_invalid_certs(true);
        }
        let client = builder.build().map_err(|e| ProviderError {
            retryable: false,
            message: format!("http client build failed: {e}"),
            ..Default::default()
        })?;
        let url = format!("{}/api/chat", cfg.base_url.trim_end_matches('/'));
        Ok(Self {
            cfg,
            client,
            url,
            session_id: std::sync::OnceLock::new(),
        })
    }
}

#[async_trait]
impl LlmProvider for MinicpmProvider {
    fn model_name(&self) -> &str {
        &self.cfg.model
    }

    fn context_window(&self) -> u32 {
        self.cfg.context_window
    }

    fn bind_session_id(&self, session_id: &str) {
        let _ = self.session_id.set(session_id.to_string());
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        options: &ChatOptions,
    ) -> Result<BoxStream<'static, StreamEvent>, ProviderError> {
        // Reuse Ollama's body builder with EMPTY tools — MiniCPM5 was trained
        // to call tools from a text description in the prompt, not from
        // Ollama's tool-call template. atomcode already renders tool docs into
        // the system prompt, which is exactly what MiniCPM5 expects. Passing
        // an empty slice means `build_request_body` omits the `tools` field.
        let mut body = super::ollama::build_request_body(
            &self.cfg.model,
            messages,
            &[],
            options,
            &self.cfg,
        );
        // Ollama's MiniCPM5 GGUF defaults to num_ctx=4096, but atomcode's
        // system prompt + tool schemas alone exceed that. Force the context
        // window to the configured value (32768 default) so the request
        // isn't rejected with "exceeds available context size". The base
        // OllamaProvider doesn't set num_ctx because other models' defaults
        // are usually adequate; MiniCPM5 is the exception.
        if let Some(obj) = body.as_object_mut() {
            let opts = obj
                .entry("options".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(opts_obj) = opts.as_object_mut() {
                opts_obj.insert("num_ctx".to_string(), Value::from(self.cfg.context_window));
            }
        }
        super::wire_dump_request(&self.cfg.model, &body);

        let policy = self.cfg.retry.clone();
        let client = self.client.clone();
        let url = self.url.clone();
        let api_key = self.cfg.api_key.clone();
        let session_id = self.session_id.get().cloned().unwrap_or_default();
        let idle = self.cfg.idle_timeout;
        let resp = open_stream(&client, &url, &body, &api_key, &session_id, &policy).await?;

        let s = async_stream::stream! {
            const MAX_STREAM_ATTEMPTS: u32 = 3;
            let mut stream_attempt = 1u32;
            let mut resp = resp;
            'reopen: loop {
                let mut dec = MinicpmDecoder::new();
                let mut emitted_any = false;
                let byte_stream = resp.bytes_stream();
                futures::pin_mut!(byte_stream);
                loop {
                    match tokio::time::timeout(idle, byte_stream.next()).await {
                        Err(_elapsed) => {
                            yield StreamEvent::Error(ProviderError {
                                retryable: false,
                                message: "stream idle timeout".to_string(),
                                ..Default::default()
                            });
                            return;
                        }
                        Ok(None) => {
                            for ev in dec.finish() { yield ev; }
                            return;
                        }
                        Ok(Some(Err(e))) => {
                            if !emitted_any && stream_attempt < MAX_STREAM_ATTEMPTS {
                                tokio::time::sleep(super::retry::compute_backoff(stream_attempt, &policy)).await;
                                if let Ok(fresh) = open_stream(&client, &url, &body, &api_key, &session_id, &policy).await {
                                    stream_attempt += 1;
                                    resp = fresh;
                                    continue 'reopen;
                                }
                            }
                            yield StreamEvent::Error(ProviderError {
                                retryable: false,
                                message: super::retry::stream_read_error_message(&e),
                                ..Default::default()
                            });
                            return;
                        }
                        Ok(Some(Ok(chunk))) => {
                            let mut saw_done = false;
                            for ev in dec.feed(chunk.as_ref()) {
                                emitted_any = true;
                                if matches!(ev, StreamEvent::Done { .. }) {
                                    saw_done = true;
                                }
                                yield ev;
                            }
                            if saw_done { return; }
                        }
                    }
                }
            }
        };

        // tools is unused (we drop the field above) but the signature requires it.
        let _ = tools;
        Ok(s.boxed())
    }
}

/// NDJSON decoder that intercepts MiniCPM5's inline `<function>` / `<think>`
/// markup in the `content` channel and re-emits it as ToolCall / Reasoning.
struct MinicpmDecoder {
    buf: Vec<u8>,
    xml: XmlReplayer,
    tool_index: u32,
    truncated: bool,
    done: bool,
}

impl MinicpmDecoder {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            xml: XmlReplayer::new(),
            tool_index: 0,
            truncated: false,
            done: false,
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<StreamEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&raw);
            let text = text.trim();
            if !text.is_empty() {
                self.process_line(text, &mut out);
            }
            if self.done {
                break;
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        // Flush any buffered XML markup before the terminal Done.
        for f in self.xml.finish() {
            self.emit_flush(f, &mut out);
        }
        if !self.done {
            out.push(StreamEvent::Done {
                truncated: self.truncated,
            });
            self.done = true;
        }
        out
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<StreamEvent>) {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
            out.push(StreamEvent::Error(ProviderError {
                retryable: false,
                message: format!("provider error: {}", err),
                ..Default::default()
            }));
            self.done = true;
            return;
        }
        if let Some(msg) = v.get("message") {
            // The crucial difference from OllamaNdjsonDecoder: route `content`
            // through the XML replayer so <function> blocks become ToolCalls.
            if let Some(c) = msg.get("content").and_then(|c| c.as_str()) {
                if !c.is_empty() {
                    for f in self.xml.push(c) {
                        self.emit_flush(f, out);
                    }
                }
            }
            // Ollama surfaces MiniCPM5's <think> as a separate `thinking` field
            // when the template splits it — pass it straight through.
            if let Some(t) = msg.get("thinking").and_then(|t| t.as_str()) {
                if !t.is_empty() {
                    out.push(StreamEvent::Reasoning(t.to_string()));
                }
            }
            // Some Ollama builds DO populate tool_calls for MiniCPM5 — accept
            // them too as a belt-and-braces fallback.
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    let f = tc.get("function");
                    let name = f
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = f
                        .and_then(|f| f.get("arguments"))
                        .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".into()))
                        .unwrap_or_else(|| "{}".into());
                    let id = format!("minicpm_call_{}", self.tool_index);
                    self.tool_index += 1;
                    out.push(StreamEvent::ToolCall(ToolCall { id, name, arguments: args }));
                }
            }
        }
        if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
            if v.get("done_reason").and_then(|r| r.as_str()) == Some("length") {
                self.truncated = true;
            }
            // Flush buffered XML before reporting Done.
            for f in self.xml.finish() {
                self.emit_flush(f, out);
            }
            let prompt = v
                .get("prompt_eval_count")
                .and_then(|n| n.as_u64())
                .unwrap_or(0) as u32;
            let completion = v.get("eval_count").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            if prompt > 0 || completion > 0 {
                out.push(StreamEvent::Usage(TokenUsage {
                    prompt,
                    completion,
                    cached: 0,
                }));
            }
            out.push(StreamEvent::Done {
                truncated: self.truncated,
            });
            self.done = true;
        }
    }

    fn emit_flush(&mut self, f: Flush, out: &mut Vec<StreamEvent>) {
        match f {
            Flush::Text(s) => out.push(StreamEvent::TextDelta(s)),
            Flush::Reasoning(s) => out.push(StreamEvent::Reasoning(s)),
            Flush::ToolCall { name, arguments } => {
                let id = format!("minicpm_call_{}", self.tool_index);
                self.tool_index += 1;
                out.push(StreamEvent::ToolCall(ToolCall { id, name, arguments }));
            }
        }
    }
}

// ── XML replayer (ported from OpenBMB's minicpm5xml_tool_parser.py) ──────

enum Flush {
    Text(String),
    Reasoning(String),
    ToolCall { name: String, arguments: String },
}

struct XmlReplayer {
    pending: String,
    in_think: bool,
    in_function: bool,
}

impl XmlReplayer {
    fn new() -> Self {
        Self {
            pending: String::new(),
            in_think: false,
            in_function: false,
        }
    }

    fn push(&mut self, chunk: &str) -> Vec<Flush> {
        self.pending.push_str(chunk);
        let mut out = Vec::new();
        self.drain(&mut out);
        out
    }

    fn finish(&mut self) -> Vec<Flush> {
        let mut out = Vec::new();
        if self.in_think {
            if !self.pending.is_empty() {
                out.push(Flush::Reasoning(std::mem::take(&mut self.pending)));
            }
            self.in_think = false;
        } else if self.in_function {
            if let Some((name, args)) = parse_function_block(&self.pending) {
                out.push(Flush::ToolCall {
                    name,
                    arguments: args,
                });
            } else if !self.pending.is_empty() {
                out.push(Flush::Text(std::mem::take(&mut self.pending)));
            }
            self.in_function = false;
        } else if !self.pending.is_empty() {
            out.push(Flush::Text(std::mem::take(&mut self.pending)));
        }
        out
    }

    fn drain(&mut self, out: &mut Vec<Flush>) {
        loop {
            if self.in_think {
                if let Some(end) = self.pending.find("</think>") {
                    let body = self.pending[..end].to_string();
                    self.pending = self.pending[end + 8..].to_string();
                    if !body.is_empty() {
                        out.push(Flush::Reasoning(body));
                    }
                    self.in_think = false;
                    continue;
                }
                let keep = 7; // len("</think>")-1
                if self.pending.len() > keep {
                    let safe = self.pending.len() - keep;
                    let body: String = self.pending.drain(..safe).collect();
                    if !body.is_empty() {
                        out.push(Flush::Reasoning(body));
                    }
                }
                return;
            }
            if self.in_function {
                if let Some(end) = self.pending.find("</function>") {
                    let block = self.pending[..end].to_string();
                    self.pending = self.pending[end + 11..].to_string();
                    if let Some((name, args)) = parse_function_block(&block) {
                        out.push(Flush::ToolCall {
                            name,
                            arguments: args,
                        });
                    }
                    self.in_function = false;
                    continue;
                }
                return;
            }
            match self.pending.find('<') {
                None => {
                    if !self.pending.is_empty() {
                        out.push(Flush::Text(std::mem::take(&mut self.pending)));
                    }
                    return;
                }
                Some(lt) => {
                    if lt > 0 {
                        let text: String = self.pending.drain(..lt).collect();
                        out.push(Flush::Text(text));
                    }
                    let after = &self.pending[1..];
                    if after.starts_with("think") {
                        if let Some(gt) = self.pending.find('>') {
                            self.pending = self.pending[gt + 1..].to_string();
                            self.in_think = true;
                            continue;
                        }
                        return;
                    }
                    if after.starts_with("function") || after.starts_with("functionname=") {
                        self.in_function = true;
                        continue;
                    }
                    // Unknown tag — flush up to and including its '>' as text.
                    if let Some(gt) = self.pending.find('>') {
                        let text: String = self.pending.drain(..=gt).collect();
                        out.push(Flush::Text(text));
                        continue;
                    }
                    return;
                }
            }
        }
    }
}

/// Parse `<function name="…" …>…<param name="k">v</param>…</function>`
/// (opening tag already stripped of nothing — fed the full inner XML).
fn parse_function_block(raw: &str) -> Option<(String, String)> {
    let normalised = raw.replace('\u{0120}', " ").replace("<functionname=", "<function name=");
    let open_end = normalised.find('>')?;
    let open_tag = &normalised[..open_end];
    let name = extract_attr(open_tag, "name")?;

    let body = &normalised[open_end + 1..];
    let mut args = Map::new();
    let mut rest = body;
    while let Some(pstart) = rest.find("<param") {
        let pname = extract_attr(&rest[pstart..], "name")?;
        let tag_end = rest[pstart..].find('>')? + pstart;
        let val_start = tag_end + 1;
        let after = &rest[val_start..];
        let val_end = after.find("</param>")?;
        let raw_val = &after[..val_end];
        // CDATA unwrap — but tolerate the malformed endings a 1B model
        // produces: standard `]]>`, truncated `]]` / `]`, or nothing.
        // Only strip the opener when we actually saw it; otherwise leave the
        // value untouched (a literal '<' in content is rare but legal).
        let raw_val = if let Some(inner) = raw_val.strip_prefix("<![CDATA[") {
            inner
                .strip_suffix("]]>")
                .or_else(|| inner.strip_suffix("]]"))
                .or_else(|| inner.strip_suffix("]"))
                .unwrap_or(inner)
        } else {
            raw_val
        };
        let decoded = decode_xml_entities(raw_val);
        let value = if decoded.trim().is_empty() {
            Value::String(String::new())
        } else {
            serde_json::from_str::<Value>(&decoded).unwrap_or(Value::String(decoded))
        };
        args.insert(pname, value);
        rest = &after[val_end + 8..];
    }
    Some((name, Value::Object(args).to_string()))
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    if let Some(s) = tag.find(&needle) {
        let vs = s + needle.len();
        let ve = tag[vs..].find('"')? + vs;
        return Some(tag[vs..ve].to_string());
    }
    let needle = format!("{attr}='");
    if let Some(s) = tag.find(&needle) {
        let vs = s + needle.len();
        let ve = tag[vs..].find('\'')? + vs;
        return Some(tag[vs..ve].to_string());
    }
    None
}

fn decode_xml_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(input: &str) -> Vec<String> {
        let mut r = XmlReplayer::new();
        let mut out = r.push(input);
        out.extend(r.finish());
        out.into_iter()
            .map(|f| match f {
                Flush::Text(s) => format!("T:{}", s),
                Flush::Reasoning(s) => format!("R:{}", s),
                Flush::ToolCall { name, arguments } => format!("C:{}({})", name, arguments),
            })
            .collect()
    }

    #[test]
    fn plain_text() {
        assert_eq!(run("hello"), vec!["T:hello"]);
    }

    #[test]
    fn think_to_reasoning() {
        assert_eq!(
            run("<think>considering</think>done"),
            vec!["R:considering", "T:done"]
        );
    }

    #[test]
    fn function_to_toolcall() {
        let out = run(
            "<function name=\"get_weather\"><arguments>\
             <param name=\"city\">London</param>\
             </arguments></function>",
        );
        assert_eq!(out, vec!["C:get_weather({\"city\":\"London\"})"]);
    }

    #[test]
    fn real_minicpm_output_no_arguments_wrapper() {
        // The EXACT format observed from openbmb/minicpm5:Q4_K_M via Ollama —
        // <param> sits directly under <function>, no <arguments> wrapper.
        let out = run("<function name=\"get_weather\"><param name=\"city\">北京</param></function>");
        assert_eq!(out, vec!["C:get_weather({\"city\":\"北京\"})"]);
    }

    #[test]
    fn number_param_typed() {
        let out = run(
            "<function name=\"edit\"><arguments>\
             <param name=\"line\">42</param>\
             </arguments></function>",
        );
        assert_eq!(out, vec!["C:edit({\"line\":42})"]);
    }

    #[test]
    fn cdata_survives() {
        let xml = "<function name=\"write\"><arguments>\
                   <param name=\"content\"><![CDATA[if x < 2 {]]></param>\
                   </arguments></function>";
        let out = run(xml);
        assert!(out[0].contains("x < 2"));
        // The CDATA wrapper MUST be fully stripped — no leading '<![CDATA['
        // and no trailing ']]>' leakage into the value.
        assert!(!out[0].contains("CDATA"), "CDATA marker leaked: {}", out[0]);
    }

    #[test]
    fn cdata_with_json_content_no_residue() {
        // Real-world failure: model wraps a JSON value in CDATA. The parser
        // must return the clean JSON, not leak '<![CDATA[' at the start or
        // leave a dangling ']' at the end.
        let xml = "<function name=\"write_file\"><arguments>\
                   <param name=\"file_path\">config.json</param>\
                   <param name=\"content\"><![CDATA[{\"name\":\"minicpm5\",\"version\":\"1.0\"}]]></param>\
                   </arguments></function>";
        let out = run(xml);
        assert_eq!(out.len(), 1, "expected one tool call, got: {:?}", out);
        let args_str = &out[0];
        assert!(args_str.starts_with("C:write_file("));
        let args_json = &args_str["C:write_file(".len()..args_str.len() - 1];
        let v: Value = serde_json::from_str(args_json).expect("args must be valid JSON");
        assert_eq!(v["file_path"], "config.json");
        assert_eq!(v["content"]["name"], "minicpm5");
        assert_eq!(v["content"]["version"], "1.0");
    }

    #[test]
    fn malformed_cdata_truncated_ending_still_unwrapped() {
        // 1B models emit malformed CDATA: opener present, ending truncated
        // to "]" or "]]" instead of the standard "]]>". The parser must still
        // recover the inner value rather than leak the wrapper.
        for bad_ending in ["]", "]]"] {
            let xml = format!(
                "<function name=\"write_file\"><arguments>\
                 <param name=\"content\"><![CDATA[hello world{bad_ending}</param>\
                 </arguments></function>"
            );
            let out = run(&xml);
            assert_eq!(out.len(), 1, "ending {:?}: got {:?}", bad_ending, out);
            assert!(
                out[0].contains("hello world"),
                "ending {:?}: content lost: {}",
                bad_ending,
                out[0]
            );
            assert!(
                !out[0].contains("CDATA") && !out[0].contains("]\""),
                "ending {:?}: wrapper leaked: {}",
                bad_ending,
                out[0]
            );
        }
    }

    #[test]
    fn functionname_quirk() {
        let out = run(
            "<functionname=\"f\"><arguments><param name=\"k\">v</param></arguments></function>",
        );
        assert_eq!(out, vec!["C:f({\"k\":\"v\"})"]);
    }

    #[test]
    fn bare_less_than_is_text() {
        // A bare '<' with no matching tag flushes as text. The scanner splits
        // at '<' (it can't know yet the '<' isn't a tag start), then flushes
        // the remainder once no '>' follows — two TextDeltas that concatenate
        // to the original. Both reach the agent as plain text (no tool call).
        let out = run("1 < 2");
        let joined: String = out
            .iter()
            .filter_map(|s| s.strip_prefix("T:"))
            .collect();
        assert_eq!(joined, "1 < 2");
    }

    #[test]
    fn streamed_byte_by_byte_matches_whole() {
        let xml = "<function name=\"f\"><arguments><param name=\"k\">v</param></arguments></function>";
        let mut r1 = XmlReplayer::new();
        let mut o1 = r1.push(xml);
        o1.extend(r1.finish());
        let mut r2 = XmlReplayer::new();
        let mut o2 = Vec::new();
        for c in xml.chars() {
            o2.extend(r2.push(&c.to_string()));
        }
        o2.extend(r2.finish());
        assert_eq!(
            o1.into_iter()
                .map(|f| match f {
                    Flush::ToolCall { name, arguments } => format!("{}{}", name, arguments),
                    _ => "?".into(),
                })
                .collect::<Vec<_>>(),
            o2.into_iter()
                .map(|f| match f {
                    Flush::ToolCall { name, arguments } => format!("{}{}", name, arguments),
                    _ => "?".into(),
                })
                .collect::<Vec<_>>(),
        );
    }
}
