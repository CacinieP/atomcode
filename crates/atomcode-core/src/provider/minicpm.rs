//! MiniCPM5 provider — bridges OpenBMB MiniCPM5-1B's text/XML tool-call
//! format into atomcode's `StreamEvent::ToolCall*` stream.
//!
//! ## Why this exists
//!
//! MiniCPM5-1B does NOT emit OpenAI-style `tool_calls` JSON. When served
//! via Ollama / llama.cpp / vLLM-without-parser, it emits tool calls as
//! inline XML *text* inside `content`:
//!
//! ```xml
//! <function name="get_weather">
//!   <arguments>
//!     <param name="city">London</param>
//!   </arguments>
//! </function>
//! ```
//!
//! atomcode's agent loop only reacts to `StreamEvent::ToolCallStart/Delta/Done`
//! — a plain-text `<function>` block is shown to the user as prose and the
//! loop never advances (model "talks about" calling a tool instead of
//! calling it). This provider intercepts the streamed `content`, peels off
//! `<function>` blocks, and re-emits them as proper tool-call events so the
//! kernel agent loop drives tools exactly as it does for OpenAI/Claude.
//!
//! The XML format and the `\u0120` / CDATA / `<functionname=` quirks handled
//! below are ported from OpenBMB's reference parser
//! `tool_parsers/minicpm5xml_tool_parser.py` (Apache-2.0). We keep the
//! behaviour faithful to that parser so any prompt-template change upstream
//! stays compatible.
//!
//! ## Hybrid reasoning
//!
//! MiniCPM5's `<think>…</think>` blocks are re-emitted as
//! `StreamEvent::Reasoning` so they render in the thinking channel rather
//! than polluting the assistant text or the tool-call scanner.

use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::conversation::message::{Message, MessageContent, Role};
use crate::stream::StreamEvent;
use crate::tool::ToolDef;
use atomcode_config::config::provider::ProviderConfig;

use super::LlmProvider;

pub struct MinicpmProvider {
    client: Client,
    model: String,
    base_url: String,
}

impl MinicpmProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self> {
        Ok(Self {
            client: super::build_http_client(
                config.user_agent.as_deref(),
                config.skip_tls_verify,
                false,
            )?,
            model: config.model.clone(),
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string()),
        })
    }

    /// Format conversation into Ollama `/api/chat` message array.
    ///
    /// MiniCPM5 only consumes `content` text — historical tool calls must be
    /// rendered back as the model's own XML format so it sees a consistent
    /// transcript. We emit the same `<function>…</function>` shape the model
    /// itself produces, which is what the chat template was trained on.
    fn format_messages(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .filter_map(|m| match &m.content {
                MessageContent::Text(s) => {
                    let role = match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        // Tool results carry structured output — handled below.
                        Role::Tool => return None,
                    };
                    if s.trim().is_empty() {
                        return None;
                    }
                    Some(json!({"role": role, "content": s}))
                }
                MessageContent::AssistantWithToolCalls { text, tool_calls, .. } => {
                    // Re-render this assistant turn the way MiniCPM5 would have
                    // produced it: optional preamble text, then one
                    // `<function>` block per call. This keeps the model's view
                    // of its own history consistent with its training format.
                    let mut content = text.as_deref().unwrap_or("").to_string();
                    for tc in tool_calls {
                        content.push_str(&render_function_xml(&tc.name, &tc.arguments));
                    }
                    if content.trim().is_empty() {
                        return None;
                    }
                    Some(json!({"role": "assistant", "content": content}))
                }
                MessageContent::ToolResult(r) => Some(json!({
                    "role": "tool",
                    "content": r.output,
                })),
                MessageContent::ToolResultRef(r) => Some(json!({
                    "role": "tool",
                    "content": r.summary,
                })),
                MessageContent::MultiPart { text, .. } => {
                    let t = text.as_deref().unwrap_or("");
                    if t.is_empty() {
                        return None;
                    }
                    Some(json!({"role": "user", "content": t}))
                }
            })
            .collect()
    }
}

/// Render a tool call as the MiniCPM5 `<function>` XML the model emits.
///
/// `arguments` is the JSON string atomcode carries (e.g. `{"city":"London"}`).
/// We expand it back into `<param name="…">value</param>` children so the
/// round-trip (model→parser→history→model) is byte-faithful to training.
fn render_function_xml(name: &str, arguments_json: &str) -> String {
    let args: serde_json::Value = serde_json::from_str(arguments_json).unwrap_or(json!({}));
    let mut params = String::new();
    if let Some(obj) = args.as_object() {
        for (k, v) in obj {
            // Strings go in raw; everything else is JSON-encoded so the value
            // survives the round-trip through the parser's `json.loads` path.
            let val = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            params.push_str(&format!("<param name=\"{k}\">{val}</param>"));
        }
    }
    format!("<function name=\"{name}\"><arguments>{params}</arguments></function>")
}

#[derive(Deserialize)]
struct OllamaChunk {
    message: Option<OllamaMessage>,
    done: bool,
    #[serde(default)]
    prompt_eval_count: usize,
    #[serde(default)]
    eval_count: usize,
}

#[derive(Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    content: String,
}

/// Streaming scanner that peels MiniCPM5's inline XML out of the `content`
/// text channel and re-emits it as tool-call / reasoning events.
///
/// The model emits three kinds of markup inline:
///   1. `<think>…</think>`        → `Reasoning`
///   2. `<function …>…</function>` → `ToolCallStart/Delta/Done`
///   3. everything else            → `Delta` (assistant text)
///
/// Because Ollama streams token-by-token, a tag can be split across many
/// chunks. We hold back any text from `<` onward until we can either (a)
/// match a known opening tag, or (b) rule it out and flush it as plain
/// text. This is the same "buffer the ambiguous prefix" trick the Python
/// parser uses, just over a stream instead of a full string.
struct XmlReplayer {
    /// Accumulated text not yet decided — may be start of a tag or plain text.
    pending: String,
    /// When inside `<think>`, reasoning text not yet closed.
    in_think: bool,
    /// When inside `<function …`, the raw XML up to `</function>`.
    in_function: bool,
    /// Monotonic id counter for `call_N`.
    tool_call_counter: u32,
}

enum Flush {
    /// Send this as assistant text right away.
    Delta(String),
    /// Enter/exit a think block; emit accumulated reasoning.
    Reasoning(String),
    /// A complete `<function>` block was captured — emit a full tool call.
    ToolCall { name: String, arguments: String },
}

impl XmlReplayer {
    fn new() -> Self {
        Self {
            pending: String::new(),
            in_think: false,
            in_function: false,
            tool_call_counter: 0,
        }
    }

    fn push(&mut self, chunk: &str, out: &mut Vec<Flush>) {
        self.pending.push_str(chunk);
        self.drain(out);
    }

    fn finish(&mut self, out: &mut Vec<Flush>) {
        // Flush any remaining buffered text as plain delta/reasoning. At
        // end-of-stream there can be no half-open tag we still care about:
        // a truncated `<funct` with no closing `>` is just text.
        if self.in_think {
            // Unterminated <think> at EOS — treat the remainder as reasoning.
            out.push(Flush::Reasoning(std::mem::take(&mut self.pending)));
            self.in_think = false;
        } else if self.in_function {
            // Truncated function block — best effort: try to parse what we
            // have so the agent loop still gets a (possibly partial) call
            // rather than silently dropping it.
            if let Some((name, args)) = parse_function_block(&self.pending) {
                out.push(Flush::ToolCall { name, arguments: args });
            } else {
                out.push(Flush::Delta(std::mem::take(&mut self.pending)));
            }
            self.in_function = false;
        } else if !self.pending.is_empty() {
            out.push(Flush::Delta(std::mem::take(&mut self.pending)));
        }
    }

    fn drain(&mut self, out: &mut Vec<Flush>) {
        loop {
            if self.in_think {
                // Look for the closing </think>. Everything before it is
                // reasoning; everything after resumes plain-text scanning.
                if let Some(end) = self.pending.find("</think>") {
                    let body = self.pending[..end].to_string();
                    self.pending = self.pending[end + "</think>".len()..].to_string();
                    if !body.is_empty() {
                        out.push(Flush::Reasoning(body));
                    }
                    self.in_think = false;
                    continue;
                }
                // Not closed yet — but we can still stream out the part that
                // can't possibly contain the closing tag (keep the last
                // `len("</think>")-1` chars in case the split lands mid-tag).
                let keep = "</think>".len() - 1;
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
                // Accumulate until we have the full `</function>`.
                if let Some(end) = self.pending.find("</function>") {
                    let block = self.pending[..end].to_string();
                    self.pending = self.pending[end + "</function>".len()..].to_string();
                    if let Some((name, args)) = parse_function_block(&block) {
                        out.push(Flush::ToolCall { name, arguments: args });
                    }
                    self.in_function = false;
                    continue;
                }
                return;
            }

            // Not inside any block. Scan for the next opening tag.
            match self.pending.find('<') {
                None => {
                    // No tag start — flush everything as text.
                    if !self.pending.is_empty() {
                        out.push(Flush::Delta(std::mem::take(&mut self.pending)));
                    }
                    return;
                }
                Some(lt) => {
                    // Text before '<' is safe assistant output.
                    if lt > 0 {
                        let text: String = self.pending.drain(..lt).collect();
                        out.push(Flush::Delta(text));
                    }
                    // Now self.pending starts with '<'. Decide what tag it is.
                    // We need enough bytes to read the tag name — if the chunk
                    // ended mid-tag, wait for more.
                    let after = &self.pending[1..];
                    if after.starts_with("think") {
                        // `<think>` (possibly followed by '>' or '/'). Strip
                        // up to and including the first '>'.
                        if let Some(gt) = self.pending.find('>') {
                            self.pending = self.pending[gt + 1..].to_string();
                            self.in_think = true;
                            continue;
                        }
                        return; // need more bytes
                    }
                    if after.starts_with("function") || after.starts_with("functionname=") {
                        // Wait for the closing '>' of the opening tag, then
                        // switch to function-accumulation mode. The body up
                        // to `</function>` is parsed wholesale later.
                        if let Some(gt) = self.pending.find('>') {
                            // Keep the opening tag in the buffer so
                            // parse_function_block sees the full element.
                            self.in_function = true;
                            // Don't trim — leave pending as-is; the body
                            // accumulates below in the in_function branch.
                            let _ = gt; // just a marker; actual split happens above
                            continue;
                        }
                        return;
                    }
                    // Not a recognised tag. Could be `<` used as prose (rare
                    // but legal in code the model emits). If the next char
                    // is a letter we don't recognise, treat the '<' as text
                    // only once we're sure it isn't the start of one of our
                    // tags. Simplest robust rule: if we have a '>' ahead,
                    // it's some tag we don't care about → flush the whole
                    // thing as text; otherwise hold one byte and retry.
                    if let Some(gt) = self.pending.find('>') {
                        let text: String = self.pending.drain(..=gt).collect();
                        out.push(Flush::Delta(text));
                        continue;
                    }
                    // Ambiguous prefix — wait for more input.
                    return;
                }
            }
        }
    }
}

/// Parse a complete `<function …>…</function>` opening-tag-plus-body into
/// `(name, arguments_json)`. Mirrors OpenBMB's parser: name from the
/// `name="…"` attribute, params from `<param name="…">value</param>` (or
/// `<arguments><param/></arguments>`), values JSON-decoded when the schema
/// expects a non-string.
///
/// We don't have the schema here, so we keep values as strings when they
/// look like strings and JSON-parse anything that looks like a number /
/// bool / object — exactly the Python parser's `json.loads →
/// ast.literal_eval` fallback, minus `ast` (we use serde_json only).
fn parse_function_block(raw: &str) -> Option<(String, String)> {
    // Normalise tokenizer quirks the Python parser handles:
    //   - "\u0120" (Ġ, the SentencePiece space marker) → ' '
    //   - "<functionname=" (collapsed by the BPE) → "<function name="
    let normalised = raw.replace('\u{0120}', " ").replace("<functionname=", "<function name=");

    let open_end = normalised.find('>')?;
    let open_tag = &normalised[..open_end];
    // open_tag looks like `<function name="get_weather"` (no '>' yet)
    let name = extract_attr(open_tag, "name")?;

    let body = &normalised[open_end + 1..];
    // Body may contain `<arguments>…</arguments>` wrapper or direct params.
    let mut args = serde_json::Map::new();
    let mut rest = body;
    while let Some(pstart) = rest.find("<param") {
        let pname = extract_attr(&rest[pstart..], "name")?;
        // Value is between the first '>' after <param … and the next '</param>'.
        let tag_end = rest[pstart..].find('>')? + pstart;
        let val_start = tag_end + 1;
        let after = &rest[val_start..];
        let val_end = after.find("</param>")?;
        let mut raw_val = &after[..val_end];
        // CDATA unwrapping — code/file payloads can contain '<' and '&'.
        raw_val = raw_val.strip_prefix("<![CDATA[").and_then(|s| s.strip_suffix("]]>")).unwrap_or(raw_val);
        let decoded = decode_xml_entities(raw_val);
        // Heuristic typing: try JSON first (numbers/bools/objects), fall back
        // to plain string. This matches the Python parser's contract: string
        // schemas stay strings, everything else is parsed.
        let value = if decoded.trim().is_empty() {
            serde_json::Value::String(String::new())
        } else {
            serde_json::from_str::<serde_json::Value>(&decoded)
                .unwrap_or(serde_json::Value::String(decoded))
        };
        args.insert(pname, value);
        // Advance past this param.
        rest = &after[val_end + "</param>".len()..];
    }

    Some((name, serde_json::Value::Object(args).to_string()))
}

/// Pull `name="value"` out of a tag fragment. Handles double or single quotes.
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    if let Some(s) = tag.find(&needle) {
        let val_start = s + needle.len();
        let val_end = tag[val_start..].find('"')? + val_start;
        return Some(tag[val_start..val_end].to_string());
    }
    let needle = format!("{attr}='");
    if let Some(s) = tag.find(&needle) {
        let val_start = s + needle.len();
        let val_end = tag[val_start..].find('\'')? + val_start;
        return Some(tag[val_start..val_end].to_string());
    }
    None
}

/// Minimal XML entity decode — &lt; &gt; &amp; &quot; &apos;.
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

#[async_trait]
impl LlmProvider for MinicpmProvider {
    fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let url = format!("{}/api/chat", self.base_url);
        // We deliberately do NOT pass Ollama's `tools` field. MiniCPM5 was
        // trained to call tools from a text description in the system/user
        // prompt (its SGLang parser runs over `content`). Letting Ollama
        // inject its own tool-call template would collide with the model's
        // native format. atomcode already renders tool descriptions into the
        // system prompt, which is exactly what MiniCPM5 expects.
        let _ = tools;
        let body = json!({
            "model": self.model,
            "messages": Self::format_messages(messages),
            "stream": true,
            // MiniCPM5's hybrid-reasoning `<think>` is driven by the chat
            // template, not by options — we don't touch it here. Recommended
            // sampling (no-think): temperature 0.7, top_p 0.95.
            "options": {
                "temperature": 0.7,
                "top_p": 0.95,
            },
        });

        let request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        let policy = crate::provider::retry::RetryPolicy::default_policy();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let response = match crate::provider::retry::send_with_retry(request, &policy).await {
                Ok(resp) => resp,
                Err(e) => {
                    let _ = tx.send(Ok(StreamEvent::Error(format!("Connection failed: {}", e))));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let msg = super::extract_error_message(&body);
                let _ = tx.send(Ok(StreamEvent::Error(format!(
                    "MiniCPM error ({}): {}",
                    status, msg
                ))));
                return;
            }

            let mut byte_buffer: Vec<u8> = Vec::with_capacity(4096);
            let mut buffer = String::new();
            let mut byte_stream = response.bytes_stream();
            let mut replayer = XmlReplayer::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(bytes) => byte_buffer.extend_from_slice(&bytes),
                    Err(e) => {
                        let _ = tx.send(Ok(StreamEvent::Error(e.to_string())));
                        return;
                    }
                }

                let text = match String::from_utf8(byte_buffer.clone()) {
                    Ok(s) => {
                        byte_buffer.clear();
                        s
                    }
                    Err(e) => {
                        let valid_len = e.utf8_error().valid_up_to();
                        if valid_len == 0 {
                            continue;
                        }
                        let valid = String::from_utf8_lossy(&byte_buffer[..valid_len]).to_string();
                        byte_buffer = byte_buffer[valid_len..].to_vec();
                        valid
                    }
                };

                buffer.push_str(&text);

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(chunk) = serde_json::from_str::<OllamaChunk>(&line) else {
                        continue;
                    };

                    if let Some(ref msg) = chunk.message {
                        if msg.content.is_empty() {
                            continue;
                        }
                        let mut flushes = Vec::new();
                        replayer.push(&msg.content, &mut flushes);
                        for f in flushes {
                            match f {
                                Flush::Delta(s) => {
                                    let _ = tx.send(Ok(StreamEvent::Delta(s)));
                                }
                                Flush::Reasoning(s) => {
                                    let _ = tx.send(Ok(StreamEvent::Reasoning(s)));
                                }
                                Flush::ToolCall { name, arguments } => {
                                    replayer.tool_call_counter += 1;
                                    let id = format!("call_{}", replayer.tool_call_counter);
                                    let _ = tx.send(Ok(StreamEvent::ToolCallStart {
                                        id: id.clone(),
                                        name: name.clone(),
                                    }));
                                    let _ = tx
                                        .send(Ok(StreamEvent::ToolCallDelta(arguments.clone())));
                                    let _ = tx.send(Ok(StreamEvent::ToolCallDone(
                                        crate::tool::ToolCall {
                                            id,
                                            name,
                                            arguments,
                                        },
                                    )));
                                }
                            }
                        }
                    }

                    if chunk.done {
                        // End of stream — flush any buffered/partial markup.
                        let mut flushes = Vec::new();
                        replayer.finish(&mut flushes);
                        for f in flushes {
                            match f {
                                Flush::Delta(s) => {
                                    let _ = tx.send(Ok(StreamEvent::Delta(s)));
                                }
                                Flush::Reasoning(s) => {
                                    let _ = tx.send(Ok(StreamEvent::Reasoning(s)));
                                }
                                Flush::ToolCall { name, arguments } => {
                                    replayer.tool_call_counter += 1;
                                    let id = format!("call_{}", replayer.tool_call_counter);
                                    let _ = tx.send(Ok(StreamEvent::ToolCallStart {
                                        id: id.clone(),
                                        name: name.clone(),
                                    }));
                                    let _ = tx
                                        .send(Ok(StreamEvent::ToolCallDelta(arguments.clone())));
                                    let _ = tx.send(Ok(StreamEvent::ToolCallDone(
                                        crate::tool::ToolCall {
                                            id,
                                            name,
                                            arguments,
                                        },
                                    )));
                                }
                            }
                        }
                        if chunk.eval_count > 0 || chunk.prompt_eval_count > 0 {
                            let _ = tx.send(Ok(StreamEvent::Usage(crate::stream::TokenUsage {
                                prompt_tokens: chunk.prompt_eval_count,
                                completion_tokens: chunk.eval_count,
                                cached_tokens: 0,
                            })));
                        }
                        let _ = tx.send(Ok(StreamEvent::Done { truncated: false }));
                        return;
                    }
                }
            }

            // Stream ended without an explicit `done` chunk — still flush.
            let mut flushes = Vec::new();
            replayer.finish(&mut flushes);
            for f in flushes {
                match f {
                    Flush::Delta(s) => {
                        let _ = tx.send(Ok(StreamEvent::Delta(s)));
                    }
                    Flush::Reasoning(s) => {
                        let _ = tx.send(Ok(StreamEvent::Reasoning(s)));
                    }
                    Flush::ToolCall { name, arguments } => {
                        replayer.tool_call_counter += 1;
                        let id = format!("call_{}", replayer.tool_call_counter);
                        let _ = tx.send(Ok(StreamEvent::ToolCallStart {
                            id: id.clone(),
                            name: name.clone(),
                        }));
                        let _ = tx.send(Ok(StreamEvent::ToolCallDelta(arguments.clone())));
                        let _ = tx.send(Ok(StreamEvent::ToolCallDone(crate::tool::ToolCall {
                            id,
                            name,
                            arguments,
                        })));
                    }
                }
            }
            let _ = tx.send(Ok(StreamEvent::Done { truncated: false }));
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
        ))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(input: &str) -> Vec<String> {
        let mut r = XmlReplayer::new();
        let mut out = Vec::new();
        // Feed the whole string at once (non-streaming) — tests the parser
        // core. Streaming-split tests below cover chunk-boundary cases.
        r.push(input, &mut out);
        r.finish(&mut out);
        out.into_iter()
            .map(|f| match f {
                Flush::Delta(s) => format!("D:{}", s),
                Flush::Reasoning(s) => format!("R:{}", s),
                Flush::ToolCall { name, arguments } => format!("T:{}({})", name, arguments),
            })
            .collect()
    }

    #[test]
    fn plain_text_passes_through_as_delta() {
        let out = run("hello world");
        assert_eq!(out, vec!["D:hello world"]);
    }

    #[test]
    fn think_block_becomes_reasoning() {
        let out = run("<think>let me consider</think>done");
        assert_eq!(out, vec!["R:let me consider", "D:done"]);
    }

    #[test]
    fn function_block_becomes_tool_call() {
        let out = run(
            "<function name=\"get_weather\"><arguments>\
             <param name=\"city\">London</param>\
             </arguments></function>",
        );
        assert_eq!(out, vec!["T:get_weather({\"city\":\"London\"})"]);
    }

    #[test]
    fn mixed_text_think_and_function() {
        let input = "Sure. <think>checking</think>I'll look it up. \
                     <function name=\"read_file\"><arguments>\
                     <param name=\"path\">/tmp/a.rs</param>\
                     </arguments></function>";
        let out = run(input);
        assert_eq!(
            out,
            vec![
                "D:Sure. ",
                "R:checking",
                "D:I'll look it up. ",
                "T:read_file({\"path\":\"/tmp/a.rs\"})",
            ]
        );
    }

    #[test]
    fn number_param_is_json_typed() {
        let out = run(
            "<function name=\"edit\"><arguments>\
             <param name=\"line\">42</param>\
             </arguments></function>",
        );
        // Numbers parse to JSON numbers, not strings — matches Python parser.
        assert_eq!(out, vec!["T:edit({\"line\":42})"]);
    }

    #[test]
    fn cdata_payload_survives() {
        let xml = "<function name=\"write_file\"><arguments>\
                   <param name=\"content\"><![CDATA[fn main() { let x = 1 < 2; }]]></param>\
                   </arguments></function>";
        let out = run(xml);
        assert!(out.len() == 1);
        assert!(out[0].starts_with("T:write_file("));
        // The '<' inside the CDATA must survive intact.
        assert!(out[0].contains("1 < 2"));
    }

    #[test]
    fn functionname_quirk_is_normalised() {
        // Tokenizer-collapsed opening tag — the Python parser fixes this;
        // we must too or the whole block is missed.
        let out = run(
            "<functionname=\"read\"><arguments>\
             <param name=\"path\">x</param>\
             </arguments></function>",
        );
        assert_eq!(out, vec!["T:read({\"path\":\"x\"})"]);
    }

    // ── streaming: feed the input byte-by-byte to stress chunk boundaries ──
    fn run_streamed(input: &str) -> Vec<String> {
        let mut r = XmlReplayer::new();
        let mut out = Vec::new();
        for c in input.chars() {
            r.push(&c.to_string(), &mut out);
        }
        r.finish(&mut out);
        out.into_iter()
            .map(|f| match f {
                Flush::Delta(s) => format!("D:{}", s),
                Flush::Reasoning(s) => format!("R:{}", s),
                Flush::ToolCall { name, arguments } => format!("T:{}({})", name, arguments),
            })
            .collect()
    }

    #[test]
    fn function_block_survives_byte_streaming() {
        let xml = "<function name=\"f\"><arguments><param name=\"k\">v</param></arguments></function>";
        let streamed = run_streamed(xml);
        let whole = run(xml);
        assert_eq!(streamed, whole);
    }

    #[test]
    fn think_survives_byte_streaming() {
        let input = "a<think>bb</think>c";
        assert_eq!(run_streamed(input), run(input));
    }

    #[test]
    fn plain_less_than_sign_is_text() {
        // A bare '<' used as prose (e.g. in code output) must not be eaten.
        let out = run("1 < 2");
        assert_eq!(out, vec!["D:1 < 2"]);
    }
}
