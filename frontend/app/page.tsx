"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnimatePresence, motion } from "framer-motion";

type Mode = "auto" | "local" | "general";

type Source = {
  label: string;
  doc_path: string;
  chunk_index: number;
  score?: number | null;
};

type AskResponse = {
  answer: string;
  mode: string;
  task?: string;
  sources?: Source[];
  used_tools?: string[];
  model?: string;
  routing_reason?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  mode?: string;
  task?: string;
  model?: string;
  sources?: Source[];
  error?: boolean;
};

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function uid(prefix = "m") {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const chatRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  const canAsk = useMemo(() => question.trim().length > 0 && !loading, [question, loading]);

  async function ask() {
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    setError("");

    const userMsg: ChatMessage = {
      id: uid("u"),
      role: "user",
      content: q,
      createdAt: Date.now(),
      mode,
    };

    const assistantId = uid("a");
    const pendingAssistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "Thinking…",
      createdAt: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg, pendingAssistant]);
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, mode, top_k: 10 }),
      });

      const data: AskResponse = await res.json();

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
              ...m,
              content: data.answer || "(no answer)",
              sources: data.sources || [],
              mode: data.mode,
              task: data.task,
              model: data.model,
            }
            : m
        )
      );
    } catch {
      const msg = "Could not reach backend. Is FastAPI running on http://localhost:8000 ?";
      setError(msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
              ...m,
              content: msg,
              error: true,
            }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter newline
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  function clearChat() {
    setMessages([]);
    setError("");
  }

  return (
    <div
      className="app-shell relative min-h-screen bg-[#0d1218] text-slate-100 overflow-y-scroll flex flex-col"
      style={{ scrollbarGutter: "stable" }}
    >
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-32 -left-32 w-130 rounded-full blur-3xl opacity-22 bg-linear-to-r from-[#113042] via-[#0f2330] to-[#0b141e]" />
        <div className="absolute -bottom-40 -right-40 h-155 w-155 rounded-full blur-3xl opacity-16 bg-linear-to-r from-[#0c1a26] via-[#0a131c] to-[#05090f]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.015),transparent_60%)]" />
        <div className="absolute inset-0 opacity-[0.06] bg-[radial-gradient(circle_at_20%_20%,rgba(45,212,191,0.1),transparent_25%),radial-gradient(circle_at_80%_12%,rgba(125,211,252,0.1),transparent_22%),radial-gradient(circle_at_40%_78%,rgba(94,234,212,0.1),transparent_28%)]" />
        <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(0deg,rgba(255,255,255,0.08)_1px,transparent_1px)] bg-size-[120px_120px]" />
      </div>

      <div className="absolute top-4 left-4 z-20 flex items-center gap-3">
        <div className="orb-container static-orb w-8 h-8">
          <div className="orb inner-anim" />
        </div>
        <div className="flex flex-col">
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-2xl font-semibold jarvis-anim leading-tight"
          >
            Jarvis
          </motion.div>
          <p className="text-xs text-slate-400">Local + general AI</p>
        </div>
      </div>

      <div className="relative mx-auto max-w-5xl px-4 pt-2 pb-4 flex-1 flex flex-col gap-3 min-h-0">
        {/* error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-2xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-300 backdrop-blur"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* chat history above prompt */}
        <div
          className="mx-auto w-full max-w-5xl rounded-3xl border border-slate-800/60 bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a] p-4 mt-4 flex-1 min-h-80 min-w-0 overflow-y-auto scroll-smooth shadow-[0_18px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl"
          ref={chatRef}
          style={{ scrollbarGutter: "stable" }}
        >
          <div className="space-y-3">
            <AnimatePresence>
              {messages.map((m) => (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className={cx(
                    "rounded-3xl border border-slate-800/60 backdrop-blur-xl shadow-[0_18px_80px_rgba(0,0,0,0.35)] overflow-hidden",
                    m.role === "user"
                      ? "ml-auto bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a]"
                      : "mr-auto bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a]"
                  )}
                >
                  <div className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={cx(
                          "h-2.5 w-2.5 rounded-full shadow-[0_0_18px_rgba(56,189,248,0.35)]",
                          m.role === "user"
                            ? "bg-linear-to-r from-emerald-300 to-cyan-400"
                            : "bg-linear-to-r from-slate-300 to-slate-500"
                        )}
                      />
                      <div className="font-semibold">
                        {m.role === "user" ? "You" : "Jarvis"}
                      </div>
                      {m.role === "assistant" && (m.mode || m.task) && (
                        <span className="text-xs text-slate-400">
                          {m.mode ? m.mode : ""}
                          {m.mode && m.task ? " • " : ""}
                          {m.task ? m.task : ""}
                        </span>
                      )}
                    </div>

                    {m.role === "assistant" && !m.error && m.content && m.content !== "Thinking…" && (
                      <button
                        className="rounded-xl border border-slate-800/70 bg-slate-950/40 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-950/70 transition"
                        onClick={() => navigator.clipboard.writeText(m.content)}
                        title="Copy answer"
                      >
                        Copy
                      </button>
                    )}
                  </div>

                  <div className="px-5 pb-5">
                    {m.role === "assistant" ? (
                      <div className="prose prose-invert max-w-none leading-relaxed prose-p:my-3 prose-li:my-1">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap text-sm text-slate-100 leading-relaxed">
                        {m.content}
                      </div>
                    )}

                    {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                      <div className="mt-5">
                        <details className="group">
                          <summary className="cursor-pointer list-none flex items-center justify-between rounded-2xl border border-slate-800/60 bg-slate-950/50 px-4 py-3 hover:bg-slate-950/70 transition">
                            <span className="text-sm font-semibold">
                              Sources ({m.sources.length})
                            </span>
                            <span className="text-xs text-zinc-400 group-open:rotate-180 transition-transform">
                              ▾
                            </span>
                          </summary>

                          <div className="mt-3 grid gap-2">
                            {m.sources.map((s) => (
                              <div
                                key={`${m.id}-${s.doc_path}-${s.chunk_index}-${s.label}`}
                                className="rounded-2xl border border-slate-800/60 bg-slate-950/50 px-4 py-3"
                              >
                                <div className="flex items-center justify-between">
                                  <div className="text-sm font-semibold">{s.label}</div>
                                  {typeof s.score === "number" && (
                                    <div className="text-xs text-zinc-400">
                                      score {s.score.toFixed(3)}
                                    </div>
                                  )}
                                </div>
                                <div className="mt-1 text-xs text-zinc-300 break-all">
                                  {s.doc_path}
                                </div>
                                <div className="mt-1 text-xs text-zinc-400">
                                  chunk {s.chunk_index}
                                </div>
                              </div>
                            ))}
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* input at bottom */}
        <div className="rounded-3xl h-35 border border-slate-800/60 bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a] backdrop-blur-xl shadow-[0_18px_80px_rgba(0,0,0,0.45)] mt-2 mb-1 mx-auto w-full max-w-5xl">
          <div className="p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="relative">
                <select
                  className="select-modern appearance-none rounded-xl border border-slate-800/70 bg-linear-to-r from-[#0f1823] via-[#0f1c28] to-[#0b1420] px-4 py-2 pr-9 text-sm font-medium text-slate-100 shadow-sm focus:outline-none focus:ring-0 focus:border-cyan-500/40 hover:border-cyan-400/50 transition"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as Mode)}
                >
                  <option className="bg-[#0f1620] text-slate-100" value="auto">
                    Auto
                  </option>
                  <option className="bg-[#0f1620] text-slate-100" value="local">
                    Local
                  </option>
                  <option className="bg-[#0f1620] text-slate-100" value="general">
                    General
                  </option>
                </select>
                <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-cyan-200">
                  ▾
                </span>
              </div>

              <div className="text-xs text-zinc-500">{question.trim().length} chars</div>
              <button
                onClick={clearChat}
                className="rounded-xl border border-slate-800/70 bg-slate-950/40 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-950/70 transition"
                title="Clear chat"
              >
                Clear
              </button>
            </div>

            <div className="mt-2 flex items-end gap-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder='Try: "Summarize my resume in 3 bullets" or "How do computers work?"'
                className="flex-1 min-h-16 w-200 h-16 resize-y rounded-2xl border border-slate-800/70 bg-[#0d131c] px-4 py-3 text-sm leading-relaxed transition outline-none focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 focus:border-slate-800/70"
              />
              <button
                onClick={ask}
                disabled={!canAsk}
                className={cx(
                  "rounded-2xl px-4 py-2 h-14 mb-1 text-sm font-semibold transition relative overflow-hidden",
                  canAsk
                    ? "text-white shadow-lg shadow-slate-900/40"
                    : "bg-slate-800 text-slate-500 cursor-not-allowed"
                )}
              >
                {canAsk && (
                  <span className="absolute inset-0 bg-linear-to-r from-cyan-600 via-cyan-500 to-emerald-400 opacity-90" />
                )}
                <span className={cx("relative", canAsk ? "drop-shadow" : "")}>
                  {loading ? "Thinking…" : "Ask"}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
