"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  FileText,
  LoaderCircle,
  MessageSquareQuote,
  Newspaper,
  ScanSearch,
  Send,
  ShieldCheck,
  Upload,
} from "lucide-react";

import {
  ChatResponse,
  ChatTurn,
  ContextChunk,
  sendChatMessage,
  uploadDocument,
} from "../lib/api";

type AssistantMessage = {
  role: "assistant";
  content: string;
  contexts: ContextChunk[];
  usage: ChatResponse["usage"];
};

type Message = ChatTurn | AssistantMessage;

const LOADER_STEPS = [
  "Parsing edition",
  "Cutting columns",
  "Embedding fragments",
  "Publishing to hybrid index",
];

const STATUS_ITEMS = [
  "Kimi-K2.5 on Azure AI Foundry",
  "Azure AI Search hybrid retrieval",
  "No external model switchers",
  "Upload once, chat immediately",
];

function formatEditionDate(): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "2-digit",
    year: "numeric",
  }).format(new Date());
}

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [chunkCount, setChunkCount] = useState<number>(0);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [latestContexts, setLatestContexts] = useState<ContextChunk[]>([]);
  const [processing, setProcessing] = useState(false);
  const [chatting, setChatting] = useState(false);
  const [error, setError] = useState<string>("");
  const [loaderIndex, setLoaderIndex] = useState(0);

  useEffect(() => {
    if (!processing) {
      setLoaderIndex(0);
      return;
    }

    const interval = window.setInterval(() => {
      setLoaderIndex((current) => (current + 1) % LOADER_STEPS.length);
    }, 900);

    return () => window.clearInterval(interval);
  }, [processing]);

  async function handleUpload() {
    if (!selectedFile) {
      setError("Choose a PDF or TXT file first.");
      return;
    }

    setProcessing(true);
    setError("");

    try {
      const response = await uploadDocument(selectedFile);
      setSessionId(response.session_id);
      setFileName(response.file_name);
      setChunkCount(response.chunk_count);
      setMessages([
        {
          role: "assistant",
          content:
            "Your file is indexed in Azure AI Search. Ask a question and I will answer from the uploaded material only.",
          contexts: [],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
        },
      ]);
      setLatestContexts([]);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed.",
      );
    } finally {
      setProcessing(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!sessionId || !question.trim() || chatting) {
      return;
    }

    const nextQuestion = question.trim();
    const nextMessages = [...messages, { role: "user", content: nextQuestion } satisfies ChatTurn];
    setMessages(nextMessages);
    setQuestion("");
    setChatting(true);
    setError("");

    try {
      const history = nextMessages.map((message) => ({
        role: message.role,
        content: message.content,
      })) as ChatTurn[];

      const response = await sendChatMessage(sessionId, nextQuestion, history);
      const assistantMessage: AssistantMessage = {
        role: "assistant",
        content: response.answer,
        contexts: response.contexts,
        usage: response.usage,
      };
      setMessages([...nextMessages, assistantMessage]);
      setLatestContexts(response.contexts);
    } catch (chatError) {
      setMessages(messages);
      setError(chatError instanceof Error ? chatError.message : "Chat failed.");
    } finally {
      setChatting(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="masthead">
        <div className="masthead-row">
          <span className="edition-meta">Vol. 1 | {formatEditionDate()} | CA Edition</span>
          <span className="edition-meta">Hybrid Retrieval Bulletin</span>
        </div>
        <div className="masthead-grid">
          <div className="headline-panel">
            <p className="section-label">Special Report</p>
            <h1 className="hero-title">CA-RAG</h1>
            <p className="hero-deck">
              A stripped-down newsroom for document ingestion and grounded chat,
              backed only by Azure AI Foundry and Azure AI Search.
            </p>
          </div>
          <div className="sidebar-panel inverted-panel">
            <p className="section-label section-label-light">Desk Notes</p>
            <div className="feature-stack">
              <div className="feature-line">
                <ShieldCheck size={18} />
                <span>No external model controls</span>
              </div>
              <div className="feature-line">
                <ScanSearch size={18} />
                <span>Hybrid vector + keyword retrieval</span>
              </div>
              <div className="feature-line">
                <MessageSquareQuote size={18} />
                <span>Single upload-to-chat flow</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section className="ticker-strip" aria-label="Status ticker">
        <div className="ticker-track">
          {[...STATUS_ITEMS, ...STATUS_ITEMS].map((item, index) => (
            <span className="ticker-item" key={`${item}-${index}`}>
              {item}
            </span>
          ))}
        </div>
      </section>

      <section className="workspace-grid">
        <article className="panel newsprint-texture">
          <div className="panel-header">
            <p className="section-label">Document Desk</p>
            <FileText size={18} />
          </div>
          <h2 className="panel-title">Load One File, Then Stay in Chat</h2>
          <p className="body-copy dropcap">
            Upload a PDF or TXT file. The backend parses it, chunks it, embeds it
            through Azure, and publishes the document fragments into a hybrid Azure
            Search index scoped to your session.
          </p>

          <label className="upload-frame" htmlFor="document-upload">
            <input
              id="document-upload"
              className="hidden-input"
              type="file"
              accept=".pdf,.txt"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
            <Upload size={26} />
            <div>
              <p className="upload-title">
                {selectedFile ? selectedFile.name : "Choose PDF or TXT"}
              </p>
              <p className="upload-caption">
                Max quality comes from clean, extractable text.
              </p>
            </div>
          </label>

          <button
            className="primary-button"
            onClick={handleUpload}
            disabled={!selectedFile || processing}
            type="button"
          >
            {processing ? (
              <>
                <LoaderCircle className="spin" size={18} />
                Processing
              </>
            ) : (
              <>
                <Upload size={18} />
                Index File
              </>
            )}
          </button>

          <div className="status-board">
            <div className="status-item">
              <span className="status-label">Current File</span>
              <span className="status-value">{fileName || "None loaded"}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Session</span>
              <span className="status-value mono">
                {sessionId ? sessionId.slice(0, 8) : "Not started"}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Indexed Chunks</span>
              <span className="status-value mono">{chunkCount || 0}</span>
            </div>
          </div>

          <div className="loader-rail">
            {LOADER_STEPS.map((step, index) => (
              <div
                className={`loader-step ${processing && index <= loaderIndex ? "loader-step-active" : ""}`}
                key={step}
              >
                <span className="loader-index mono">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span>{step}</span>
              </div>
            ))}
          </div>

          {error ? <p className="error-banner">{error}</p> : null}
        </article>

        <article className="panel chat-panel">
          <div className="panel-header">
            <p className="section-label">Conversation Wire</p>
            <Newspaper size={18} />
          </div>
          <h2 className="panel-title">Grounded Answers Only</h2>

          <div className="message-feed">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p className="section-label">Front Page</p>
                <p className="body-copy">
                  After indexing, every answer stays tied to the uploaded file and
                  returns the supporting snippets used for retrieval.
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <section className="message-card" key={`${message.role}-${index}`}>
                  <div className="message-meta">
                    <span className="section-label">
                      {message.role === "user" ? "Reporter" : "Desk Reply"}
                    </span>
                    {message.role === "assistant" && "usage" in message ? (
                      <span className="edition-meta mono">
                        {message.usage.total_tokens} tokens
                      </span>
                    ) : null}
                  </div>
                  <p className="message-copy">{message.content}</p>
                  {message.role === "assistant" &&
                  "contexts" in message &&
                  message.contexts.length > 0 ? (
                    <details className="sources-panel">
                      <summary>Evidence deck</summary>
                      <div className="sources-list">
                        {message.contexts.map((context) => (
                          <article className="source-card" key={`${context.file_name}-${context.chunk_id}`}>
                            <p className="section-label">
                              Fig. {context.chunk_id + 1} | {context.file_name}
                            </p>
                            <p className="source-copy">{context.content}</p>
                          </article>
                        ))}
                      </div>
                    </details>
                  ) : null}
                </section>
              ))
            )}
          </div>

          <form className="chat-form" onSubmit={handleSubmit}>
            <textarea
              className="chat-input"
              placeholder={
                sessionId
                  ? "Ask a question about the indexed document..."
                  : "Index a file to unlock chat."
              }
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={!sessionId || chatting}
              rows={4}
            />
            <button
              className="primary-button"
              type="submit"
              disabled={!sessionId || !question.trim() || chatting}
            >
              {chatting ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Filing Answer
                </>
              ) : (
                <>
                  <Send size={18} />
                  Send to Desk
                </>
              )}
            </button>
          </form>
        </article>

        <aside className="panel side-column">
          <div className="panel-header">
            <p className="section-label">Retrieval Notes</p>
            <ScanSearch size={18} />
          </div>
          <h2 className="panel-title">Latest Source Material</h2>
          <p className="body-copy">
            The hybrid retriever blends keyword search with vector similarity in the
            same Azure Search query so follow-up questions stay fast without adding
            provider sprawl.
          </p>
          <div className="facts-grid">
            <div className="fact-card">
              <span className="status-label">LLM</span>
              <span className="status-value">Kimi-K2.5</span>
            </div>
            <div className="fact-card">
              <span className="status-label">Embeddings</span>
              <span className="status-value">embed-v-4-0</span>
            </div>
            <div className="fact-card">
              <span className="status-label">Store</span>
              <span className="status-value">Azure AI Search</span>
            </div>
            <div className="fact-card">
              <span className="status-label">Surface</span>
              <span className="status-value">Upload + chat only</span>
            </div>
          </div>

          <div className="sources-list latest-contexts">
            {latestContexts.length === 0 ? (
              <p className="edition-meta">
                No retrieval snippets yet. They appear here after the first answer.
              </p>
            ) : (
              latestContexts.map((context) => (
                <article className="source-card" key={`${context.file_name}-${context.chunk_id}-latest`}>
                  <p className="section-label">
                    Snippet {context.chunk_id + 1} | {context.file_name}
                  </p>
                  <p className="source-copy">{context.content}</p>
                </article>
              ))
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
