"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  FilePlus2,
  FileText,
  Folder,
  FolderPlus,
  GitBranch,
  HardDriveUpload,
  LoaderCircle,
  MessageSquareText,
  ScanSearch,
  Send,
  Sparkles,
} from "lucide-react";

import {
  ChatResponse,
  ChatTurn,
  ContextChunk,
  HealthResponse,
  SessionResponse,
  WorkspaceDocument,
  createFolder,
  createWorkspace,
  getHealth,
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

type TreeNode = {
  name: string;
  path: string;
  type: "folder" | "file";
  children: TreeNode[];
  document?: WorkspaceDocument;
};

const ROOT_FOLDER = "";
const CONTEXT_PREVIEW_LENGTH = 240;

function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function joinFolderPath(parentPath: string, folderName: string): string {
  return [parentPath, folderName].filter(Boolean).join("/");
}

function buildContextPreview(content: string): { text: string; truncated: boolean } {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= CONTEXT_PREVIEW_LENGTH) {
    return {
      text: normalized,
      truncated: false,
    };
  }

  return {
    text: `${normalized.slice(0, CONTEXT_PREVIEW_LENGTH).trimEnd()}...`,
    truncated: true,
  };
}

function buildWorkspaceTree(session: SessionResponse | null): TreeNode[] {
  if (!session) {
    return [];
  }

  const roots: TreeNode[] = [];
  const folderNodes = new Map<string, TreeNode>();

  function ensureFolder(path: string): TreeNode | null {
    if (!path) {
      return null;
    }

    const existingNode = folderNodes.get(path);
    if (existingNode) {
      return existingNode;
    }

    const parts = path.split("/").filter(Boolean);
    const name = parts[parts.length - 1] ?? path;
    const parentPath = parts.slice(0, -1).join("/");
    const node: TreeNode = {
      name,
      path,
      type: "folder",
      children: [],
    };

    folderNodes.set(path, node);

    if (parentPath) {
      const parentNode = ensureFolder(parentPath);
      parentNode?.children.push(node);
    } else {
      roots.push(node);
    }

    return node;
  }

  const folderIndex = new Map<string, TreeNode>();
  for (const folderPath of session.folders) {
    const folderNode = ensureFolder(folderPath);
    if (folderNode) {
      folderIndex.set(folderPath, folderNode);
    }
  }

  for (const document of session.documents) {
    const fileNode: TreeNode = {
      name: document.file_name,
      path: document.file_path,
      type: "file",
      children: [],
      document,
    };

    if (document.folder_path) {
      const folderNode = folderIndex.get(document.folder_path) ?? ensureFolder(document.folder_path);
      if (folderNode) {
        folderNode.children.push(fileNode);
      }
      continue;
    }

    roots.push(fileNode);
  }

  return roots.sort((left, right) => {
    if (left.type !== right.type) {
      return left.type === "folder" ? -1 : 1;
    }
    return left.name.localeCompare(right.name);
  });
}

function sortTree(node: TreeNode): TreeNode {
  const sortedChildren = [...node.children]
    .map(sortTree)
    .sort((left, right) => {
      if (left.type !== right.type) {
        return left.type === "folder" ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    });

  return {
    ...node,
    children: sortedChildren,
  };
}

function TreeBranch({
  node,
  activeFolder,
  activeFilePath,
  expandedFolders,
  onToggleFolder,
  onSelectFolder,
}: {
  node: TreeNode;
  activeFolder: string;
  activeFilePath: string | null;
  expandedFolders: Set<string>;
  onToggleFolder: (folderPath: string) => void;
  onSelectFolder: (folderPath: string) => void;
}) {
  if (node.type === "file") {
    return (
      <button
        className={`tree-file ${activeFilePath === node.path ? "tree-item-active" : ""}`}
        onClick={() => onSelectFolder(node.document?.folder_path ?? ROOT_FOLDER)}
        type="button"
      >
        <FileText size={14} />
        <span>{node.name}</span>
      </button>
    );
  }

  const expanded = expandedFolders.has(node.path);

  return (
    <div className="tree-branch">
      <button
        className={`tree-folder ${activeFolder === node.path ? "tree-item-active" : ""}`}
        onClick={() => onSelectFolder(node.path)}
        type="button"
      >
        <span
          className="tree-disclosure"
          onClick={(event) => {
            event.stopPropagation();
            onToggleFolder(node.path);
          }}
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <Folder size={14} />
        <span>{node.name}</span>
      </button>

      {expanded && node.children.length > 0 ? (
        <div className="tree-children">
          {node.children.map((child) => (
            <TreeBranch
              key={child.path}
              node={child}
              activeFolder={activeFolder}
              activeFilePath={activeFilePath}
              expandedFolders={expandedFolders}
              onToggleFolder={onToggleFolder}
              onSelectFolder={onSelectFolder}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default function HomePage() {
  const [workspaceName, setWorkspaceName] = useState("Project Workspace");
  const [workspace, setWorkspace] = useState<SessionResponse | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [activeFolder, setActiveFolder] = useState(ROOT_FOLDER);
  const [newFolderName, setNewFolderName] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [latestContexts, setLatestContexts] = useState<ContextChunk[]>([]);
  const [processing, setProcessing] = useState(false);
  const [chatting, setChatting] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());

  useEffect(() => {
    let active = true;

    async function loadHealth() {
      try {
        const response = await getHealth();
        if (!active) {
          return;
        }
        setHealth(response);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setHealthError(
          loadError instanceof Error ? loadError.message : "Health check failed.",
        );
      }
    }

    void loadHealth();

    return () => {
      active = false;
    };
  }, []);

  const tree = useMemo(
    () => buildWorkspaceTree(workspace).map(sortTree),
    [workspace],
  );

  async function handleCreateWorkspace() {
    if (workspace) {
      return;
    }

    setError("");

    try {
      const nextWorkspace = await createWorkspace(workspaceName.trim() || "Project Workspace");
      setWorkspace(nextWorkspace);
      setExpandedFolders(new Set(nextWorkspace.folders));
    } catch (createError) {
      setError(
        createError instanceof Error
          ? createError.message
          : "Workspace creation failed.",
      );
    }
  }

  async function handleCreateFolder() {
    if (!workspace) {
      setError("Create a workspace before adding folders.");
      return;
    }

    const folderName = newFolderName.trim();
    if (!folderName) {
      setError("Enter a folder name first.");
      return;
    }

    setError("");

    try {
      const nextPath = joinFolderPath(activeFolder, folderName);
      const nextWorkspace = await createFolder(workspace.session_id, nextPath);
      setWorkspace(nextWorkspace);
      setExpandedFolders((current) => new Set([...current, nextPath, activeFolder].filter(Boolean)));
      setActiveFolder(nextPath);
      setNewFolderName("");
    } catch (folderError) {
      setError(
        folderError instanceof Error ? folderError.message : "Folder creation failed.",
      );
    }
  }

  async function handleUpload() {
    if (selectedFiles.length === 0) {
      setError("Choose one or more PDF or TXT files first.");
      return;
    }

    setProcessing(true);
    setError("");

    try {
      let currentWorkspace = workspace;
      const workspaceLabel = workspaceName.trim() || "Project Workspace";

      for (const [index, file] of selectedFiles.entries()) {
        currentWorkspace = await uploadDocument(file, {
          sessionId: currentWorkspace?.session_id,
          workspaceName: currentWorkspace ? undefined : workspaceLabel,
          folderPath: activeFolder,
        });

        if (index === 0 && messages.length === 0) {
          setMessages([
            {
              role: "assistant",
              content:
                "Workspace ready. Ask a question about the indexed files and I will answer from the retrieved chunks only.",
              contexts: [],
              usage: {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
              },
            },
          ]);
        }
      }

      if (currentWorkspace) {
        setWorkspace(currentWorkspace);
        setExpandedFolders(new Set(currentWorkspace.folders));
      }
      setSelectedFiles([]);
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
    if (!workspace || !question.trim() || chatting) {
      return;
    }

    const nextQuestion = question.trim();
    const previousMessages = messages;
    const nextMessages = [
      ...previousMessages,
      { role: "user", content: nextQuestion } satisfies ChatTurn,
    ];
    setMessages(nextMessages);
    setQuestion("");
    setChatting(true);
    setError("");

    try {
      const history = nextMessages.map((message) => ({
        role: message.role,
        content: message.content,
      })) as ChatTurn[];

      const response = await sendChatMessage(
        workspace.session_id,
        nextQuestion,
        history,
      );
      const assistantMessage: AssistantMessage = {
        role: "assistant",
        content: response.answer,
        contexts: response.contexts,
        usage: response.usage,
      };
      setMessages([...nextMessages, assistantMessage]);
      setLatestContexts(response.contexts);
    } catch (chatError) {
      setMessages(previousMessages);
      setError(chatError instanceof Error ? chatError.message : "Chat failed.");
    } finally {
      setChatting(false);
    }
  }

  const activeFilePath = latestContexts[0]?.file_path ?? null;
  const systemHealthy = health?.status === "ok";

  return (
    <main className="workspace-shell">
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark">CA</div>
          <div>
            <h1>CA-RAG</h1>
            <p>Workspace retrieval app with folder-aware evidence</p>
          </div>
        </div>

        <div className="topbar-meta">
          <span className={`status-pill ${systemHealthy ? "status-pill-live" : "status-pill-warn"}`}>
            <Sparkles size={14} />
            {health?.chat_deployment ?? "Model pending"}
          </span>
          <span className="status-pill">
            <ScanSearch size={14} />
            {health?.embedding_deployment ?? "Embedding pending"}
          </span>
          <span className="status-pill">
            <GitBranch size={14} />
            {workspace?.workspace_name ?? "No workspace"}
          </span>
        </div>
      </header>

      <section className="main-grid">
        <aside className="panel sidebar-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Workspace</p>
              <h2>Folders and files</h2>
            </div>
            <HardDriveUpload size={18} />
          </div>

          {!workspace ? (
            <div className="stack-block">
              <label className="field-label" htmlFor="workspace-name">
                Workspace name
              </label>
              <input
                id="workspace-name"
                className="text-input"
                value={workspaceName}
                onChange={(event) => setWorkspaceName(event.target.value)}
                placeholder="Project Workspace"
              />
              <button className="primary-button" onClick={handleCreateWorkspace} type="button">
                <FilePlus2 size={16} />
                Create workspace
              </button>
            </div>
          ) : (
            <div className="workspace-summary">
              <div className="summary-row">
                <span className="summary-label">Documents</span>
                <strong>{compactNumber(workspace.document_count)}</strong>
              </div>
              <div className="summary-row">
                <span className="summary-label">Chunks</span>
                <strong>{compactNumber(workspace.chunk_count)}</strong>
              </div>
              <div className="summary-row">
                <span className="summary-label">Target folder</span>
                <strong className="path-chip">
                  {activeFolder || "root"}
                </strong>
              </div>
            </div>
          )}

          <div className="stack-block">
            <label className="field-label" htmlFor="file-upload">
              Add files
            </label>
            <label className="upload-dropzone" htmlFor="file-upload">
              <input
                id="file-upload"
                className="hidden-input"
                type="file"
                accept=".pdf,.txt"
                multiple
                onChange={(event) => setSelectedFiles(Array.from(event.target.files ?? []))}
              />
              <FileText size={18} />
              <div>
                <strong>
                  {selectedFiles.length > 0
                    ? `${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""} selected`
                    : "Choose PDF or TXT files"}
                </strong>
                <span>Upload into {activeFolder || "root"}</span>
              </div>
            </label>
            <button
              className="primary-button"
              disabled={selectedFiles.length === 0 || processing}
              onClick={handleUpload}
              type="button"
            >
              {processing ? (
                <>
                  <LoaderCircle className="spin" size={16} />
                  Indexing files
                </>
              ) : (
                <>
                  <HardDriveUpload size={16} />
                  Add to workspace
                </>
              )}
            </button>
          </div>

          <div className="stack-block">
            <label className="field-label" htmlFor="folder-name">
              Create folder
            </label>
            <div className="inline-form">
              <input
                id="folder-name"
                className="text-input"
                value={newFolderName}
                onChange={(event) => setNewFolderName(event.target.value)}
                placeholder={activeFolder ? `${activeFolder}/new-folder` : "new-folder"}
              />
              <button className="secondary-button" onClick={handleCreateFolder} type="button">
                <FolderPlus size={16} />
              </button>
            </div>
          </div>

          <div className="tree-panel">
            <div className="tree-head">
              <span className="panel-eyebrow">Document tree</span>
              <button
                className={`tree-root-button ${activeFolder === ROOT_FOLDER ? "tree-item-active" : ""}`}
                onClick={() => setActiveFolder(ROOT_FOLDER)}
                type="button"
              >
                Root
              </button>
            </div>

            {tree.length === 0 ? (
              <div className="empty-card">
                <p>No folders or files yet.</p>
                <span>Create a workspace, add folders, and upload documents into the tree.</span>
              </div>
            ) : (
              <div className="tree-list">
                {tree.map((node) => (
                  <TreeBranch
                    key={node.path}
                    node={node}
                    activeFolder={activeFolder}
                    activeFilePath={activeFilePath}
                    expandedFolders={expandedFolders}
                    onToggleFolder={(folderPath) => {
                      setExpandedFolders((current) => {
                        const next = new Set(current);
                        if (next.has(folderPath)) {
                          next.delete(folderPath);
                        } else {
                          next.add(folderPath);
                        }
                        return next;
                      });
                    }}
                    onSelectFolder={setActiveFolder}
                  />
                ))}
              </div>
            )}
          </div>
        </aside>

        <section className="panel conversation-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Conversation</p>
              <h2>Grounded answers</h2>
            </div>
            <MessageSquareText size={18} />
          </div>

          <div className="conversation-meta">
            <span className="meta-pill">
              {health?.search_index ?? "Index pending"}
            </span>
            <span className="meta-pill">
              {workspace ? `${workspace.document_count} files in workspace` : "No workspace yet"}
            </span>
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-card empty-chat">
                <Bot size={18} />
                <p>Upload files into your workspace and ask across the indexed folders.</p>
                <span>
                  Answers stay grounded to retrieved chunks and show the exact file path for each supporting snippet.
                </span>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  className={`message-card ${message.role === "assistant" ? "message-card-assistant" : "message-card-user"}`}
                  key={`${message.role}-${index}`}
                >
                  <div className="message-meta">
                    <div className="message-role">
                      {message.role === "assistant" ? <Bot size={15} /> : <Send size={15} />}
                      <span>{message.role === "assistant" ? "Assistant" : "You"}</span>
                    </div>
                    {message.role === "assistant" && "usage" in message ? (
                      <span className="token-pill">
                        {compactNumber(message.usage.total_tokens)} tokens
                      </span>
                    ) : null}
                  </div>

                  {message.role === "assistant" ? (
                    <div className="markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="user-message">{message.content}</p>
                  )}

                  {message.role === "assistant" &&
                  "contexts" in message &&
                  message.contexts.length > 0 ? (
                    <details className="context-details">
                      <summary>Evidence</summary>
                      <div className="context-grid">
                        {message.contexts.map((context) => {
                          const preview = buildContextPreview(context.content);

                          return (
                            <article className="context-card" key={`${context.document_id}-${context.chunk_id}`}>
                              <div className="context-card-head">
                                <strong>{context.file_path}</strong>
                                <span>Chunk {context.chunk_id + 1}</span>
                              </div>
                              <p>{preview.text}</p>
                              {preview.truncated ? (
                                <span className="context-truncated">Preview only</span>
                              ) : null}
                            </article>
                          );
                        })}
                      </div>
                    </details>
                  ) : null}
                </article>
              ))
            )}
          </div>

          <form className="chat-form" onSubmit={handleSubmit}>
            <textarea
              className="chat-input"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                workspace
                  ? "Ask about a file, folder, section, or concept in the indexed workspace..."
                  : "Create a workspace and upload files first."
              }
              disabled={!workspace || chatting}
              rows={4}
            />
            <button
              className="primary-button"
              disabled={!workspace || !question.trim() || chatting}
              type="submit"
            >
              {chatting ? (
                <>
                  <LoaderCircle className="spin" size={16} />
                  Generating
                </>
              ) : (
                <>
                  <Send size={16} />
                  Ask
                </>
              )}
            </button>
          </form>
        </section>

        <aside className="panel inspector-panel">
          <div className="panel-header">
            <div>
              <p className="panel-eyebrow">Inspector</p>
              <h2>Model and retrieval</h2>
            </div>
            <ScanSearch size={18} />
          </div>

          <div className="stat-stack">
            <div className="stat-card">
              <span className="summary-label">Backend</span>
              <strong>{health?.status ?? "Loading..."}</strong>
              <p>{healthError || "Health endpoint is connected."}</p>
            </div>
            <div className="stat-card">
              <span className="summary-label">Chat model</span>
              <strong>{health?.chat_deployment ?? "Pending"}</strong>
              <p>Live answer generation route.</p>
            </div>
            <div className="stat-card">
              <span className="summary-label">Embedding model</span>
              <strong>{health?.embedding_deployment ?? "Pending"}</strong>
              <p>Chunk vector generation route.</p>
            </div>
          </div>

          <div className="latest-panel">
            <div className="latest-head">
              <span className="panel-eyebrow">Latest retrieved chunks</span>
              <span>{latestContexts.length}</span>
            </div>

            {latestContexts.length === 0 ? (
              <div className="empty-card">
                <p>No retrieval snippets yet.</p>
                <span>The latest supporting chunks will appear here after the first answer.</span>
              </div>
            ) : (
              <div className="latest-list">
                {latestContexts.map((context) => {
                  const preview = buildContextPreview(context.content);

                  return (
                    <article className="latest-card" key={`${context.document_id}-${context.chunk_id}-latest`}>
                      <div className="latest-path">{context.file_path}</div>
                      <div className="latest-meta">Chunk {context.chunk_id + 1}</div>
                      <p>{preview.text}</p>
                      {preview.truncated ? (
                        <span className="context-truncated">Preview only</span>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          {error ? (
            <div className="error-card">
              <strong>Action blocked</strong>
              <p>{error}</p>
            </div>
          ) : null}
        </aside>
      </section>
    </main>
  );
}
