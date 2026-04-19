export type UploadResponse = {
  session_id: string;
  file_name: string;
  chunk_count: number;
  message: string;
};

export type ContextChunk = {
  file_name: string;
  chunk_id: number;
  content: string;
  score?: number | null;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
  contexts: ContextChunk[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type HealthResponse = {
  status: "ok" | "misconfigured" | string;
  provider?: string;
  search_index?: string;
  chat_deployment?: string;
  embedding_deployment?: string;
  detail?: string;
};

export type SessionResponse = {
  session_id: string;
  file_name: string;
  chunk_count: number;
};

const AZURE_API_BASE_URL = "https://ca-rag-app-36972.azurewebsites.net";
const REQUEST_TIMEOUT_MS = 120_000;

export const API_BASE_URL = normalizeApiBaseUrl(
  process.env.NEXT_PUBLIC_API_BASE_URL ?? AZURE_API_BASE_URL,
);

function normalizeApiBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

function timeoutSignal(timeoutMs = REQUEST_TIMEOUT_MS): AbortSignal {
  const controller = new AbortController();
  window.setTimeout(() => controller.abort(), timeoutMs);
  return controller.signal;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(body?.detail ?? `Request failed with ${response.status}.`);
  }
  return (await response.json()) as T;
}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs?: number,
): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      cache: "no-store",
      signal: timeoutSignal(timeoutMs),
    });
    return parseResponse<T>(response);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Azure API request timed out. Try again after the app wakes up.");
    }
    throw error;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", undefined, 30_000);
}

export async function getSession(sessionId: string): Promise<SessionResponse> {
  return request<SessionResponse>(`/api/sessions/${sessionId}`);
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<UploadResponse>("/api/sessions", {
    method: "POST",
    body: formData,
  });
}

export async function sendChatMessage(
  sessionId: string,
  question: string,
  history: ChatTurn[],
): Promise<ChatResponse> {
  return request<ChatResponse>(`/api/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      history,
    }),
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request<{ status: string }>(`/api/sessions/${sessionId}`, {
    method: "DELETE",
  });
}
