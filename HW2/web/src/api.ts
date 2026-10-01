import type { AgentResponse, ApiChatMessage } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch {
      // 非 JSON 错误仍使用上面的通用提示。
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export function sendQuestion(
  sessionId: string,
  query: string,
  history: ApiChatMessage[],
): Promise<AgentResponse> {
  return requestJson<AgentResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, query, history }),
  })
}

export function artifactUrl(name: string): string {
  return `${API_BASE}/api/artifacts/${encodeURIComponent(name)}`
}
