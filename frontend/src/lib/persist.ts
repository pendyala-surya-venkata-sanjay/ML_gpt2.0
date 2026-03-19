const KEY = 'ml-assistant-ui:v1'

export type ChatMessage = { id: string; role: 'user' | 'assistant' | 'system'; content: string; ts: number }

export type ChatThread = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  datasetPath: string | null
  datasetName: string | null
  mlContext?: unknown
  messages: ChatMessage[]
}

type PersistedState = {
  activeThreadId: string | null
  threads: ChatThread[]
  activeModelId: string | null
}

export function loadState(): PersistedState | null {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return null
    return JSON.parse(raw) as PersistedState
  } catch {
    return null
  }
}

export function saveState(state: PersistedState) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    // ignore
  }
}

export function migrateLegacyState(legacy: any): PersistedState | null {
  // Legacy schema stored { datasetPath, datasetName, messages, activeModelId }
  try {
    if (!legacy || typeof legacy !== 'object') return null
    if (!Array.isArray(legacy.messages)) return null

    const now = Date.now()
    const thread: ChatThread = {
      id: 'chat_' + now.toString(16),
      title: 'New chat',
      createdAt: now,
      updatedAt: now,
      datasetPath: legacy.datasetPath ?? null,
      datasetName: legacy.datasetName ?? null,
      messages: legacy.messages as ChatMessage[],
    }

    return {
      activeThreadId: thread.id,
      threads: [thread],
      activeModelId: legacy.activeModelId ?? null,
    }
  } catch {
    return null
  }
}

