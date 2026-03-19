type FetchJsonOptions = Omit<RequestInit, 'headers'> & {
  headers?: Record<string, string>
}

export function apiBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000'
}

export async function apiFetchJson<T>(path: string, options: FetchJsonOptions = {}): Promise<T> {
  const base = apiBaseUrl().replace(/\/$/, '')
  const url = `${base}${path.startsWith('/') ? path : `/${path}`}`

  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options.body && { 'Content-Type': 'application/json' }),
      ...(options.headers ?? {}),
    },
  })

  const text = await res.text()
  let data: unknown = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = text
    }
  }

  if (!res.ok) {
    const msg =
      typeof data === 'object' && data && 'message' in data
        ? String((data as { message?: unknown }).message)
        : `Request failed (HTTP ${res.status})`
    throw new Error(msg)
  }

  return data as T
}

