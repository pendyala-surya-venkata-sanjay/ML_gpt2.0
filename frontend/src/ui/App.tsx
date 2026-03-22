import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Toaster, toast } from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Bot,
  BrainCircuit,
  Database,
  Download,
  FlaskConical,
  Loader2,
  Mic,
  Pause,
  Settings,
  Paperclip,
  Play,
  Plus,
  Send,
  Sparkles,
  Trash2,
  LogIn,
  Search,
  BarChart3,
} from 'lucide-react'

import clsx from 'clsx'
import { apiBaseUrl, apiFetchJson } from '../lib/api'
import { loadState, migrateLegacyState, saveState, type ChatThread } from '../lib/persist'

type Role = 'user' | 'assistant' | 'system'

type ChatMessage = {
  id: string
  role: Role
  content: string
  ts: number
  streaming?: boolean
  revealed?: number
}

type TrainingResult = {
  status: 'success' | 'error'
  model_id?: string
  logs?: string[]
  analysis?: unknown
  visualizations?: unknown
  training_results?: unknown
  evaluation?: unknown
  pipeline_code?: string
  export_path?: string
  message?: string
}

type PredictResult =
  | { status: 'success'; prediction: unknown; confidence?: number }
  | { status: 'error'; message: string }

type AnalyzeResult =
  | { status: 'success'; analysis: unknown }
  | { status: 'error'; message: string }

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16)
}

function prettyJson(value: unknown) {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function parseFlexibleJsonDict(input: string): Record<string, unknown> | null {
  const trimmed = input.trim()
  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null

  // 1) Try strict JSON first.
  try {
    const parsed = JSON.parse(trimmed) as unknown
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>
  } catch {
    // continue
  }

  // 2) Best-effort "object literal" support:
  // - Quote unquoted keys: { age: 32 } -> { "age": 32 }
  // - Quote bareword string values: { city: Delhi } -> { "city": "Delhi" }
  try {
    let s = trimmed
    // Quote keys after `{` or `,`
    s = s.replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:/g, (_m, p1, key) => `${p1}"${String(key)}":`)

    // Quote bareword values after `:`, before `,` or `}`
    s = s.replace(/:\s*([A-Za-z_][A-Za-z0-9_]*)(\s*[},])/g, (_m, v, suffix) => {
      const vv = String(v)
      if (vv === 'true' || vv === 'false' || vv === 'null') return `: ${vv}${suffix}`
      return `: "${vv}"${suffix}`
    })

    const parsed = JSON.parse(s) as unknown
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>
  } catch {
    // ignore
  }

  return null
}

function vizPathToUrl(pathOrNull: unknown) {
  if (!pathOrNull || typeof pathOrNull !== 'string') return null
  // backend returns paths like "visualizations/foo.png"
  const filename = pathOrNull.split(/[/\\]/).pop()
  if (!filename) return null
  return `${apiBaseUrl().replace(/\/$/, '')}/static/visualizations/${encodeURIComponent(filename)}`
}

function buildVizMarkdown(visualizations: unknown) {
  if (!visualizations || typeof visualizations !== 'object') return null
  const v = visualizations as {
    correlation_heatmap?: unknown
    missing_values_chart?: unknown
    target_distribution?: unknown
    feature_distributions?: unknown
  }

  const parts: string[] = []
  const heat = vizPathToUrl(v.correlation_heatmap)
  if (heat) parts.push(`#### Correlation heatmap\n\n![](${heat})`)

  const missing = vizPathToUrl(v.missing_values_chart)
  if (missing) parts.push(`#### Missing values\n\n![](${missing})`)

  const target = vizPathToUrl(v.target_distribution)
  if (target) parts.push(`#### Target distribution\n\n![](${target})`)

  if (Array.isArray(v.feature_distributions)) {
    const urls = v.feature_distributions.map(vizPathToUrl).filter(Boolean) as string[]
    if (urls.length) {
      parts.push(`#### Feature distributions`)
      for (const u of urls) parts.push(`![](${u})`)
    }
  }

  if (!parts.length) return null
  return ['### Visualizations', ...parts].join('\n\n')
}

type VizKey = 'correlation_heatmap' | 'missing_values_chart' | 'target_distribution' | 'feature_distributions'

function buildVizMarkdownForKeys(visualizations: unknown, keys: VizKey[]) {
  if (!visualizations || typeof visualizations !== 'object') return null
  const v = visualizations as {
    correlation_heatmap?: unknown
    missing_values_chart?: unknown
    target_distribution?: unknown
    feature_distributions?: unknown
  }

  const parts: string[] = []

  if (keys.includes('correlation_heatmap')) {
    const heat = vizPathToUrl(v.correlation_heatmap)
    if (heat) parts.push(`#### Correlation heatmap\n\n![](${heat})`)
  }

  if (keys.includes('missing_values_chart')) {
    const missing = vizPathToUrl(v.missing_values_chart)
    if (missing) parts.push(`#### Missing values\n\n![](${missing})`)
  }

  if (keys.includes('target_distribution')) {
    const target = vizPathToUrl(v.target_distribution)
    if (target) parts.push(`#### Target distribution\n\n![](${target})`)
  }

  if (keys.includes('feature_distributions') && Array.isArray(v.feature_distributions)) {
    const urls = v.feature_distributions.map(vizPathToUrl).filter(Boolean) as string[]
    if (urls.length) {
      parts.push(`#### Feature distributions`)
      for (const u of urls) parts.push(`![](${u})`)
    }
  }

  if (!parts.length) return null
  return ['### Visualizations', ...parts].join('\n\n')
}

function parseVizKeys(lowerText: string): VizKey[] | null {
  const lower = lowerText.toLowerCase()
  const allKeys: VizKey[] = ['correlation_heatmap', 'missing_values_chart', 'target_distribution', 'feature_distributions']

  // Explicit commands
  if (lower === '/visualizations' || lower === '/viz' || lower === 'show visualization' || lower === 'show visualizations') {
    return allKeys
  }

  // Correlation heatmap / matrix (including common typo: "corelation")
  // Match plain-language requests like "give corelation map", "correlation heatmap", etc.
  const mentionsCorrelation =
    lower.includes('correlation') || lower.includes('corelation') || lower.includes('corr heat') || lower.includes('corr') || lower.includes('rel')
  const mentionsHeatmapMatrix =
    lower.includes('heatmap') ||
    lower.includes('heat map') ||
    lower.includes('matrix') ||
    lower.includes('map') ||
    lower.includes('correlation plot') ||
    lower.includes('corr plot')
  if (mentionsCorrelation && mentionsHeatmapMatrix) return ['correlation_heatmap']

  // Missing values
  const wantsMissing =
    lower.includes('missing') &&
    (lower.includes('value') || lower.includes('values') || lower.includes('null') || lower.includes('n/a') || lower.includes('na'))
  if (wantsMissing) return ['missing_values_chart']

  // Target distribution
  const wantsTargetDistribution =
    (lower.includes('target') || lower.includes('class') || lower.includes('label') || lower.includes('category')) &&
    (lower.includes('distribution') || lower.includes('count') || lower.includes('frequency'))
  if (wantsTargetDistribution) return ['target_distribution']

  // Feature distributions
  const wantsFeatureDistributions =
    (lower.includes('feature') || lower.includes('features')) && (lower.includes('distribution') || lower.includes('distributions') || lower.includes('histogram') || lower.includes('plot'))
  if (wantsFeatureDistributions) return ['feature_distributions']

  // Generic "distributions/histograms" request that likely refers to features.
  if (
    (lower.includes('distributions') || lower.includes('distribution') || lower.includes('histogram')) &&
    (lower.includes('feature') || lower.includes('features'))
  ) {
    return ['feature_distributions']
  }

  // Generic "visualization" request -> show everything we have
  if (lower.includes('visualization') || lower.includes('viz')) return allKeys

  return null
}

type TabKey = 'chat' | 'dataset' | 'predict'

export default function App() {
  const [tab, setTab] = useState<TabKey>('chat')

  const [threads, setThreads] = useState<ChatThread[]>(() => {
    const saved = loadState() as any
    if (saved?.threads?.length) return saved.threads as ChatThread[]

    const migrated = migrateLegacyState(saved)
    if (migrated?.threads?.length) return migrated.threads

    const now = Date.now()
    return [
      {
        id: 'chat_' + now.toString(16),
        title: 'New chat',
        createdAt: now,
        updatedAt: now,
        datasetPath: null,
        datasetName: null,
        messages: [
          {
            id: uid(),
            role: 'assistant',
            content:
              "Hi! Upload a CSV dataset, then ask me to train a model or answer questions about your data. You can also use the **Train** and **Predict** tabs for structured workflows.",
            ts: now,
          },
        ],
      },
    ]
  })

  const [activeThreadId, setActiveThreadId] = useState<string | null>(() => {
    const saved = loadState() as any
    if (saved?.activeThreadId) return saved.activeThreadId as string
    const migrated = migrateLegacyState(saved)
    return migrated?.activeThreadId ?? null
  })

  const activeThread = useMemo(() => {
    const id = activeThreadId ?? threads[0]?.id ?? null
    return threads.find((t) => t.id === id) ?? threads[0]
  }, [activeThreadId, threads])

  const datasetPath = activeThread?.datasetPath ?? null
  const datasetName = activeThread?.datasetName ?? null
  const mlContext = activeThread?.mlContext ?? null

  const setMlContext = (ctx: unknown) => {
    if (!activeThread) return
    setThreads((prev) => prev.map((t) => (t.id === activeThread.id ? { ...t, mlContext: ctx } : t)))
  }
  const [activeModelId, setActiveModelId] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [models, setModels] = useState<any[]>([])
  const [exportsList, setExportsList] = useState<any[]>([])

  const [messages, setMessages] = useState<ChatMessage[]>(() => activeThread?.messages ?? [])
  const streamingTimerRef = useRef<number | null>(null)
  const [composer, setComposer] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const chatAbortRef = useRef<AbortController | null>(null)
  const isPausedRef = useRef(false)

  const [isUploading, setIsUploading] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [training, setTraining] = useState<TrainingResult | null>(null)
  const [lastVisualizations, setLastVisualizations] = useState<unknown>(null)

  const [predictJson, setPredictJson] = useState<string>(() =>
    prettyJson({ age: 32, salary: 70000, experience: 6, city: 'Delhi' }),
  )
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null)

  // New states for voice input and authentication
  const [isRecording, setIsRecording] = useState(false)
  const speechRecRef = useRef<any>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [showLogin, setShowLogin] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showModelSearch, setShowModelSearch] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    isPausedRef.current = isPaused
  }, [isPaused])

  useEffect(() => {
    setMessages(activeThread?.messages ?? [])
  }, [activeThread?.id])

  useEffect(() => {
    const saved = loadState() as any
    const migrated = migrateLegacyState(saved)
    if (saved?.activeModelId) setActiveModelId(saved.activeModelId)
    else if (migrated?.activeModelId) setActiveModelId(migrated.activeModelId)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }, [theme])

  // Cleanup any in-flight work when the component unmounts.
  useEffect(() => {
    return () => {
      try {
        chatAbortRef.current?.abort()
      } catch {
        // ignore
      }
      chatAbortRef.current = null
      try {
        speechRecRef.current?.stop?.()
      } catch {
        // ignore
      }
      speechRecRef.current = null
    }
  }, [])

  useEffect(() => {
    saveState({
      activeThreadId: activeThread?.id ?? null,
      threads,
      activeModelId,
    })
  }, [threads, activeThread?.id, activeModelId])

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages.length, isSending])

  useEffect(() => {
    const streaming = messages.find((m) => m.streaming)
    if (!streaming) return

    if (isPaused) {
      if (streamingTimerRef.current) window.clearInterval(streamingTimerRef.current)
      streamingTimerRef.current = null
      return
    }

    if (streamingTimerRef.current) window.clearInterval(streamingTimerRef.current)
    streamingTimerRef.current = window.setInterval(() => {
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== streaming.id) return m
          const next = Math.min((m.revealed ?? 0) + 6, m.content.length)
          const done = next >= m.content.length
          return { ...m, revealed: next, streaming: done ? false : true }
        }),
      )
    }, 18)

    return () => {
      if (streamingTimerRef.current) window.clearInterval(streamingTimerRef.current)
      streamingTimerRef.current = null
    }
  }, [messages, isPaused])

  useEffect(() => {
    // Write current messages back into the active thread
    if (!activeThread) return
    setThreads((prev) =>
      prev.map((t) =>
        t.id === activeThread.id
          ? {
              ...t,
              messages,
              updatedAt: Date.now(),
              title:
                t.title === 'New chat'
                  ? (messages.find((m) => m.role === 'user')?.content?.slice(0, 40) || 'New chat')
                  : t.title,
            }
          : t,
      ),
    )
  }, [messages])

  const datasetBadge = useMemo(() => {
    if (!datasetPath) return { label: 'No dataset', tone: 'muted' as const }
    return { label: datasetName ?? 'Dataset uploaded', tone: 'ok' as const }
  }, [datasetPath, datasetName])

  const activeModelLabel = useMemo(() => {
    if (!activeModelId) return null
    const m = models.find((x) => x?.id === activeModelId)
    return (m?.display_name as string | undefined) ?? (m?.dataset_name as string | undefined) ?? activeModelId
  }, [activeModelId, models])

  async function refreshModelsAndExports() {
    try {
      const m = await apiFetchJson<{ status: string; active_model_id: string | null; models: any[] }>('/models', {
        method: 'GET',
      })
      if (m.status === 'success') {
        setModels(m.models ?? [])
        setActiveModelId(m.active_model_id ?? null)
      }
    } catch {
      // ignore
    }

    try {
      const ex = await apiFetchJson<{ status: string; exports: any[] }>('/exports', { method: 'GET' })
      if (ex.status === 'success') setExportsList(ex.exports ?? [])
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    void refreshModelsAndExports()
  }, [])

  async function setBackendActiveModel(modelId: string) {
    const res = await apiFetchJson<{ status: 'success' | 'error'; active_model_id?: string; message?: string }>(
      `/models/active?model_id=${encodeURIComponent(modelId)}`,
      { method: 'POST' },
    )
    if (res.status !== 'success') throw new Error(res.message || 'Failed to set active model')
    setActiveModelId(modelId)
    toast.success('Active model updated')
  }

  function togglePauseAgent() {
    const streaming = messages.find((m) => m.streaming)
    const inFlightChat = chatAbortRef.current != null

    // Nothing to pause.
    if (!streaming && !inFlightChat) return

    if (!isPausedRef.current) {
      // Pause: stop simulated streaming + abort any in-flight /chat request.
      isPausedRef.current = true
      setIsPaused(true)

      try {
        chatAbortRef.current?.abort()
      } catch {
        // ignore
      }
      chatAbortRef.current = null

      if (streamingTimerRef.current) window.clearInterval(streamingTimerRef.current)
      streamingTimerRef.current = null

      return
    }

    // Resume
    isPausedRef.current = false
    setIsPaused(false)
  }

  function startVoiceInput() {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      toast.error('Voice input is not supported in this browser.')
      return
    }

    try {
      // Ensure any previous instance is stopped.
      speechRecRef.current?.stop?.()
    } catch {
      // ignore
    }

    const rec = new SpeechRecognition()
    speechRecRef.current = rec
    rec.lang = 'en-US'
    rec.interimResults = true
    rec.continuous = true

    rec.onresult = (event: any) => {
      // Aggregate transcript from the latest result slice.
      let transcript = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript
      }
      if (transcript.trim().length) setComposer(transcript.trim())
    }

    rec.onerror = (err: any) => {
      setIsRecording(false)
      speechRecRef.current = null
      const msg = err?.error ? String(err.error) : 'Voice recognition error'
      toast.error(msg)
    }

    rec.onend = () => {
      // `onend` also fires after manual stop.
      setIsRecording(false)
      speechRecRef.current = null
    }

    rec.start()
    setIsRecording(true)
  }

  function stopVoiceInput() {
    try {
      speechRecRef.current?.stop?.()
    } catch {
      // ignore
    }
    speechRecRef.current = null
    setIsRecording(false)
  }

  async function sendChat(text: string) {
    const trimmed = text.trim()
    if (!trimmed) return

    // ChatGPT-like controlled workflow via chat commands:
    // - /upload (use the paperclip in chat)
    // - /analyze
    // - /train
    // - /predict {json}
    const lower = trimmed.toLowerCase()
    const isTrain = lower === 'train model' || lower === 'train' || lower === '/train'
    const isAnalyze = lower === 'analyze' || lower === '/analyze' || lower === 'analyze dataset'
    const isPredict = lower.startsWith('/predict')
    const vizKeys = parseVizKeys(lower)
    const isViz = vizKeys !== null
    const isExplainTraining =
      lower === '/explain_training' ||
      lower === '/explain' ||
      lower === 'explain training' ||
      lower === 'explain the training' ||
      lower === 'explain the above training'

    const userMsg: ChatMessage = { id: uid(), role: 'user', content: trimmed, ts: Date.now() }
    setMessages((m) => [...m, userMsg])
    setComposer('')

    if (isAnalyze) {
      if (!datasetPath) {
        const msg = 'Please upload a dataset first (use the paperclip button).'
        toast.error(msg)
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: msg, ts: Date.now() }])
        return
      }
      setIsSending(true)
      try {
        const data = await apiFetchJson<AnalyzeResult>(
          `/analyze_dataset?dataset_path=${encodeURIComponent(datasetPath)}`,
          { method: 'GET' },
        )
        if (data.status === 'success') {
          setMessages((m) => [
            ...m,
            {
              id: uid(),
              role: 'assistant',
              content:
                `### Dataset analysis\n\n\`\`\`json\n${prettyJson(data.analysis)}\n\`\`\`\n\nNext: type **/train** to train a model.`,
              ts: Date.now(),
            },
          ])
        } else {
          throw new Error(data.message)
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        toast.error(msg)
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: `Analysis error: ${msg}`, ts: Date.now() }])
      } finally {
        setIsSending(false)
      }
      return
    }

    if (isTrain) {
      if (!datasetPath) {
        const msg = 'Please upload a dataset first (use the paperclip button).'
        toast.error(msg)
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: msg, ts: Date.now() }])
        return
      }

      setIsTraining(true)
      setIsSending(true)
      setTraining(null)
      try {
        const url = `/train_model?dataset_path=${encodeURIComponent(datasetPath)}`
        const data = await apiFetchJson<TrainingResult>(url, { method: 'POST' })
        setTraining(data)
        if (data.status === 'success') setLastVisualizations(data.visualizations ?? null)
        if (data.status === 'success') {
          setMlContext({
            dataset: { name: datasetName, path: datasetPath },
            analysis: data.analysis ?? null,
            training: data.training_results ?? null,
            evaluation: data.evaluation ?? null,
            pipeline: {
              feature_engineering: [
                'datetime features (columns containing \"date\")',
                'pairwise numeric interaction features',
              ],
              preprocessing: {
                numeric: ['mean imputation', 'standard scaling'],
                categorical: ['most_frequent imputation', 'one-hot encoding (handle_unknown=ignore)'],
              },
              models_tried:
                (data.training_results as any)?.results != null ? Object.keys((data.training_results as any).results) : null,
              best_model: (data.training_results as any)?.best_model ?? null,
            },
          })
        }

        const mdParts: string[] = []
        mdParts.push('### Training result')
        mdParts.push(`**Status:** ${data.status}`)
        if (data.status === 'error') mdParts.push(`\n**Error:** ${data.message ?? 'Unknown error'}`)
        if (data.logs?.length) mdParts.push(`\n**Logs:**\n${data.logs.map((l) => `- ${l}`).join('\n')}`)
        if (data.training_results) mdParts.push(`\n**Training results (JSON):**\n\`\`\`json\n${prettyJson(data.training_results)}\n\`\`\``)
        if (data.evaluation) mdParts.push(`\n**Evaluation (JSON):**\n\`\`\`json\n${prettyJson(data.evaluation)}\n\`\`\``)
        // Do NOT show visualizations by default. User can request them with /visualizations.
        if (data.export_path) mdParts.push(`\n**Export zip (backend path):** \`${data.export_path}\``)
        mdParts.push('\nIf you want plots (relationships / heatmaps), type **/visualizations**.')
        mdParts.push('\nNext: type **/predict {json}** (or open the Predict tab) to run inference.')

        setMessages((m) => [
          ...m,
          { id: uid(), role: 'assistant', content: mdParts.join('\n\n'), ts: Date.now() },
        ])

        if (data.status === 'success') toast.success('Training completed')
        else toast.error(data.message || 'Training failed')
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        toast.error(msg)
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: `Training error: ${msg}`, ts: Date.now() }])
      } finally {
        setIsTraining(false)
        setIsSending(false)
      }
      return
    }

    if (isViz) {
      const vizMd =
        vizKeys!.length === 4 ? buildVizMarkdown(lastVisualizations) : buildVizMarkdownForKeys(lastVisualizations, vizKeys!)
      if (!vizMd) {
        const msg = 'No visualizations available yet. Train a model first, then ask for a visualization (e.g., "show correlation map").'
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: msg, ts: Date.now() }])
        return
      }
      setMessages((m) => [
        ...m,
        { id: uid(), role: 'assistant', content: vizMd, ts: Date.now() },
      ])
      return
    }

    if (isExplainTraining) {
      if (!mlContext) {
        const msg = 'No training context found yet. Train a model first using **/train**.'
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: msg, ts: Date.now() }])
        return
      }

      const report = [
        '### Training report',
        '',
        'This summary is based on your most recent training run in this chat.',
        '',
        '```json',
        prettyJson(mlContext),
        '```',
        '',
        'If you want, tell me what part to explain deeper (feature engineering, preprocessing, model selection, or metrics).',
      ].join('\n')

      setMessages((m) => [...m, { id: uid(), role: 'assistant', content: report, ts: Date.now() }])
      return
    }

    // New request: allow pure JSON-dictionary prediction like `{ "feature": value }` directly in chat.
    const looksLikeJsonDict = trimmed.startsWith('{') && trimmed.endsWith('}')
    if (!isPredict && looksLikeJsonDict) {
      try {
        let parsed: unknown = null
        try {
          parsed = JSON.parse(trimmed) as unknown
        } catch {
          parsed = parseFlexibleJsonDict(trimmed)
        }

        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          setIsPredicting(true)
          setIsSending(true)
          setPredictResult(null)
          const payload = parsed as Record<string, unknown>
          try {
            const modelQuery = activeModelId ? `?model_id=${encodeURIComponent(activeModelId)}` : ''
            const data = await apiFetchJson<PredictResult>(`/predict${modelQuery}`, {
              method: 'POST',
              body: JSON.stringify(payload),
            })
            setPredictResult(data)
            setMessages((m) => [
              ...m,
              {
                id: uid(),
                role: 'assistant',
                content: `### Prediction\n\n\`\`\`json\n${prettyJson(data)}\n\`\`\``,
                ts: Date.now(),
              },
            ])
            if (data.status === 'success') toast.success('Prediction complete')
            else toast.error(data.message || 'Prediction failed')
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e)
            toast.error(msg)
            setMessages((m) => [...m, { id: uid(), role: 'assistant', content: `Prediction error: ${msg}`, ts: Date.now() }])
          } finally {
            setIsPredicting(false)
            setIsSending(false)
          }
          return
        }
        // If it's a dict-shaped input but we couldn't parse it, give a helpful hint.
        setMessages((m) => [
          ...m,
          {
            id: uid(),
            role: 'assistant',
            content:
              'I can predict from a dictionary input. Example:\n\n`{ "age": 32, "salary": 70000, "city": "Delhi" }`\n\nIf you used unquoted keys/values, try quoting strings (like `"Delhi"`).',
            ts: Date.now(),
          },
        ])
        return
      } catch {
        // Not a valid JSON object; fall through to other commands.
      }
    }

    if (isPredict) {
      const jsonPart = trimmed.replace(/^\/predict\s*/i, '').trim()
      if (!jsonPart) {
        const help =
          'Usage: `/predict {"feature1": 1, "feature2": "x"}`\n\nOr use the Predict tab for a JSON editor.'
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: help, ts: Date.now() }])
        return
      }

      setIsPredicting(true)
      setIsSending(true)
      setPredictResult(null)
      try {
        const payload = JSON.parse(jsonPart) as Record<string, unknown>
        const modelQuery = activeModelId ? `?model_id=${encodeURIComponent(activeModelId)}` : ''
        const data = await apiFetchJson<PredictResult>(`/predict${modelQuery}`, {
          method: 'POST',
          body: JSON.stringify(payload),
        })
        setPredictResult(data)
        setMessages((m) => [
          ...m,
          {
            id: uid(),
            role: 'assistant',
            content: `### Prediction\n\n\`\`\`json\n${prettyJson(data)}\n\`\`\``,
            ts: Date.now(),
          },
        ])
        if (data.status === 'success') toast.success('Prediction complete')
        else toast.error(data.message || 'Prediction failed')
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        toast.error(msg)
        setMessages((m) => [...m, { id: uid(), role: 'assistant', content: `Prediction error: ${msg}`, ts: Date.now() }])
      } finally {
        setIsPredicting(false)
        setIsSending(false)
      }
      return
    }

    // New request: ensure pause is cleared and attach an abort controller.
    const controller = new AbortController()
    chatAbortRef.current = controller
    setIsPaused(false)
    isPausedRef.current = false

    setIsSending(true)
    try {
      const data = await apiFetchJson<{ response: string }>('/chat', {
        method: 'POST',
        signal: controller.signal,
        body: JSON.stringify({
          message: trimmed,
          thread_id: activeThread?.id ?? null,
          history: [
            ...(mlContext != null
              ? [
                  {
                    role: 'system',
                    content:
                      'CONTEXT (use this to answer questions about the user dataset/model/training pipeline).\\n' +
                      JSON.stringify(mlContext),
                  },
                ]
              : []),
            ...messages.slice(-20).map((m) => ({ role: m.role, content: m.content })),
          ],
        }),
      })

      const assistantMsg: ChatMessage = {
        id: uid(),
        role: 'assistant',
        content: data.response ?? '(no response)',
        ts: Date.now(),
        streaming: true,
        revealed: 0,
      }
      setMessages((m) => [...m, assistantMsg])
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      const msg = e instanceof Error ? e.message : String(e)
      toast.error(msg)
      setMessages((m) => [
        ...m,
        { id: uid(), role: 'assistant', content: `Server error: ${msg}`, ts: Date.now() },
      ])
    } finally {
      setIsSending(false)
      if (chatAbortRef.current === controller) chatAbortRef.current = null
    }
  }

  async function onUpload(file: File) {
    setIsUploading(true)
    setTraining(null)
    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch(`${apiBaseUrl()}/upload_dataset`, {
        method: 'POST',
        body: form,
      })
      const data = (await res.json()) as { status: string; dataset_path?: string; message?: string }
      if (!res.ok || data.status !== 'success' || !data.dataset_path) {
        throw new Error(data.message || `Upload failed (HTTP ${res.status})`)
      }

      const threadId = activeThreadId ?? activeThread?.id ?? null
      if (!threadId) throw new Error('No active chat thread found for dataset upload')
      setThreads((prev) =>
        prev.map((t) =>
          t.id === threadId ? { ...t, datasetPath: data.dataset_path ?? null, datasetName: file.name ?? null } : t,
        ),
      )
      toast.success('Dataset uploaded')

      setMessages((m) => [
        ...m,
        {
          id: uid(),
          role: 'assistant',
          content:
            `✅ Dataset uploaded: **${file.name}**\n\nNow you can type **/analyze** for analysis or **/train** to train the model.`,
          ts: Date.now(),
        },
      ])
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      toast.error(msg)
      // Make sure we don't leave the UI in an "uploading" state without feedback.
      setMessages((m) => [
        ...m,
        { id: uid(), role: 'assistant', content: `Upload error: ${msg}`, ts: Date.now() },
      ])
    } finally {
      setIsUploading(false)
    }
  }

  async function runTraining() {
    if (!datasetPath) {
      toast.error('Upload a dataset first')
      setTab('dataset')
      return
    }

    setIsTraining(true)
    setTraining(null)
    try {
      // FastAPI will treat this as a query param by default
      const url = `/train_model?dataset_path=${encodeURIComponent(datasetPath)}`
      const data = await apiFetchJson<TrainingResult>(url, { method: 'POST' })
      setTraining(data)
      if (data.status === 'success') setLastVisualizations(data.visualizations ?? null)
      if (data.status === 'success') {
        setMlContext({
          dataset: { name: datasetName, path: datasetPath },
          analysis: data.analysis ?? null,
          training: data.training_results ?? null,
          evaluation: data.evaluation ?? null,
          pipeline: {
            feature_engineering: [
              'datetime features (columns containing \"date\")',
              'pairwise numeric interaction features',
            ],
            preprocessing: {
              numeric: ['mean imputation', 'standard scaling'],
              categorical: ['most_frequent imputation', 'one-hot encoding (handle_unknown=ignore)'],
            },
            models_tried:
              (data.training_results as any)?.results != null ? Object.keys((data.training_results as any).results) : null,
            best_model: (data.training_results as any)?.best_model ?? null,
          },
        })
      }
      await refreshModelsAndExports()
      if (data.status === 'success') toast.success('Training completed')
      else toast.error(data.message || 'Training failed')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      const hint = `Failed to fetch. Check backend is running at ${apiBaseUrl()} and CORS is allowed.`
      toast.error(msg === 'Failed to fetch' ? hint : msg)
      setTraining({ status: 'error', message: msg })
    } finally {
      setIsTraining(false)
    }
  }

  async function runPredict() {
    setIsPredicting(true)
    setPredictResult(null)
    try {
      const payload = JSON.parse(predictJson) as Record<string, unknown>
      const modelQuery = activeModelId ? `?model_id=${encodeURIComponent(activeModelId)}` : ''
      const data = await apiFetchJson<PredictResult>(`/predict${modelQuery}`, {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      setPredictResult(data)
      if (data.status === 'success') toast.success('Prediction complete')
      else toast.error(data.message || 'Prediction failed')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      toast.error(msg)
      setPredictResult({ status: 'error', message: msg })
    } finally {
      setIsPredicting(false)
    }
  }

  function newChat() {
    const now = Date.now()
    const thread: ChatThread = {
      id: 'chat_' + now.toString(16) + '_' + Math.random().toString(16).slice(2),
      title: 'New chat',
      createdAt: now,
      updatedAt: now,
      datasetPath: null,
      datasetName: null,
      mlContext: null,
      messages: [
        {
          id: uid(),
          role: 'assistant',
          content: 'How can I help you today?',
          ts: now,
        },
      ],
    }

    setThreads((prev) => [thread, ...prev])
    setActiveThreadId(thread.id)
    setTab('chat')
  }

  async function deleteChatThread(threadId: string) {
    const thread = threads.find((t) => t.id === threadId)
    if (!thread) return

    const ok = window.confirm('Delete this chat and remove its trained models?')
    if (!ok) return

    // If there is an uploaded dataset tied to this chat, delete models trained from it.
    let deletionError: string | null = null
    let removedModels = 0
    let deletedDataset = false

    try {
      if (thread.datasetPath) {
        const normalizedDatasetPath = String(thread.datasetPath).replace(/\\/g, '/').trim()
        const data = await apiFetchJson<{
          status: string
          removed_models?: number
          deleted_dataset?: boolean
          message?: string
        }>('/models/delete_by_dataset', {
          method: 'POST',
          body: JSON.stringify({ dataset_path: normalizedDatasetPath }),
        })

        if (data.status !== 'success') {
          deletionError = data.message || 'Failed to delete models'
        } else {
          removedModels = data.removed_models ?? 0
          deletedDataset = data.deleted_dataset ?? false
        }
      }
    } catch (e) {
      deletionError = e instanceof Error ? e.message : String(e)
    }

    // Remove the chat from UI + persisted localStorage.
    setThreads((prev) => {
      const remaining = prev.filter((t) => t.id !== threadId)
      if (activeThreadId === threadId) {
        setActiveThreadId(remaining[0]?.id ?? null)
        setTab('chat')
      }
      return remaining
    })

    // Refresh active models list after deletion.
    await refreshModelsAndExports()
    if (deletionError) {
      toast.error(`Chat deleted, but models were not fully removed: ${deletionError}`)
    } else {
      const suffix =
        removedModels > 0 ? ` (removed ${removedModels} model${removedModels === 1 ? '' : 's'}${deletedDataset ? ', deleted dataset' : ''})` : ''
      toast.success(`Chat deleted${suffix}`)
    }
  }

  async function deleteModelAndArtifacts(modelId: string) {
    const model = models.find((m) => m?.id === modelId)
    const name = model?.display_name ?? model?.dataset_name ?? model?.id ?? modelId

    const ok = window.confirm(`Delete model "${String(name)}" and all its related artifacts?`)
    if (!ok) return

    try {
      const res = await apiFetchJson<{ status: string; message?: string }>(
        '/models/delete',
        {
          method: 'POST',
          body: JSON.stringify({ model_id: modelId }),
        },
      )

      if (res.status !== 'success') throw new Error(res.message || 'Failed to delete model')

      if (activeModelId === modelId) setActiveModelId(null)
      toast.success('Model deleted')
      await refreshModelsAndExports()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      toast.error(msg)
    }
  }

  return (
    <>
      {/* Login/Signup Modal */}
      {showLogin && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/60" onClick={() => setShowLogin(false)} />
          <div className="absolute inset-0 p-4 md:p-8 grid place-items-center">
            <div className="glass w-full max-w-md rounded-2xl overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between gap-3">
                <div className="text-sm font-semibold">Login / Signup</div>
                <button className="btn p-2" onClick={() => setShowLogin(false)}>
                  ×
                </button>
              </div>
              <div className="p-4 grid gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <input
                    type="email"
                    className="input"
                    placeholder="Enter your email"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <input
                    type="password"
                    className="input"
                    placeholder="Enter your password"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button 
                    className="btn btn-primary"
                    onClick={() => {
                      setIsAuthenticated(true)
                      setShowLogin(false)
                      toast.success('Logged in successfully!')
                    }}
                  >
                    Login
                  </button>
                  <button 
                    className="btn"
                    onClick={() => {
                      setIsAuthenticated(true)
                      setShowLogin(false)
                      toast.success('Account created successfully!')
                    }}
                  >
                    Signup
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(15, 23, 42, 0.85)',
            color: 'white',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            backdropFilter: 'blur(8px)',
          },
        }}
      />

      <div id="jfj32k" className="h-screen overflow-hidden grid grid-cols-1 md:grid-cols-[320px_1fr] bg-slate-50 dark:bg-slate-950">
        <aside className="hidden md:flex flex-col border-r border-gray-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 h-screen">
          <div className="p-3">
            <div className="glass rounded-2xl p-3 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <div className="h-9 w-9 rounded-xl bg-indigo-500/20 grid place-items-center border border-indigo-400/25">
                  <Sparkles className="h-5 w-5 text-indigo-200" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold truncate">SAVESA</div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-300 truncate">
                    {activeModelLabel ? `Active model: ${activeModelLabel}` : 'No active model'}
                  </div>
                </div>
              </div>

              <button className="btn" onClick={newChat} title="New chat">
                <Plus className="h-4 w-4" />
              </button>
              {!isAuthenticated && (
                <button className="btn" onClick={() => setShowLogin(true)} title="Login">
                  <LogIn className="h-4 w-4" />
                </button>
              )}
              {isAuthenticated && (
                <button className="btn" onClick={() => setShowModelSearch(true)} title="Search Models">
                  <Search className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          <div className="px-3 pb-2">
            <span
              className={clsx(
                'pill w-full justify-between',
                datasetBadge.tone === 'ok' && 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
              )}
              title={datasetPath ?? 'No dataset uploaded'}
            >
              <span className="inline-flex items-center gap-2 min-w-0">
                <Database className="h-3.5 w-3.5" />
                <span className="truncate">{datasetBadge.label}</span>
              </span>
            </span>
          </div>

          <div className="flex-1 overflow-auto scrollbar px-2 pb-2 min-h-0">
            <div className="px-2 py-2 text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">Chats</div>
            <div className="px-1 pb-2">
              <button
                className="w-full text-left rounded-xl px-3 py-2 border border-transparent hover:border-gray-200 hover:bg-white flex items-center justify-between text-sm text-slate-700 dark:text-slate-200"
                onClick={() => setShowModelSearch(true)}
              >
                <span className="flex items-center gap-2">
                  <Search className="h-4 w-4" />
                  Search models
                </span>
              </button>
            </div>
            <div className="grid gap-1 px-1">
              {threads
                .slice()
                .sort((a, b) => b.updatedAt - a.updatedAt)
                .map((t) => (
                  <div key={t.id} className="flex items-stretch gap-2">
                    <button
                      className={clsx(
                        'flex-1 text-left rounded-xl px-3 py-2 border transition-colors',
                        t.id === activeThread?.id
                          ? 'bg-white/10 border-white/10'
                          : 'bg-transparent border-transparent hover:bg-white/5 hover:border-white/10',
                      )}
                      onClick={() => {
                        setActiveThreadId(t.id)
                        setTab('chat')
                      }}
                      title={t.title}
                    >
                      <div className="text-sm font-medium truncate">{t.title}</div>
                      <div className="text-[11px] text-slate-600 dark:text-slate-300 truncate">
                        {t.datasetName ? t.datasetName : 'No dataset'}
                      </div>
                    </button>
                    <button
                      className={clsx(
                        'btn !h-auto !px-2',
                        t.id === activeThread?.id ? 'bg-rose-500/10 border-rose-400/20' : 'bg-transparent',
                      )}
                      title="Delete chat"
                      onClick={(e) => {
                        e.stopPropagation()
                        void deleteChatThread(t.id)
                      }}
                    >
                      <Trash2 className="h-4 w-4 text-rose-200" />
                    </button>
                  </div>
                ))}
            </div>
          </div>

            <div className="p-3 border-t border-gray-200 dark:border-slate-800">
            <div className="glass rounded-2xl p-2 flex items-center justify-between gap-2">
              <button className="btn" onClick={() => setSettingsOpen(true)} title="Settings">
                <Settings className="h-4 w-4" />
              </button>
              <div className="text-[11px] text-slate-600 dark:text-slate-300 truncate">Settings</div>
            </div>
          </div>
        </aside>

        <main className="h-screen overflow-hidden bg-slate-50 dark:bg-slate-950">
          <header className="md:hidden sticky top-0 z-10 border-b border-gray-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 backdrop-blur-md">
            <div className="px-4 py-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <div className="h-9 w-9 rounded-xl bg-indigo-500/20 grid place-items-center border border-indigo-400/25">
                  <Sparkles className="h-5 w-5 text-indigo-200" />
                </div>
                <div>
                  <div className="text-sm font-semibold">SAVESA</div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-300 truncate max-w-[55vw]">
                    {datasetPath ? `Dataset: ${datasetName ?? 'uploaded'}` : 'No dataset'}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button className="btn" onClick={() => setSettingsOpen(true)} title="Settings">
                  <Settings className="h-4 w-4" />
                </button>
                <button className={clsx('btn', tab === 'chat' && 'btn-primary')} onClick={() => setTab('chat')}>
                  <Bot className="h-4 w-4" />
                </button>
                <button
                  className={clsx('btn', tab === 'dataset' && 'btn-primary')}
                  onClick={() => setTab('dataset')}
                >
                  <FlaskConical className="h-4 w-4" />
                </button>
                <button
                  className={clsx('btn', tab === 'predict' && 'btn-primary')}
                  onClick={() => setTab('predict')}
                >
                  <BrainCircuit className="h-4 w-4" />
                </button>
              </div>
            </div>
          </header>

          {tab === 'chat' && (
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-auto scrollbar" ref={listRef}>
                <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
                  <div className="glass rounded-2xl p-3 text-sm text-slate-800 dark:text-slate-100">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex flex-col">
                        <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Dataset workflow</span>
                        <span className="text-sm text-slate-800 dark:text-slate-100">
                          Upload a CSV and train a model directly from chat.
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <label className={clsx('btn', isUploading && 'opacity-60 cursor-not-allowed')}>
                          <Paperclip className="h-4 w-4" />
                          {isUploading ? 'Uploading…' : 'Upload CSV'}
                          <input
                            type="file"
                            accept=".csv,text/csv"
                            className="hidden"
                            disabled={isUploading}
                            onChange={(e) => {
                              const f = e.target.files?.[0]
                              if (f) void onUpload(f)
                            }}
                          />
                        </label>
                        <button
                          className="btn btn-primary"
                          disabled={!datasetPath || isTraining}
                          onClick={() => void sendChat('/train')}
                          title={!datasetPath ? 'Upload a dataset first' : 'Train'}
                        >
                          {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                          Train
                        </button>
                      </div>
                    </div>
                  </div>
                  {messages.map((m) => (
                    <ChatBubble
                      key={m.id}
                      role={m.role}
                      content={m.content}
                      streaming={m.streaming}
                      revealed={m.revealed}
                    />
                  ))}

                  {(isSending || isTraining || isUploading || isPredicting) && (
                    <div className="flex items-center gap-3 text-sm text-slate-700 dark:text-slate-200">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Working…
                    </div>
                  )}
                </div>
              </div>

              <div className="border-t border-gray-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 backdrop-blur-md">
                <div className="max-w-4xl mx-auto px-4 py-4">
                  <div className="glass rounded-2xl p-3">
                    <div className="flex items-end gap-2">
                      <textarea
                        className="textarea min-h-[44px] max-h-[160px]"
                        placeholder="Message SAVESA… (Shift+Enter for newline)"
                        value={composer}
                        onChange={(e) => setComposer(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            if (!isSending) void sendChat(composer)
                          }
                        }}
                        disabled={isSending}
                      />
                      <button className="btn btn-primary h-[44px] px-4" disabled={isSending} onClick={() => sendChat(composer)}>
                        <Send className="h-4 w-4" />
                        Send
                      </button>
                      <button
                        className={clsx('btn h-[44px] px-3', isPaused && 'bg-indigo-500/15')}
                        disabled={!isPaused && !chatAbortRef.current && !messages.some((m) => m.streaming)}
                        onClick={() => togglePauseAgent()}
                        title={isPaused ? 'Resume agent response' : 'Pause agent response'}
                      >
                        {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                        {isPaused ? 'Resume' : 'Pause'}
                      </button>
                      <button 
                        className={clsx('btn h-[44px] px-3', isRecording && 'bg-red-500 text-white')}
                        onClick={() => {
                          if (isRecording) stopVoiceInput()
                          else startVoiceInput()
                        }}
                        disabled={isSending}
                      >
                        <Mic className="h-4 w-4" />
                      </button>
                    </div>

                    {/* Removed "Dataset ready" pill per request */}
                  </div>
                </div>
              </div>
            </div>
          )}

          {tab === 'dataset' && (
            <div className="h-full overflow-auto scrollbar">
              <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
                <SectionTitle
                  icon={<FlaskConical className="h-5 w-5" />}
                  title="Train a model"
                  subtitle="Upload a CSV, then run the automated pipeline (analysis → preprocessing → training → evaluation)."
                />

                <div className="glass rounded-2xl p-4">
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">Dataset</div>
                      <div className="text-xs text-slate-600 dark:text-slate-300">
                        {datasetPath ? (
                          <>
                            <span className="text-emerald-200">Uploaded:</span> {datasetName ?? datasetPath}
                          </>
                        ) : (
                          'No dataset uploaded yet'
                        )}
                      </div>
                    </div>

                    <label className={clsx('btn', isUploading && 'opacity-60 cursor-not-allowed')}>
                      <Paperclip className="h-4 w-4" />
                      {isUploading ? 'Uploading…' : 'Upload CSV'}
                      <input
                        type="file"
                        accept=".csv,text/csv"
                        className="hidden"
                        disabled={isUploading}
                        onChange={(e) => {
                          const f = e.target.files?.[0]
                          if (f) void onUpload(f)
                        }}
                      />
                    </label>
                  </div>
                </div>

                <div className="glass rounded-2xl p-4">
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">Run training</div>
                      <div className="text-xs text-slate-600 dark:text-slate-300">
                        This calls <code>/train_model</code> on the backend using your uploaded dataset path.
                      </div>
                    </div>
                    <button className="btn btn-primary" disabled={isTraining} onClick={() => void runTraining()}>
                      {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                      Train
                    </button>
                  </div>

                  {training && (
                    <div className="mt-4 grid gap-3">
                      <div className="glass rounded-xl p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">Status</div>
                        <div className={clsx('mt-1 text-sm', training.status === 'success' ? 'text-emerald-200' : 'text-rose-200')}>
                          {training.status}
                        </div>
                        {training.message && <div className="mt-1 text-xs text-slate-700 dark:text-slate-200">{training.message}</div>}
                      </div>

                      {training.logs && training.logs.length > 0 && (
                        <div className="glass rounded-xl p-3">
                          <div className="text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">Logs</div>
                          <ul className="mt-2 space-y-1 text-sm text-slate-200">
                            {training.logs.map((l, i) => (
                              <li key={i} className="flex gap-2">
                                <span className="text-slate-600 dark:text-slate-300">•</span>
                                <span className="break-words">{l}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {!!(training.training_results || training.evaluation) && (
                        <div className="grid md:grid-cols-2 gap-3">
                          <JsonCard title="Training results" value={training.training_results} />
                          <JsonCard title="Evaluation" value={training.evaluation} />
                        </div>
                      )}

                      {training.export_path && (
                        <div className="glass rounded-xl p-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">Export</div>
                            <div className="text-sm text-slate-700 dark:text-slate-200 truncate">{training.export_path}</div>
                            <div className="text-xs text-slate-600 dark:text-slate-300">Backend path to the zip (download depends on backend serving static files).</div>
                          </div>
                          <button
                            className="btn"
                            onClick={() => {
                              navigator.clipboard.writeText(training.export_path!)
                              toast.success('Copied export path')
                            }}
                          >
                            <Download className="h-4 w-4" />
                            Copy path
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {tab === 'predict' && (
            <div className="h-full overflow-auto scrollbar">
              <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
                <SectionTitle
                  icon={<BrainCircuit className="h-5 w-5" />}
                  title="Predict"
                  subtitle="Send a JSON object of feature values to the trained pipeline."
                />

                <div className="glass rounded-2xl p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">Input JSON</div>
                      <div className="text-xs text-slate-600 dark:text-slate-300">
                        Must match your dataset feature names (excluding the target column).
                      </div>
                    </div>
                    <button className="btn btn-primary" disabled={isPredicting} onClick={() => void runPredict()}>
                      {isPredicting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                      Predict
                    </button>
                  </div>

                  <textarea
                    className="textarea mt-3 h-48 font-mono text-xs"
                    value={predictJson}
                    onChange={(e) => setPredictJson(e.target.value)}
                    spellCheck={false}
                  />
                </div>

                {predictResult && (
                  <div className="glass rounded-2xl p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium">Result</div>
                        <div className="text-xs text-slate-600 dark:text-slate-300">
                          From <code>/predict</code>
                        </div>
                      </div>
                      <button
                        className="btn"
                        onClick={() => {
                          navigator.clipboard.writeText(prettyJson(predictResult))
                          toast.success('Copied')
                        }}
                      >
                        <Download className="h-4 w-4" />
                        Copy
                      </button>
                    </div>
                    <pre className="mt-3 text-xs">{prettyJson(predictResult)}</pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {settingsOpen && (
        <SettingsModal
          onClose={() => setSettingsOpen(false)}
          apiUrl={apiBaseUrl()}
          activeModelId={activeModelId}
          models={models}
          exportsList={exportsList}
          onRefresh={() => void refreshModelsAndExports()}
          onSetActive={(id) => void setBackendActiveModel(id)}
          chatMessages={messages}
          theme={theme}
          onThemeChange={setTheme}
          onLogout={() => {
            setIsAuthenticated(false)
            setShowLogin(false)
            toast.success('Logged out')
          }}
        />
      )}

      {showModelSearch && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/60" onClick={() => setShowModelSearch(false)} />
          <div className="absolute inset-0 p-4 md:p-8 grid place-items-center">
            <div className="glass w-full max-w-4xl rounded-2xl overflow-hidden max-h-[80vh]">
              <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between gap-3">
                <div className="text-sm font-semibold">Search Trained Models</div>
                <button className="btn p-2" onClick={() => setShowModelSearch(false)}>
                  ×
                </button>
              </div>
              <div className="p-4">
                <div className="mb-4">
                  <input
                    type="text"
                    className="input"
                    placeholder="Search models..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <div className="grid gap-3 max-h-[60vh] overflow-auto">
                  {models.length === 0 && (
                    <div className="text-center text-gray-600 dark:text-gray-400 py-8">
                      No trained models found. Train a model first!
                    </div>
                  )}
                  {models
                    .filter((model) => {
                      if (!searchQuery) return true
                      const q = String(searchQuery).toLowerCase()
                      return (
                        model.display_name?.toLowerCase().includes(q) ||
                        model.dataset_name?.toLowerCase().includes(q) ||
                        model.id?.toLowerCase().includes(q)
                      )
                    })
                    .map((model, idx) => (
                      <div key={idx} className="glass rounded-xl p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">
                              {model.display_name ?? model.dataset_name ?? model.id}
                            </div>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              Dataset: {model.dataset_name || 'Unknown'}
                            </div>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              Created: {new Date(model.created_at).toLocaleDateString()}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              className={clsx('btn', activeModelId === model.id && 'btn-primary')}
                              onClick={() => {
                                setActiveModelId(model.id)
                                void setBackendActiveModel(model.id)
                                setShowModelSearch(false)
                                toast.success(`Switched to model: ${model.display_name ?? model.id}`)
                              }}
                              title="Use this model"
                            >
                              Select
                            </button>
                            <button
                              className="btn"
                              title="Delete model"
                              onClick={(e) => {
                                e.stopPropagation()
                                void deleteModelAndArtifacts(model.id)
                              }}
                            >
                              <Trash2 className="h-4 w-4 text-rose-200" />
                            </button>
                            <button
                              className="btn"
                              title="Export ZIP"
                              onClick={() => {
                                const name = `ml_project_${model.id}`
                                apiFetchJson<{ status: string; zip_path?: string; message?: string }>(
                                  `/exports/create?model_id=${encodeURIComponent(model.id)}&project_name=${encodeURIComponent(name)}`,
                                  { method: 'POST' },
                                )
                                  .then((res) => {
                                    if (res.status !== 'success' || !res.zip_path) {
                                      throw new Error(res.message || 'Export failed')
                                    }
                                    const filename = res.zip_path.split(/[/\\]/).pop()
                                    if (!filename) throw new Error('Invalid export path from backend')
                                    const url = `${apiBaseUrl().replace(/\/$/, '')}/static/exports/${encodeURIComponent(filename)}`
                                    window.open(url, '_blank')
                                    toast.success('Export ZIP downloaded')
                                    void refreshModelsAndExports()
                                  })
                                  .catch((err) => toast.error(err instanceof Error ? err.message : String(err)))
                              }}
                            >
                              <Download className="h-4 w-4" />
                              ZIP
                            </button>
                            <button
                              className="btn"
                              title="Download visualizations ZIP"
                              onClick={() => {
                                apiFetchJson<{ status: string; zip_path?: string; message?: string }>(
                                  '/visualizations/export_all',
                                  { method: 'GET' },
                                )
                                  .then((res) => {
                                    if (res.status !== 'success' || !res.zip_path) {
                                      throw new Error(res.message || 'No visualizations available')
                                    }
                                    const filename = res.zip_path.split(/[/\\]/).pop()
                                    if (!filename) throw new Error('Invalid visualization zip path')
                                    const url = `${apiBaseUrl().replace(/\/$/, '')}/static/exports/${encodeURIComponent(filename)}`
                                    window.open(url, '_blank')
                                    toast.success('Visualizations ZIP downloaded')
                                  })
                                  .catch((err) => toast.error(err instanceof Error ? err.message : String(err)))
                              }}
                            >
                              <BarChart3 className="h-4 w-4" />
                              Visualizations
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function SettingsModal(props: {
  onClose: () => void
  apiUrl: string
  activeModelId: string | null
  models: any[]
  exportsList: any[]
  onRefresh: () => void
  onSetActive: (id: string) => void
  chatMessages: ChatMessage[]
  theme: 'light' | 'dark'
  onThemeChange: (t: 'light' | 'dark') => void
  onLogout: () => void
}) {
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={props.onClose} />
      <div className="absolute inset-0 p-4 md:p-8 grid place-items-center">
        <div className="glass w-full max-w-3xl rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between gap-3">
            <div className="text-sm font-semibold">Settings</div>
            <div className="flex items-center gap-2">
              <button className="btn" onClick={props.onRefresh}>
                Refresh
              </button>
              <button className="btn btn-primary" onClick={props.onClose}>
                Close
              </button>
            </div>
          </div>

          <div className="p-4 grid gap-4 max-h-[80vh] overflow-auto scrollbar">
            <div className="glass rounded-xl p-3">
              <div className="text-sm font-semibold mb-3">Change Theme</div>
              <div className="flex items-center gap-4">
                <label className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="theme"
                    checked={props.theme === 'light'}
                    onChange={() => props.onThemeChange('light')}
                  />
                  Light
                </label>
                <label className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="theme"
                    checked={props.theme === 'dark'}
                    onChange={() => props.onThemeChange('dark')}
                  />
                  Dark
                </label>
              </div>
            </div>

            <div className="glass rounded-xl p-3">
              <div className="text-sm font-semibold mb-2">Change Active Model</div>
              <select
                className="input max-w-xs"
                value={props.activeModelId ?? ''}
                onChange={(e) => props.onSetActive(e.target.value)}
              >
                <option value="">Latest (default)</option>
                {props.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name ?? m.dataset_name ?? m.id}
                  </option>
                ))}
              </select>
            </div>

            <div className="glass rounded-xl p-3">
              <div className="text-sm font-semibold mb-2">Account</div>
              <button className="btn w-full bg-red-50 border-red-200 text-red-600 hover:bg-red-100" onClick={props.onLogout}>
                Log out
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SectionTitle(props: { icon: ReactNode; title: string; subtitle: string }) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-start gap-3">
        <div className="h-10 w-10 rounded-xl bg-indigo-500/15 grid place-items-center border border-indigo-400/25">
          {props.icon}
        </div>
        <div>
          <div className="text-lg font-semibold">{props.title}</div>
          <div className="text-sm text-slate-600 dark:text-slate-300">{props.subtitle}</div>
        </div>
      </div>
    </div>
  )
}

function ChatBubble(props: { role: Role; content: string; streaming?: boolean; revealed?: number }) {
  const isUser = props.role === 'user'
  const shown = props.streaming ? props.content.slice(0, props.revealed ?? 0) : props.content
  return (
    <div className={clsx('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={clsx(
          'max-w-[min(720px,100%)] rounded-2xl px-4 py-3 border fade-up shadow-sm',
          isUser
            ? 'bg-blue-600 text-white border-blue-600'
            : 'bg-white dark:bg-slate-900 border-gray-200 dark:border-slate-800 text-slate-900 dark:text-slate-100',
        )}
      >
        <div className="flex items-center gap-2 mb-2 text-xs text-slate-600 dark:text-slate-300">
          {isUser ? (
            <>
              <span className="pill bg-indigo-500/15 border-indigo-400/25 text-indigo-100">You</span>
            </>
          ) : (
            <>
              <span className="pill">
                <Bot className="h-3.5 w-3.5" />
                SAVESA
              </span>
            </>
          )}
        </div>

        <div className="md text-sm">
          {props.streaming ? (
            <pre className="whitespace-pre-wrap font-sans m-0">{shown}</pre>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{shown}</ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  )
}

function JsonCard(props: { title: string; value: unknown }) {
  return (
    <div className="glass rounded-xl p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">{props.title}</div>
        <span className="pill">
          <Sparkles className="h-3.5 w-3.5" />
          JSON
        </span>
      </div>
      <pre className="mt-2 text-xs overflow-auto">{prettyJson(props.value)}</pre>
    </div>
  )
}

