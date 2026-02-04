import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Modal, Select, Slider, Tabs, Tag, Tooltip } from 'antd'
import type { TabsProps } from 'antd'
import { Bar, Line } from '@ant-design/plots'

type World = { id: string; name: string; snapshot: any; version: number }

type Message = {
  id: string
  role: string
  content: string
  emotion_label?: string
  emotion_confidence?: number
  emotion_method?: string
  desire_score?: number
  topic_tag?: string
  turn_index: number
  created_at?: string
}

type Stats = {
  total_messages: number
  total_turns: number
  avg_desire: number
  emotion_distribution: Record<string, number>
  lexical_diversity: { ttr: number; mtld: number }
  topic_diversity: { unique: number; entropy: number }
  desire_series: number[]
  avg_turn_duration_sec: number
  trend_summary: string
  trend_direction: string
  trend_delta: number
  trend_window: number
}

type MemoryConfig = {
  chat_id: string
  add_mode: string
  search_mode: string
  supported_add_modes: string[]
  supported_search_modes: string[]
}

type SessionConfig = {
  chat_id: string
  session_time: string
}

type KV = { key: string; value: string }

type WorldForm = {
  sceneSetting: string
  ontologyUserProfile: KV[]
  ontologyUserState: KV[]
  plotHooks: string[]
  norms: string[]
  beliefs: string[]
  memorySemantic: string[]
  memoryEpisodic: string[]
  memoryWorking: string[]
}

type NpcForm = {
  mode: 'default' | 'volcengine'
  roleName: string
  userName: string
  initRole: string
  userInfo: string
  goldenSp: string
  notes: string[]
}

type UserForm = {
  displayName: string
  background: string
  persona: string
  goals: string
  constraints: string
  notes: string[]
}

const API_BASE = (import.meta as any).env.VITE_API_BASE || 'http://localhost:8000'
const { TextArea } = Input

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || 'Request failed')
    }
    return res.json()
  } catch (err: any) {
    throw new Error(err?.message || 'API unreachable')
  }
}

function cleanList(items: string[]): string[] {
  return items.map((item) => item.trim()).filter(Boolean)
}

function listFromAny(value: any): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => String(item))
}

function objectToKv(value: any): KV[] {
  if (!value || typeof value !== 'object') return []
  return Object.entries(value).map(([key, val]) => ({
    key,
    value: typeof val === 'string' ? val : JSON.stringify(val),
  }))
}

function parseMaybeJson(value: string): any {
  const trimmed = value.trim()
  if (!trimmed) return ''
  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    try {
      return JSON.parse(trimmed)
    } catch (err) {
      return value
    }
  }
  return value
}

function kvToObject(items: KV[]): Record<string, any> {
  const result: Record<string, any> = {}
  items.forEach(({ key, value }) => {
    const k = key.trim()
    if (!k) return
    result[k] = parseMaybeJson(value)
  })
  return result
}

function toKVExtras(profile: Record<string, any>, ignore: Set<string>): KV[] {
  return Object.entries(profile)
    .filter(([key, value]) => !ignore.has(key) && value !== undefined)
    .map(([key, value]) => ({
      key,
      value: typeof value === 'string' ? value : JSON.stringify(value),
    }))
}

function worldFormFromSnapshot(snapshot: any): WorldForm {
  const scene = snapshot?.scene ?? {}
  const ontology = scene?.ontology ?? {}
  const memory = snapshot?.memory ?? {}
  return {
    sceneSetting: scene?.setting ?? '',
    ontologyUserProfile: objectToKv(ontology?.user_profile),
    ontologyUserState: objectToKv(ontology?.user_state_machine),
    plotHooks: listFromAny(snapshot?.event_schema?.plot_hooks),
    norms: listFromAny(snapshot?.norms),
    beliefs: listFromAny(snapshot?.beliefs),
    memorySemantic: listFromAny(memory?.semantic),
    memoryEpisodic: listFromAny(memory?.episodic),
    memoryWorking: listFromAny(memory?.working),
  }
}

function applyWorldForm(base: any, form: WorldForm): any {
  const next = { ...(base || {}) }
  const scene = { ...(next.scene || {}) }
  scene.setting = form.sceneSetting
  scene.ontology = {
    ...(scene.ontology || {}),
    user_profile: kvToObject(form.ontologyUserProfile),
    user_state_machine: kvToObject(form.ontologyUserState),
  }
  next.scene = scene
  next.event_schema = { ...(next.event_schema || {}), plot_hooks: cleanList(form.plotHooks) }
  next.norms = cleanList(form.norms)
  next.beliefs = cleanList(form.beliefs)
  next.memory = {
    ...(next.memory || {}),
    semantic: cleanList(form.memorySemantic),
    episodic: cleanList(form.memoryEpisodic),
    working: cleanList(form.memoryWorking),
  }
  return next
}

function Hint({ text }: { text: string }) {
  return (
    <Tooltip title={text}>
      <span className="hint-icon">?</span>
    </Tooltip>
  )
}

function ListEditor({
  items,
  onChange,
  placeholder,
  rows = 4,
  disabled,
}: {
  items: string[]
  onChange: (items: string[]) => void
  placeholder?: string
  rows?: number
  disabled?: boolean
}) {
  return (
    <TextArea
      autoSize={{ minRows: rows, maxRows: rows + 2 }}
      disabled={disabled}
      value={items.join('\n')}
      placeholder={placeholder}
      onChange={(e) => onChange(cleanList(e.target.value.split('\n')))}
    />
  )
}

function KeyValueEditor({
  items,
  onChange,
  emptyHint,
  disabled,
}: {
  items: KV[]
  onChange: (items: KV[]) => void
  emptyHint?: string
  disabled?: boolean
}) {
  const update = (index: number, field: 'key' | 'value', value: string) => {
    const next = items.map((item, i) => (i === index ? { ...item, [field]: value } : item))
    onChange(next)
  }
  const addRow = () => onChange([...items, { key: '', value: '' }])
  const removeRow = (index: number) => onChange(items.filter((_, i) => i !== index))

  return (
    <div className="kv-editor">
      {items.length === 0 ? <div className="kv-empty">{emptyHint || '暂无字段'}</div> : null}
      {items.map((item, index) => (
        <div key={`${item.key}-${index}`} className="kv-row">
          <Input
            value={item.key}
            disabled={disabled}
            onChange={(e) => update(index, 'key', e.target.value)}
            placeholder="字段名"
          />
          <TextArea
            autoSize={{ minRows: 2, maxRows: 3 }}
            value={item.value}
            disabled={disabled}
            onChange={(e) => update(index, 'value', e.target.value)}
            placeholder="值"
          />
          <Button type="text" className="ghost icon" onClick={() => removeRow(index)} disabled={disabled}>
            移除
          </Button>
        </div>
      ))}
      <Button type="default" className="ghost" onClick={addRow} disabled={disabled}>
        新增字段
      </Button>
    </div>
  )
}

function ChartLine({ values }: { values: number[] }) {
  if (!values || values.length === 0) {
    return <div className="chart-empty">暂无欲望数据</div>
  }
  const data = values.map((v, i) => ({ round: i + 1, value: v }))
  return (
    <Line
      data={data}
      xField="round"
      yField="value"
      autoFit
      height={160}
      smooth
      color="var(--primary)"
      xAxis={{ label: { style: { fill: 'var(--muted)' } } }}
      yAxis={{ min: 0, max: 10, tickCount: 6 }}
      tooltip={{ showMarkers: true }}
      animation={{ appear: { animation: 'path-in', duration: 800 } }}
    />
  )
}

function ChartBars({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data || {})
  if (entries.length === 0) {
    return <div className="chart-empty">暂无情绪数据</div>
  }
  const chartData = entries.map(([key, val]) => ({ label: key, value: val }))
  return (
    <Bar
      data={chartData}
      xField="value"
      yField="label"
      autoFit
      height={200}
      color="var(--primary)"
      xAxis={{ tickCount: 5 }}
      yAxis={{ label: { style: { fill: 'var(--muted)' } } }}
      animation={{ appear: { animation: 'scale-in-x', duration: 600 } }}
    />
  )
}

function TrendBadge({ direction, delta, window }: { direction: string; delta: number; window: number }) {
  const label =
    direction === 'up' ? '上升' : direction === 'down' ? '下降' : '平稳'
  const deltaText = Number.isFinite(delta) ? `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}` : '0.0'
  const windowText = window ? `近${window}轮` : '近几轮'
  return (
    <span className={`trend-badge ${direction}`}>
      {label} {deltaText} · {windowText}
    </span>
  )
}

export default function App() {
  const [theme, setTheme] = useState<'town' | 'dream' | 'gothic' | 'pixel'>('town')
  const [world, setWorld] = useState<World | null>(null)
  const [worldForm, setWorldForm] = useState<WorldForm>({
    sceneSetting: '',
    ontologyUserProfile: [],
    ontologyUserState: [],
    plotHooks: [],
    norms: [],
    beliefs: [],
    memorySemantic: [],
    memoryEpisodic: [],
    memoryWorking: [],
  })
  const [npcForm, setNpcForm] = useState<NpcForm>({
    mode: 'default',
    roleName: '林一',
    userName: '小周',
    initRole: '简介: 你是林一，一位自由插画师。性格开朗，待人温柔体贴。',
    userInfo: '用户叫小周，是你的未婚妻，你通常称呼她为宝贝或周周。',
    goldenSp:
      '你使用口语进行表达，必要时可用括号描述动作和情绪。\n' +
      '你需要尽可能引导用户跟你进行交流，你不应该表现地太AI。\n' +
      '每次回复推进剧情，提出问题或下一步行动。',
    notes: [],
  })
  const [npcExtras, setNpcExtras] = useState<KV[]>([])
  const [userForm, setUserForm] = useState<UserForm>({
    displayName: '',
    background: '',
    persona: '',
    goals: '',
    constraints: '',
    notes: [],
  })
  const [userExtras, setUserExtras] = useState<KV[]>([])
  const [npcActorId, setNpcActorId] = useState('')
  const [userActorId, setUserActorId] = useState('')
  const [chatId, setChatId] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [inputText, setInputText] = useState('')
  const [inputDesire, setInputDesire] = useState(6)
  const [stats, setStats] = useState<Stats | null>(null)
  const [memoryConfig, setMemoryConfig] = useState<MemoryConfig | null>(null)
  const [sessionConfig, setSessionConfig] = useState<SessionConfig | null>(null)
  const [downloadId, setDownloadId] = useState('')
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)
  const [modal, setModal] = useState<null | 'world' | 'chat'>(null)
  const [modalValue, setModalValue] = useState('')
  const [notice, setNotice] = useState<{ title: string; body: string } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<any>(null)

  const messageList = useMemo(() => {
    return [...messages].sort((a, b) => {
      const ta = a.created_at ? new Date(a.created_at).getTime() : 0
      const tb = b.created_at ? new Date(b.created_at).getTime() : 0
      if (ta && tb && ta !== tb) return ta - tb
      return a.turn_index - b.turn_index
    })
  }, [messages])

  useEffect(() => {
    document.body.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, chatId])

  const formatDuration = (seconds: number) => {
    if (!seconds || Number.isNaN(seconds)) return '0s'
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${mins}m ${secs}s`
  }

  const formatTime = (value?: string) => {
    if (!value) return ''
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return ''
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }

  const composerHint = !chatId
    ? '请先创建或加载对话。'
    : !inputText.trim()
      ? '输入内容后回车发送，Shift+Enter 换行。'
      : '回车发送，Shift+Enter 换行。'

  const refreshStats = async (id: string) => {
    if (!id) return
    try {
      const data = await api<Stats>(`/chats/${id}/stats`)
      setStats(data)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const loadMemoryConfig = async (id: string) => {
    if (!id) return
    try {
      const data = await api<MemoryConfig>(`/chats/${id}/memory/config`)
      setMemoryConfig(data)
    } catch (err: any) {
      // ignore config load errors
    }
  }

  const updateMemoryConfig = async () => {
    if (!chatId || !memoryConfig) return
    try {
      setBusy(true)
      setStatus('保存记忆配置中...')
      const data = await api<MemoryConfig>(`/chats/${chatId}/memory/config`, {
        method: 'PUT',
        body: JSON.stringify({
          add_mode: memoryConfig.add_mode,
          search_mode: memoryConfig.search_mode,
        }),
      })
      setMemoryConfig(data)
      setStatus('记忆配置已更新')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const loadSessionConfig = async (id: string) => {
    if (!id) return
    try {
      const data = await api<SessionConfig>(`/chats/${id}/session`)
      setSessionConfig(data)
    } catch (err: any) {
      // ignore config load errors
    }
  }

  const updateSessionConfig = async () => {
    if (!chatId || !sessionConfig) return
    try {
      setBusy(true)
      setStatus('保存 session 时间中...')
      const data = await api<SessionConfig>(`/chats/${chatId}/session`, {
        method: 'PUT',
        body: JSON.stringify({ session_time: sessionConfig.session_time }),
      })
      setSessionConfig(data)
      setStatus('Session 时间已更新')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const downloadSession = async () => {
    const id = downloadId.trim()
    if (!id) return
    try {
      setBusy(true)
      setStatus('准备下载...')
      const data = await api<any>(`/chats/${id}/export`)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `session_${id}.json`
      anchor.click()
      setTimeout(() => URL.revokeObjectURL(url), 500)
      setStatus('已下载')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const statsTabs: TabsProps['items'] = [
    {
      key: 'overview',
      label: '总览',
      children: stats ? (
        <div className="stats-grid">
          <div className="stat-card">
            <span>轮次</span>
            <strong>{stats.total_turns}</strong>
          </div>
          <div className="stat-card">
            <span>平均欲望</span>
            <strong>{stats.avg_desire}</strong>
          </div>
          <div className="stat-card">
            <span>平均时长</span>
            <strong>{formatDuration(stats.avg_turn_duration_sec)}</strong>
          </div>
          <div className="stat-card wide">
            <span>趋势摘要</span>
            <strong>{stats.trend_summary}</strong>
            <TrendBadge
              direction={stats.trend_direction}
              delta={stats.trend_delta}
              window={stats.trend_window}
            />
          </div>
          <div className="stat-card">
            <span>TTR</span>
            <strong>{stats.lexical_diversity.ttr}</strong>
          </div>
          <div className="stat-card">
            <span>MTLD</span>
            <strong>{stats.lexical_diversity.mtld}</strong>
          </div>
          <div className="stat-card">
            <span>话题数</span>
            <strong>{stats.topic_diversity.unique}</strong>
          </div>
          <div className="stat-card">
            <span>话题熵</span>
            <strong>{stats.topic_diversity.entropy}</strong>
          </div>
        </div>
      ) : (
        <div className="empty">暂无数据</div>
      ),
    },
    {
      key: 'desire',
      label: '欲望趋势',
      children: stats ? (
        <div className="stats-panel">
          <div className="stat-row">
            <span>趋势摘要: {stats.trend_summary}</span>
            <TrendBadge
              direction={stats.trend_direction}
              delta={stats.trend_delta}
              window={stats.trend_window}
            />
          </div>
          <ChartLine values={stats.desire_series} />
        </div>
      ) : (
        <div className="empty">暂无数据</div>
      ),
    },
    {
      key: 'emotion',
      label: '情绪分布',
      children: stats ? (
        <div className="stats-panel">
          <ChartBars data={stats.emotion_distribution} />
        </div>
      ) : (
        <div className="empty">暂无数据</div>
      ),
    },
    {
      key: 'config',
      label: '配置',
      children: (
        <div className="config-panel">
          <div className="config-block">
            <div className="config-title">记忆模式</div>
            <div className="config-row">
              <div className="config-label">add_memory</div>
              <Select
                value={memoryConfig?.add_mode}
                options={(memoryConfig?.supported_add_modes || []).map((value) => ({
                  value,
                  label: value,
                }))}
                disabled={!chatId || busy || !memoryConfig}
                onChange={(value) =>
                  setMemoryConfig((prev) =>
                    prev ? { ...prev, add_mode: value } : prev
                  )
                }
              />
            </div>
            <div className="config-row">
              <div className="config-label">search_memory</div>
              <Select
                value={memoryConfig?.search_mode}
                options={(memoryConfig?.supported_search_modes || []).map((value) => ({
                  value,
                  label: value,
                }))}
                disabled={!chatId || busy || !memoryConfig}
                onChange={(value) =>
                  setMemoryConfig((prev) =>
                    prev ? { ...prev, search_mode: value } : prev
                  )
                }
              />
            </div>
            <Button className="ghost" onClick={updateMemoryConfig} disabled={!chatId || busy || !memoryConfig}>
              保存记忆配置
            </Button>
          </div>

          <div className="config-block">
            <div className="config-title">Session 时间</div>
            <div className="config-row">
              <div className="config-label">session_time</div>
              <Input
                value={sessionConfig?.session_time || ''}
                disabled={!chatId || busy || !sessionConfig}
                onChange={(e) =>
                  setSessionConfig((prev) =>
                    prev ? { ...prev, session_time: e.target.value } : prev
                  )
                }
                placeholder="YYYY-MM-DD HH:mm:ss"
              />
            </div>
            <Button className="ghost" onClick={updateSessionConfig} disabled={!chatId || busy || !sessionConfig}>
              保存 Session 时间
            </Button>
          </div>

          <div className="config-block">
            <div className="config-title">导出会话</div>
            <div className="config-row">
              <div className="config-label">session_id</div>
              <Input
                value={downloadId}
                disabled={busy}
                onChange={(e) => setDownloadId(e.target.value)}
                placeholder="输入 session/chat id"
              />
            </div>
            <Button type="primary" onClick={downloadSession} disabled={busy || !downloadId.trim()}>
              下载 JSON
            </Button>
          </div>
        </div>
      ),
    },
  ]

  const loadWorldById = async (id: string, silent?: boolean) => {
    if (!id.trim()) return
    try {
      if (!silent) {
        setBusy(true)
        setStatus('加载世界中...')
      }
      const data = await api<World>(`/worlds/${id.trim()}`)
      setWorld(data)
      setWorldForm(worldFormFromSnapshot(data.snapshot))
      if (!silent) setStatus('世界已加载')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      if (!silent) setBusy(false)
    }
  }

  const loadChatById = async (id: string) => {
    try {
      setBusy(true)
      setStatus('加载对话中...')
      const data = await api<any>(`/chats/${id}`)
      setChatId(id)
      setDownloadId(id)
      setMessages(data.messages || [])
      if (data.world_id) {
        await loadWorldById(data.world_id, true)
      }
      try {
        const actors = await api<any>(`/chats/${id}/actors`)
        if (actors?.npc?.profile) {
          const profile = actors.npc.profile
          setNpcActorId(actors.npc.id || '')
          setNpcForm((prev) => ({
            ...prev,
            mode: profile.mode || prev.mode,
            roleName: profile.role_name || prev.roleName,
            userName: profile.user_name || prev.userName,
            initRole: profile.init_role_sp || prev.initRole,
            userInfo: profile.user_info || prev.userInfo,
            goldenSp: profile.golden_sp || prev.goldenSp,
            notes: Array.isArray(profile.notes) ? profile.notes.map(String) : prev.notes,
          }))
          const ignore = new Set([
            'role',
            'mode',
            'role_name',
            'user_name',
            'init_role_sp',
            'user_info',
            'golden_sp',
            'notes',
            'chat_id',
          ])
          setNpcExtras(toKVExtras(profile, ignore))
        }
        if (actors?.user?.profile) {
          const profile = actors.user.profile
          setUserActorId(actors.user.id || '')
          setUserForm((prev) => ({
            ...prev,
            displayName: profile.display_name || prev.displayName,
            background: profile.background || prev.background,
            persona: profile.persona || prev.persona,
            goals: profile.goals || prev.goals,
            constraints: profile.constraints || prev.constraints,
            notes: Array.isArray(profile.notes) ? profile.notes.map(String) : prev.notes,
          }))
          const ignore = new Set([
            'role',
            'display_name',
            'background',
            'persona',
            'goals',
            'constraints',
            'notes',
            'chat_id',
          ])
          setUserExtras(toKVExtras(profile, ignore))
        }
      } catch (err) {
        // Ignore actor auto-fill failures to avoid blocking chat load
      }
      await loadMemoryConfig(id)
      await loadSessionConfig(id)
      await refreshStats(id)
      setStatus('对话已加载')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const createChatForWorld = async (worldId: string) => {
    const data = await api<{ id: string }>('/chats', {
      method: 'POST',
      body: JSON.stringify({ world_id: worldId }),
    })
    try {
      await api(`/chats/${data.id}/auto/one`, { method: 'POST' })
    } catch (err) {
      // ignore npc first line failure
    }
    await loadChatById(data.id)
    setStatus('对话已创建')
    setNotice({
      title: '对话已创建',
      body: `Chat ID: ${data.id}\n下次可在对话窗口“加载对话”中恢复。`,
    })
  }

  const createWorld = async () => {
    setError('')
    setStatus('生成世界中...')
    setBusy(true)
    try {
      const data = await api<World>('/worlds/generate', {
        method: 'POST',
        body: JSON.stringify({
          seed_prompt: '',
          style:
            theme === 'town' ? 'pixel' : theme === 'dream' ? 'dreamy' : theme === 'pixel' ? 'pixel' : 'gothic',
        }),
      })
      setWorld(data)
      setWorldForm(worldFormFromSnapshot(data.snapshot))
      setStatus('世界已生成')
      setNotice({
        title: 'World 已生成',
        body: `World ID: ${data.id}\n下次可在左上角“加载世界”中恢复。`,
      })
      await createChatForWorld(data.id)
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const updateWorld = async () => {
    if (!world) return
    setError('')
    setStatus('更新世界中...')
    setBusy(true)
    try {
      const snapshot = applyWorldForm(world.snapshot, worldForm)
      const data = await api<World>(`/worlds/${world.id}`, {
        method: 'PATCH',
        body: JSON.stringify({
          expected_version: world.version,
          event_type: 'world_update',
          payload: { source: 'ui' },
          snapshot,
        }),
      })
      setWorld(data)
      setWorldForm(worldFormFromSnapshot(data.snapshot))
      setStatus('世界已更新')
      setNotice({
        title: 'World 已保存',
        body: `World ID: ${data.id}\n下次可在左上角“加载世界”中恢复。`,
      })
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const createChat = async () => {
    if (!world) return
    try {
      setBusy(true)
      setStatus('创建对话中...')
      await createChatForWorld(world.id)
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const sendMessage = async () => {
    if (!chatId) {
      setError('请先创建或加载对话')
      return
    }
    if (!inputText.trim()) {
      setError('请输入要发送的消息')
      return
    }
    const text = inputText
    setInputText('')
    try {
      setBusy(true)
      setStatus('发送中...')
      const tempId = `temp-${Date.now()}`
      const tempTurn = (() => {
        if (!messages.length) return 1
        const sorted = [...messages].sort((a, b) => {
          const ta = a.created_at ? new Date(a.created_at).getTime() : 0
          const tb = b.created_at ? new Date(b.created_at).getTime() : 0
          if (ta && tb && ta !== tb) return ta - tb
          return a.turn_index - b.turn_index
        })
        const last = sorted[sorted.length - 1]
        if (last.role !== 'user') return last.turn_index
        return last.turn_index + 1
      })()
      setMessages((prev) => [
        ...prev,
        {
          id: tempId,
          role: 'user',
          content: text,
          turn_index: tempTurn,
          created_at: new Date().toISOString(),
        },
      ])
      await api(`/chats/${chatId}/message`, {
        method: 'POST',
        body: JSON.stringify({ role: 'user', content: text, desire_score: inputDesire }),
      })
      try {
        await api(`/chats/${chatId}/auto/one`, { method: 'POST' })
      } catch (err) {
        // Ignore auto-reply failures to keep user message persisted
      }
      await loadChatById(chatId)
      setStatus('已发送')
    } catch (err: any) {
      setError(err.message)
      setInputText(text)
      setMessages((prev) => prev.filter((m) => !m.id.startsWith('temp-')))
      setStatus('')
    } finally {
      setBusy(false)
      composerRef.current?.focus()
    }
  }

  const autoRound = async (steps: number) => {
    if (!chatId) {
      setError('请先创建或加载对话')
      return
    }
    try {
      setBusy(true)
      setStatus(`自动对话（${steps}轮）...`)
      await api(`/chats/${chatId}/auto/step`, { method: 'POST', body: JSON.stringify({ max_steps: steps }) })
      await loadChatById(chatId)
      setStatus('完成')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const autoUntil = async () => {
    if (!chatId) {
      setError('请先创建或加载对话')
      return
    }
    try {
      setBusy(true)
      setStatus('自动对话中...')
      await api(`/chats/${chatId}/auto/until`, {
        method: 'POST',
        body: JSON.stringify({ max_steps: 30, desire_stop: 5 }),
      })
      await loadChatById(chatId)
      setStatus('完成')
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const buildNpcProfile = () => ({
    role: 'npc',
    mode: npcForm.mode,
    role_name: npcForm.roleName,
    user_name: npcForm.userName,
    init_role_sp: npcForm.initRole,
    user_info: npcForm.userInfo,
    golden_sp: npcForm.goldenSp,
    notes: cleanList(npcForm.notes),
    ...kvToObject(npcExtras),
    ...(chatId ? { chat_id: chatId } : {}),
  })

  const buildUserProfile = () => ({
    role: 'user',
    display_name: userForm.displayName,
    background: userForm.background,
    persona: userForm.persona,
    goals: userForm.goals,
    constraints: userForm.constraints,
    notes: cleanList(userForm.notes),
    ...kvToObject(userExtras),
    ...(chatId ? { chat_id: chatId } : {}),
  })

  const saveActor = async (role: 'npc' | 'user') => {
    const actorId = role === 'npc' ? npcActorId : userActorId
    const profile = role === 'npc' ? buildNpcProfile() : buildUserProfile()
    try {
      setBusy(true)
      setStatus(`保存 ${role} 设定中...`)
      if (actorId) {
        await api(`/actor/${actorId}`, { method: 'PATCH', body: JSON.stringify({ profile }) })
      } else {
        const data = await api<{ id: string }>(`/actor`, { method: 'POST', body: JSON.stringify({ profile }) })
        if (role === 'npc') setNpcActorId(data.id)
        if (role === 'user') setUserActorId(data.id)
      }
      setStatus(`已保存 ${role} 设定`)
      setNotice({
        title: '设定已保存',
        body: `已保存 ${role.toUpperCase()} 设定。\nChat ID: ${chatId || '未绑定'}\n下次可通过该 Chat ID 自动恢复。`,
      })
    } catch (err: any) {
      setError(err.message)
      setStatus('')
    } finally {
      setBusy(false)
    }
  }

  const openModal = (type: 'world' | 'chat') => {
    setModalValue('')
    setModal(type)
  }

  const confirmModal = async () => {
    const id = modalValue.trim()
    if (!id) return
    if (modal === 'world') await loadWorldById(id)
    if (modal === 'chat') await loadChatById(id)
    setModal(null)
  }

  const roleLabel = (role: string) => (role === 'npc' ? 'NPC' : role === 'user' ? '用户' : role)

  return (
    <div className="app">
      <div className="fx-layer" aria-hidden />
      <div className="fog-layer" aria-hidden />
      <header className="topbar">
        <div className="brand">
          <div className="title">记忆织城</div>
          <div className="subtitle">Town Dialogue Atelier</div>
        </div>
        <div className="top-meta">
          <div className="status-pill">{status || `API: ${API_BASE}`}</div>
          <div className="theme-switch">
            <span>风格</span>
            <Select
              value={theme}
              onChange={(value) => setTheme(value as any)}
              disabled={busy}
              size="small"
              options={[
                { value: 'town', label: '斯坦福小镇' },
                { value: 'dream', label: '造梦次元' },
                { value: 'gothic', label: '第五人格' },
                { value: 'pixel', label: '像素街区' },
              ]}
            />
          </div>
        </div>
      </header>

      {error ? <div className="banner error">操作失败：{error}</div> : null}

      <div className="board">
        <div className="column left">
          <section className="panel">
            <div className="panel-header">
              <div>
                <h3>World Model</h3>
                <div className="panel-sub">
                  版本 {world?.version ?? '-'} · {world?.id ? `ID ${world.id.slice(0, 8)}` : '未生成'}
                </div>
              </div>
              <div className="panel-actions">
                <Button type="primary" onClick={createWorld} disabled={busy}>
                  生成世界
                </Button>
                <Button className="ghost" onClick={() => openModal('world')} disabled={busy}>
                  加载世界
                </Button>
                <Button type="primary" onClick={updateWorld} disabled={!world || busy}>
                  保存更新
                </Button>
              </div>
            </div>
            <div className="panel-body scroll">
              {!world ? (
                <div className="empty">
                  请先生成或加载 World。
                  <Button className="ghost subtle" onClick={createWorld} disabled={busy}>
                    立即生成
                  </Button>
                </div>
              ) : (
                <div className="stack">
                  <details open>
                    <summary>场景 Scene</summary>
                    <label className="field">
                      场景描述
                      <Hint text="一句话描述当前舞台或世界的气质。" />
                    </label>
                    <TextArea
                      autoSize={{ minRows: 3, maxRows: 5 }}
                      value={worldForm.sceneSetting}
                      disabled={busy}
                      onChange={(e) => setWorldForm({ ...worldForm, sceneSetting: e.target.value })}
                    />
                    <label className="field">Ontology: 用户画像</label>
                    <KeyValueEditor
                      items={worldForm.ontologyUserProfile}
                      onChange={(items) => setWorldForm({ ...worldForm, ontologyUserProfile: items })}
                      disabled={busy}
                    />
                    <label className="field">Ontology: 用户状态机</label>
                    <KeyValueEditor
                      items={worldForm.ontologyUserState}
                      onChange={(items) => setWorldForm({ ...worldForm, ontologyUserState: items })}
                      disabled={busy}
                    />
                  </details>
                  <details>
                    <summary>事件表 Event Schema</summary>
                    <label className="field">Plot Hooks</label>
                    <ListEditor
                      items={worldForm.plotHooks}
                      onChange={(items) => setWorldForm({ ...worldForm, plotHooks: items })}
                      placeholder="每行一个未触发的剧情点"
                      disabled={busy}
                    />
                  </details>
                  <details>
                    <summary>Norms & Beliefs</summary>
                    <label className="field">Norms</label>
                    <ListEditor
                      items={worldForm.norms}
                      onChange={(items) => setWorldForm({ ...worldForm, norms: items })}
                      placeholder="每行一个规则或红线"
                      disabled={busy}
                    />
                    <label className="field">Beliefs</label>
                    <ListEditor
                      items={worldForm.beliefs}
                      onChange={(items) => setWorldForm({ ...worldForm, beliefs: items })}
                      placeholder="每行一个主观认知"
                      disabled={busy}
                    />
                  </details>
                  <details>
                    <summary>Memory</summary>
                    <label className="field">语义记忆</label>
                    <ListEditor
                      items={worldForm.memorySemantic}
                      onChange={(items) => setWorldForm({ ...worldForm, memorySemantic: items })}
                      placeholder="稳定事实 / 规则"
                      disabled={busy}
                    />
                    <label className="field">情节记忆</label>
                    <ListEditor
                      items={worldForm.memoryEpisodic}
                      onChange={(items) => setWorldForm({ ...worldForm, memoryEpisodic: items })}
                      placeholder="事件片段"
                      disabled={busy}
                    />
                    <label className="field">工作记忆</label>
                    <ListEditor
                      items={worldForm.memoryWorking}
                      onChange={(items) => setWorldForm({ ...worldForm, memoryWorking: items })}
                      placeholder="本轮目标 / 待办"
                      disabled={busy}
                    />
                  </details>
                </div>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h3>NPC 设定</h3>
                <div className="panel-sub">驱动 NPC 的视角与说话风格</div>
              </div>
              <div className="panel-actions">
                <Button type="primary" onClick={() => saveActor('npc')} disabled={busy}>
                  保存设定
                </Button>
              </div>
            </div>
            <div className="panel-body scroll">
              <details open>
                <summary>身份设定</summary>
                <label className="field">
                  模式
                  <Hint text="默认模式使用普通提示词，volcengine 会拼接 SP 结构。" />
                </label>
                <Select
                  value={npcForm.mode}
                  onChange={(value) => setNpcForm({ ...npcForm, mode: value as NpcForm['mode'] })}
                  disabled={busy}
                  size="small"
                  className="full"
                  options={[
                    { value: 'default', label: 'default' },
                    { value: 'volcengine', label: 'volcengine' },
                  ]}
                />
                <label className="field">角色名</label>
                <Input
                  value={npcForm.roleName}
                  disabled={busy}
                  onChange={(e) => setNpcForm({ ...npcForm, roleName: e.target.value })}
                  placeholder="NPC 名称"
                />
                <label className="field">用户称呼</label>
                <Input
                  value={npcForm.userName}
                  disabled={busy}
                  onChange={(e) => setNpcForm({ ...npcForm, userName: e.target.value })}
                  placeholder="NPC 如何称呼用户"
                />
              </details>
              <details>
                <summary>关系与背景</summary>
                <label className="field">角色初始设定</label>
                <TextArea
                  autoSize={{ minRows: 4, maxRows: 6 }}
                  disabled={busy}
                  value={npcForm.initRole}
                  onChange={(e) => setNpcForm({ ...npcForm, initRole: e.target.value })}
                />
                <label className="field">用户信息</label>
                <TextArea
                  autoSize={{ minRows: 3, maxRows: 5 }}
                  disabled={busy}
                  value={npcForm.userInfo}
                  onChange={(e) => setNpcForm({ ...npcForm, userInfo: e.target.value })}
                />
              </details>
              <details>
                <summary>行为准则</summary>
                <label className="field">行为准则</label>
                <TextArea
                  autoSize={{ minRows: 3, maxRows: 6 }}
                  disabled={busy}
                  value={npcForm.goldenSp}
                  onChange={(e) => setNpcForm({ ...npcForm, goldenSp: e.target.value })}
                />
                <label className="field">补充说明</label>
                <ListEditor
                  items={npcForm.notes}
                  onChange={(items) => setNpcForm({ ...npcForm, notes: items })}
                  placeholder="每行一条补充设定"
                  rows={4}
                  disabled={busy}
                />
              </details>
              <details>
                <summary>扩展字段</summary>
                <KeyValueEditor
                  items={npcExtras}
                  onChange={setNpcExtras}
                  emptyHint="没有扩展字段"
                  disabled={busy}
                />
              </details>
            </div>
          </section>
        </div>

        <div className="column center">
          <section className="panel chat">
            <div className="panel-header">
              <div>
                <h3>对话窗口</h3>
                <div className="panel-sub">
                  {world?.name ? `世界: ${world.name}` : '未绑定世界'} ·{' '}
                  {chatId ? `会话 ${chatId.slice(0, 6)}` : '未加载会话'}
                </div>
              </div>
              <div className="panel-actions">
                <Button type="primary" onClick={createChat} disabled={!world || busy}>
                  新建对话
                </Button>
                <Button className="ghost" onClick={() => openModal('chat')} disabled={busy}>
                  加载对话
                </Button>
                <Button className="ghost" onClick={() => autoRound(1)} disabled={busy || !chatId}>
                  自动对话（1轮）
                </Button>
                <Button className="ghost" onClick={() => autoRound(10)} disabled={busy || !chatId}>
                  自动对话（10轮）
                </Button>
                <Button className="play" onClick={autoUntil} disabled={busy || !chatId} title="自动对话">
                  ▶ 直到不想继续
                </Button>
              </div>
            </div>
            <div className="scene-strip">
              <span className="scene-label">场景</span>
              <span className="scene-text">{worldForm.sceneSetting || '未设置场景'}</span>
            </div>
            <div className="scene-window">
              <div className="scene-grid">
                <span className="scene-icon npc">N</span>
                <span className="scene-icon user">U</span>
              </div>
              <div className="scene-bubble">地图同步中 · 事件线索可视化</div>
            </div>
            <div className="panel-body scroll">
              {messageList.length === 0 ? (
                <div className="empty">暂无对话，先创建或加载一个对话。</div>
              ) : (
                <div className="messages">
                  {messageList.map((m) => (
                    <div key={m.id} className={`message-row ${m.role}`}>
                      <div className="avatar">{roleLabel(m.role).slice(0, 1)}</div>
                      <div className="message-card">
                        <div className="message-meta">
                          <span className="name">{roleLabel(m.role)}</span>
                          {m.created_at ? <span className="time">{formatTime(m.created_at)}</span> : null}
                          {m.emotion_label ? <Tag className="pill">{m.emotion_label}</Tag> : null}
                          {m.topic_tag ? <Tag className="pill">{m.topic_tag}</Tag> : null}
                          {m.desire_score ? <Tag className="pill">欲望 {m.desire_score}</Tag> : null}
                        </div>
                        <div className="message-text">{m.content}</div>
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </section>

          <section className="panel composer">
            <div className="panel-header">
              <div>
                <h3>发言输入</h3>
                <div className="panel-sub">用户消息 + 欲望评分</div>
              </div>
              <div className="panel-actions">
                <Button type="primary" onClick={sendMessage} disabled={busy || !chatId || !inputText.trim()}>
                  发送
                </Button>
              </div>
            </div>
            <div className="panel-body">
              {composerHint ? <div className="hint">{composerHint}</div> : null}
              <div className="composer-grid">
                <TextArea
                  placeholder="输入用户消息..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={busy}
                  ref={composerRef}
                  autoSize={{ minRows: 4, maxRows: 8 }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !e.altKey && !e.ctrlKey) {
                      e.preventDefault()
                      if (!busy && chatId && inputText.trim()) {
                        sendMessage()
                      }
                    }
                  }}
                />
                <div className="composer-side">
                  <label>继续欲望</label>
                  <div className="range-row">
                    <Slider
                      min={1}
                      max={10}
                      value={inputDesire}
                      onChange={(value) => setInputDesire(value as number)}
                      disabled={busy}
                    />
                    <span className="range-value">{inputDesire}</span>
                  </div>
                  <p className="muted">1 表示很想停止，10 表示非常想继续。</p>
                </div>
              </div>
            </div>
          </section>
        </div>

        <div className="column right">
          <section className="panel">
            <div className="panel-header">
              <div>
                <h3>用户设定</h3>
                <div className="panel-sub">用户视角信息与目标</div>
              </div>
              <div className="panel-actions">
                <Button type="primary" onClick={() => saveActor('user')} disabled={busy}>
                  保存设定
                </Button>
              </div>
            </div>
            <div className="panel-body scroll">
              <details open>
                <summary>身份信息</summary>
                <label className="field">称呼</label>
                <Input
                  value={userForm.displayName}
                  disabled={busy}
                  onChange={(e) => setUserForm({ ...userForm, displayName: e.target.value })}
                  placeholder="用户在世界中的称呼"
                />
                <label className="field">背景</label>
                <TextArea
                  autoSize={{ minRows: 3, maxRows: 5 }}
                  value={userForm.background}
                  disabled={busy}
                  onChange={(e) => setUserForm({ ...userForm, background: e.target.value })}
                />
              </details>
              <details>
                <summary>人格与目标</summary>
                <label className="field">人格与偏好</label>
                <TextArea
                  autoSize={{ minRows: 3, maxRows: 5 }}
                  value={userForm.persona}
                  disabled={busy}
                  onChange={(e) => setUserForm({ ...userForm, persona: e.target.value })}
                />
                <label className="field">目标</label>
                <TextArea
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  value={userForm.goals}
                  disabled={busy}
                  onChange={(e) => setUserForm({ ...userForm, goals: e.target.value })}
                />
                <label className="field">边界 / 约束</label>
                <TextArea
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  value={userForm.constraints}
                  disabled={busy}
                  onChange={(e) => setUserForm({ ...userForm, constraints: e.target.value })}
                />
              </details>
              <details>
                <summary>补充说明</summary>
                <ListEditor
                  items={userForm.notes}
                  onChange={(items) => setUserForm({ ...userForm, notes: items })}
                  placeholder="每行一条补充设定"
                  rows={4}
                  disabled={busy}
                />
              </details>
              <details>
                <summary>扩展字段</summary>
                <KeyValueEditor
                  items={userExtras}
                  onChange={setUserExtras}
                  emptyHint="没有扩展字段"
                  disabled={busy}
                />
              </details>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <h3>数据看板</h3>
                <div className="panel-sub">统计与多样性指标</div>
              </div>
              <div className="panel-actions">
                <Button className="ghost" onClick={() => refreshStats(chatId)} disabled={!chatId || busy}>
                  刷新
                </Button>
              </div>
            </div>
            <div className="panel-body scroll">
              {chatId ? (
                <div className="stats">
                  <Tabs items={statsTabs} />
                </div>
              ) : (
                <div className="empty">暂无数据</div>
              )}
            </div>
          </section>
        </div>
      </div>

      <Modal
        open={modal !== null}
        title={modal === 'world' ? '加载 World' : '加载对话'}
        onCancel={() => setModal(null)}
        onOk={confirmModal}
        okText="加载"
        cancelText="取消"
        okButtonProps={{ disabled: !modalValue.trim() || busy }}
      >
        <p className="muted">
          {modal === 'world'
            ? '输入已有的 World ID，左侧内容会自动填充。'
            : '输入已有的 Chat ID，系统会同步加载对应 World。'}
        </p>
        <Input
          placeholder={modal === 'world' ? 'World ID' : 'Chat ID'}
          value={modalValue}
          onChange={(e) => setModalValue(e.target.value)}
        />
      </Modal>
      <Modal
        open={!!notice}
        title={notice?.title}
        onCancel={() => setNotice(null)}
        footer={
          <Button type="primary" onClick={() => setNotice(null)}>
            知道了
          </Button>
        }
      >
        <p className="muted">{notice?.body}</p>
      </Modal>
    </div>
  )
}
