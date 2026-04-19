import React, { useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

type Message = { id: number; role: string; content: string; created_at: string }
type Conversation = { id: number; title: string; created_at: string }

const API_BASE = 'http://localhost:8000'

function App() {
  const [email, setEmail] = useState('student@example.com')
  const [password, setPassword] = useState('student123')
  const [token, setToken] = useState<string>('')
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')

  const authHeader = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token])

  const login = async () => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    })
    const data = await res.json()
    setToken(data.access_token)
    await loadConversations(data.access_token)
  }

  const loadConversations = async (t = token) => {
    const res = await fetch(`${API_BASE}/chat/conversations`, { headers: { Authorization: `Bearer ${t}` } })
    const data = await res.json()
    setConversations(data)
  }

  const createConversation = async () => {
    const res = await fetch(`${API_BASE}/chat/conversations`, {
      method: 'POST',
      headers: { ...authHeader, 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: `外语练习 ${new Date().toLocaleString()}` })
    })
    const conv = await res.json()
    setCurrentConversation(conv.id)
    await loadConversations()
    setMessages([])
  }

  const openConversation = async (id: number) => {
    setCurrentConversation(id)
    const res = await fetch(`${API_BASE}/chat/conversations/${id}/messages`, { headers: authHeader })
    const data = await res.json()
    setMessages(data)
  }

  const send = async () => {
    if (!currentConversation || !input.trim()) return
    const res = await fetch(`${API_BASE}/chat/conversations/${currentConversation}/messages`, {
      method: 'POST',
      headers: { ...authHeader, 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: input, language: 'en' })
    })
    const data = await res.json()
    setMessages((prev) => [...prev, data.user_message, data.assistant_message])
    setInput('')
  }

  if (!token) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="w-[420px] rounded-2xl bg-white p-8 shadow-lg">
          <h1 className="text-2xl font-semibold mb-6">学生学习助手登录</h1>
          <input className="input" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="邮箱" />
          <input className="input mt-3" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="密码" />
          <button className="btn mt-4 w-full" onClick={login}>登录</button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen bg-slate-50 text-slate-800 grid grid-cols-[280px_1fr]">
      <aside className="p-4 border-r border-slate-200 bg-white">
        <button className="btn w-full" onClick={createConversation}>+ 新建会话</button>
        <div className="mt-4 space-y-2 overflow-auto max-h-[88vh]">
          {conversations.map((c) => (
            <button key={c.id} className="w-full text-left rounded-lg px-3 py-2 hover:bg-slate-100" onClick={() => openConversation(c.id)}>
              <div className="font-medium line-clamp-1">{c.title}</div>
            </button>
          ))}
        </div>
      </aside>
      <main className="p-6 flex flex-col">
        <h2 className="text-xl font-semibold mb-4">外语学习对话</h2>
        <div className="flex-1 overflow-auto rounded-xl bg-white border border-slate-200 p-4 space-y-3">
          {messages.map((m) => (
            <div key={m.id} className={m.role === 'assistant' ? 'msg-ai' : 'msg-user'}>
              <div className="text-xs opacity-70">{m.role === 'assistant' ? '助教' : '我'}</div>
              <div className="whitespace-pre-wrap">{m.content}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex gap-2">
          <input value={input} onChange={(e) => setInput(e.target.value)} className="input flex-1" placeholder="输入你想练习的句子..." />
          <button className="btn" onClick={send}>发送</button>
        </div>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
