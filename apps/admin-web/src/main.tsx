import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Button, Card, Input, Space, Statistic, Table, Typography, message } from 'antd'
import 'antd/dist/reset.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [token, setToken] = useState('')
  const [stats, setStats] = useState({ message_count: 0, conversation_count: 0, active_users: 0 })
  const [rows, setRows] = useState<any[]>([])
  const [email, setEmail] = useState('')
  const [keyword, setKeyword] = useState('')

  const authHeader = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token])

  const login = async () => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'admin@example.com', password: 'admin123' })
    })
    const data = await res.json()
    if (!data.access_token) {
      message.error('登录失败')
      return
    }
    setToken(data.access_token)
    message.success('登录成功')
  }

  const loadStats = async () => {
    const res = await fetch(`${API_BASE}/admin/stats`, { headers: authHeader })
    const data = await res.json()
    setStats(data)
  }

  const search = async () => {
    const params = new URLSearchParams()
    if (email) params.set('user_email', email)
    if (keyword) params.set('keyword', keyword)
    const res = await fetch(`${API_BASE}/admin/messages?${params.toString()}`, { headers: authHeader })
    const data = await res.json()
    setRows(data)
  }

  const exportCsv = () => {
    window.open(`${API_BASE}/admin/messages/export`, '_blank')
  }

  useEffect(() => {
    if (!token) return
    loadStats()
    search()
  }, [token])

  if (!token) {
    return (
      <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', background: '#f5f7fb' }}>
        <Card title="管理员后台" style={{ width: 380 }}>
          <Button type="primary" block onClick={login}>使用默认管理员账号登录</Button>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, background: '#f5f7fb', minHeight: '100vh' }}>
      <Typography.Title level={3}>聊天记录管理后台</Typography.Title>

      <Space size={16} wrap>
        <Card><Statistic title="消息总数" value={stats.message_count} /></Card>
        <Card><Statistic title="会话总数" value={stats.conversation_count} /></Card>
        <Card><Statistic title="活跃用户" value={stats.active_users} /></Card>
      </Space>

      <Card style={{ marginTop: 16 }}>
        <Space wrap>
          <Input placeholder="学生邮箱" value={email} onChange={(e) => setEmail(e.target.value)} style={{ width: 220 }} />
          <Input placeholder="关键词" value={keyword} onChange={(e) => setKeyword(e.target.value)} style={{ width: 220 }} />
          <Button type="primary" onClick={search}>查询</Button>
          <Button onClick={exportCsv}>导出 CSV</Button>
        </Space>
      </Card>

      <Card style={{ marginTop: 16 }}>
        <Table
          rowKey="id"
          dataSource={rows}
          pagination={{ pageSize: 10 }}
          columns={[
            { title: 'ID', dataIndex: 'id', width: 80 },
            { title: '学生', dataIndex: 'user_email', width: 180 },
            { title: '会话', dataIndex: 'conversation_title', width: 200 },
            { title: '角色', dataIndex: 'role', width: 100 },
            { title: '内容', dataIndex: 'content' },
            { title: '时间', dataIndex: 'created_at', width: 200 }
          ]}
        />
      </Card>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
