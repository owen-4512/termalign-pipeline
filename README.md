# TermAlign Pipeline - 外语学习 Chatbot（桌面端 + 管理后台）

本仓库基于你要求的架构实现了一个可落地的 MVP：

- 学生端桌面应用：Electron + React + TypeScript + Tailwind
- 后端服务：FastAPI（PostgreSQL + Redis）
- 管理后台：React + Ant Design
- AI 接入：OpenAI Responses API（服务端代理）

---

## 目录结构

- `apps/api`：FastAPI 后端（鉴权、聊天、聊天记录管理、统计）
- `apps/student-desktop`：学生端桌面应用（Electron）
- `apps/admin-web`：管理员 Web 后台
- `docker-compose.yml`：PostgreSQL + Redis
- `.env.example`：环境变量样例

---

## 环境要求

- Python 3.10+
- Node.js 20+
- npm 10+
- Docker + Docker Compose

---

## 1）快速启动（开发模式）

### Step A. 启动数据库与缓存

```bash
docker compose up -d
```

### Step B. 启动后端 API（端口 8000）

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../../.env.example .env
uvicorn app.main:app --reload --port 8000
```

启动成功后可访问健康检查：

```bash
curl http://localhost:8000/health
```

### Step C. 启动管理员后台（端口 5174）

```bash
cd apps/admin-web
npm install
npm run dev
```

浏览器打开：`http://localhost:5174`

### Step D. 启动学生桌面端（Vite 5173 + Electron）

```bash
cd apps/student-desktop
npm install
npm run dev
```

执行后会自动打开 Electron 桌面窗口。

---

## 2）默认账号

- 管理员：`admin@example.com` / `admin123`
- 学生：`student@example.com` / `student123`

> 后端启动时会自动写入以上演示账号。

---

## 3）如何使用这个 App（学生端）

1. 启动后端 + 学生端后，Electron 窗口会出现登录页。
2. 输入学生账号（默认 `student@example.com` / `student123`）登录。
3. 点击左侧「+ 新建会话」。
4. 在输入框中输入外语句子或问题（例如英文写作、语法、翻译）。
5. 点击「发送」，系统会：
   - 先保存学生消息到数据库；
   - 再通过后端调用 OpenAI Responses API；
   - 最后把助教回复保存并显示在对话区。
6. 左侧会话列表可切换历史会话继续学习。

### 学生端建议体验方式

- 口语表达练习：输入中文，让助教给更自然英文说法
- 写作纠错：输入英文段落，让助教标注语法并改写
- 单词记忆：让助教按 CEFR 难度给例句

---

## 4）如何使用这个 App（管理员后台）

1. 启动后端 + 管理后台后，访问 `http://localhost:5174`。
2. 点击「使用默认管理员账号登录」。
3. 登录后可看到统计卡片：
   - 消息总数
   - 会话总数
   - 活跃用户
4. 在筛选区输入：
   - 学生邮箱（模糊匹配）
   - 关键词（匹配消息内容/会话标题）
5. 点击「查询」查看聊天记录表。
6. 点击「导出 CSV」导出聊天记录。

---

## 5）OpenAI API 配置说明

编辑 `apps/api/.env`：

```env
OPENAI_API_KEY=你的key
OPENAI_MODEL=gpt-4.1-mini
```

- 如果不填 `OPENAI_API_KEY`，系统会自动返回 Mock 助教回复（便于开发联调）。
- 生产环境务必使用服务端密钥管理，不要把 Key 放客户端。

---

## 6）已实现能力（MVP）

- 学生登录与会话管理
- 学生发送消息并调用 OpenAI Responses API（无 KEY 时降级为 mock 回复）
- 聊天记录落库（用户、会话、消息）
- 管理员按学生/关键词筛选聊天记录
- 管理员导出 CSV
- 基础统计（消息量、会话量、活跃用户）

---

## 7）常见问题（FAQ）

### Q1：管理员点击导出 CSV 提示未授权？
请先在页面完成管理员登录，确保 API 请求头包含 `Bearer token`。

### Q2：学生端一直没有 AI 回复？
先检查：
1. `apps/api/.env` 是否设置 `OPENAI_API_KEY`
2. 后端日志是否有 OpenAI 报错
3. 是否可访问 `http://localhost:8000/health`

### Q3：前端跨域报错（CORS）？
检查 `.env` 中：

```env
CORS_ORIGINS=http://localhost:5173,http://localhost:5174
```

---

## 8）下一步建议（生产化）

- 接入学校统一身份认证（SSO）
- 增加敏感内容审核与告警策略
- 增加对话总结、词汇本、错题本
- 增加自动化测试与 CI/CD
- 增加 Electron 自动更新与安装包发布流程
