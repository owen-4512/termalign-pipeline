# 用 ChatGPT 网页版批量提问并导出 TXT

你遇到的“页面一直转圈 / Cloudflare 验证不过去”问题，通常和浏览器内核有关。这个版本做了两点优化：

- 默认 `--browser auto`：优先用本机 Chrome / Edge 渠道（比 Playwright 默认 Chromium 更不容易卡验证）
- 增加“启动就绪检查”：如果没出现输入框，会反复提示你先手动完成验证/登录

---

## 1) 安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

> 如果你用 `--browser chrome` 或 `--browser msedge`，请确保本机已安装对应浏览器。

## 2) 准备问题文件

创建 `questions.txt`（每行一个问题）：

```txt
什么是向量数据库？
如何用通俗的话解释 RAG？
帮我列一个一周健身计划。
```

## 3) 运行（推荐）

```bash
python batch_chatgpt_export.py --input questions.txt --output replies.txt --browser auto
```

运行后流程：
1. 浏览器打开 ChatGPT
2. 你先手动通过 Cloudflare + 登录
3. 回终端按回车
4. 程序检测到输入框后，开始逐条提问并抓取回复

## 常用参数

- `--browser auto|chrome|msedge|chromium`：浏览器选择（推荐 `auto`）
- `--wait-seconds 180`：每题最多等待 180 秒
- `--start-timeout 300`：启动阶段最多等待 300 秒
- `--headless`：无头模式（不建议首次使用）
- `--profile-dir .playwright-profile`：登录态目录
- `--url https://chatgpt.com/`：网页地址

## 如果还卡在验证页

你可以尝试：
- 明确指定 `--browser chrome` 或 `--browser msedge`
- 关闭 VPN/代理后重试
- 在打开的浏览器里手动刷新一次页面再按回车
- 保持非无头模式（不要加 `--headless`）

## 输出格式

`replies.txt` 会按下面格式保存：

```txt
===== 问题 1 =====
...

----- 回复 1 -----
...
```
