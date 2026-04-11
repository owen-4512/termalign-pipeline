# ChatGPT Web Batch Q&A Tool

[🇨🇳 中文](#-中文说明) | [🇬🇧 English](#-english)

---

# 🇨🇳 中文说明

## 📌 项目简介

通过浏览器自动化，实现 **批量提交问题并导出回答结果**。  
适用于数据采集、批量测试、问答整理等场景。

---

## 🚀 推荐运行方式

```bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
📄 questions.txt 格式说明
✅ 格式 A：每行一个问题（简单模式）
问题1
问题2
问题3
✅ 格式 B：多行复杂问题（推荐）
请帮我写一份短视频运营计划。
要求：
1) 先给 30 天计划
2) 按周拆解


请把下面这段英文翻译成中文，并给术语表：
...


帮我比较 A 方案和 B 方案，给出利弊和推荐结论。

## 📌 分隔规则：

使用「至少两个空行」（即 ≥3 个换行）分隔问题
单个问题内部可自由换行
🧭 使用步骤（关键）
启动脚本后浏览器自动打开
手动登录 ChatGPT
点击左侧 New chat / Neuer Chat
确认输入框已出现
回到终端按回车
脚本开始批量执行并导出结果
## 📦 安装依赖
pip install -r requirements.txt
python -m playwright install chromium
## ⚙️ 常用参数
参数	说明
--browser	浏览器类型：auto / chrome / msedge / chromium
--reset-profile	清空登录态并重新登录
--profile-dir	登录目录（支持多账号）
--url	启动地址
--start-timeout	启动等待时间（秒）
--wait-seconds	每题等待回复时间
## 💡 使用建议
首次运行建议使用 --reset-profile
多账号建议使用不同 profile-dir
长回答建议提高 --wait-seconds
🇬🇧 English
## 📌 Overview

This tool automates the browser to batch submit questions to ChatGPT and export responses.
Useful for data collection, testing, and Q&A workflows.

## 🚀 Recommended Usage
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
📄 questions.txt Format
✅ Format A: One question per line (Simple)
Question 1
Question 2
Question 3
✅ Format B: Multi-line questions (Recommended)
Write a short video content plan.
Requirements:
1) 30-day plan
2) Weekly breakdown


Translate the following English text into Chinese and provide a glossary:
...


Compare Plan A and Plan B with pros/cons and recommendation.

## 📌 Separator Rule:

Use at least two empty lines (≥3 line breaks) between questions
Multi-line content within a question is fully supported
## 🧭 Steps
Run the script → browser opens automatically
Log in to ChatGPT manually
Click "New chat"
Ensure the input box is visible
Press Enter in terminal
Script starts processing and exporting results
## 📦 Installation
pip install -r requirements.txt
python -m playwright install chromium
## ⚙️ Common Arguments
Argument	Description
--browser	auto / chrome / msedge / chromium
--reset-profile	Clear session and force re-login
--profile-dir	Profile directory (multi-account support)
--url	Start URL
--start-timeout	Max wait time for input box
--wait-seconds	Wait time per question
## 💡 Tips
Use --reset-profile on first run
Use different profile-dir for multiple accounts
Increase --wait-seconds for long responses
