# ChatGPT Web Batch Q&A Tool

[🇨🇳 中文](#中文说明) \| [🇬🇧 English](#english)

------------------------------------------------------------------------

## 中文说明

### 📌 项目简介

通过浏览器自动化，实现 **批量提交问题并导出回答结果**。\
适用于数据采集、批量测试、问答整理等场景。

------------------------------------------------------------------------

### 🚀 推荐运行方式

``` bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
```

------------------------------------------------------------------------

### 📄 questions.txt 格式说明

#### ✅ 格式 A：每行一个问题（简单模式）

``` txt
问题1
问题2
问题3
```

#### ✅ 格式 B：多行复杂问题（推荐）

``` txt
请帮我写一份短视频运营计划。
要求：
1) 先给 30 天计划
2) 按周拆解


请把下面这段英文翻译成中文，并给术语表：
...


帮我比较 A 方案和 B 方案，给出利弊和推荐结论。
```

#### 📌 分隔规则

-   使用至少两个空行（≥3 个换行）分隔问题\
-   单个问题内部可自由换行

------------------------------------------------------------------------

### 🧭 使用步骤

1.  启动脚本后浏览器自动打开\
2.  手动登录 ChatGPT\
3.  点击 New chat / Neuer Chat\
4.  确认输入框已出现\
5.  回到终端按回车\
6.  脚本开始执行

------------------------------------------------------------------------

### 📦 安装依赖

``` bash
pip install -r requirements.txt
python -m playwright install chromium
```

------------------------------------------------------------------------

### ⚙️ 常用参数

  参数              说明
  ----------------- --------------
  --browser         浏览器类型
  --reset-profile   清空登录态
  --profile-dir     登录目录
  --url             启动地址
  --start-timeout   启动等待时间
  --wait-seconds    每题等待时间

------------------------------------------------------------------------

### 使用建议
- 首次运行建议使用 --reset-profile\
- 多账号建议使用不同 --profile-dir\
- 长回答建议提高 --wait-seconds

------------------------------------------------------------------------

## English

### 📌 Overview

Automate browser to batch submit questions and export answers.

------------------------------------------------------------------------

### 🚀 Usage

``` bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
```

------------------------------------------------------------------------

### 📄 Format

-   One question per line OR\
-   Multi-line questions separated by blank lines

------------------------------------------------------------------------

### 📦 Installation

``` bash
pip install -r requirements.txt
python -m playwright install chromium
```

------------------------------------------------------------------------

### 💡 Tips
- Use --reset-profile on first run\
- Use different --profile-dir values for multiple accounts\
- Increase --wait-seconds for long responses

