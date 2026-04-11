# 用 ChatGPT 网页版批量提问并导出 TXT

你现在这个情况（一直慢、可能选错了 Google 账号登录）最实用的方式是：**清空自动化浏览器的登录态后重登**。

## 快速重登（推荐命令）

```bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/auth/login
```

这条命令会做 3 件事：
1. `--reset-profile`：删除旧登录缓存（等于“退出当前账号”）
2. `--browser chrome`：用你本机 Chrome（更接近手工登录）
3. 直接打开登录页，重新选择账号登录

> 如果你用 Edge，把 `--browser chrome` 改成 `--browser msedge`。

---

## 安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 常用参数

- `--browser auto|chrome|msedge|chromium`：浏览器选择（推荐 `chrome` 或 `auto`）
- `--reset-profile`：清空 `--profile-dir`，强制重新登录
- `--profile-dir .playwright-profile`：登录态目录（可换新目录实现多账号）
- `--url https://chatgpt.com/auth/login`：直接打开登录页
- `--start-timeout 300`：启动阶段等待输入框出现的最大秒数
- `--wait-seconds 180`：每题等待回复秒数

## 如果你不想删目录，也可以“换一个新登录目录”

```bash
python batch_chatgpt_export.py --input questions.txt --output replies.txt --profile-dir .profile-new --browser chrome
```

这样会启动一个“全新浏览器身份”，你可以重新选 Google 账号。

## 输出格式

`replies.txt` 会按下面格式保存：

```txt
===== 问题 1 =====
...

----- 回复 1 -----
...
```
