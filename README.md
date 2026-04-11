# 用 ChatGPT 网页版批量提问并导出 TXT

你现在遇到的是：登录后跳到 `https://chatgpt.com/api/auth/error`。
这通常是登录态冲突或自动化浏览器会话异常导致。

## 推荐修复命令（先用这个）

```bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
```

为什么这样配：
1. `--reset-profile`：删除旧会话，彻底重登
2. `--browser chrome`：使用本机 Chrome（比默认自动化内核更稳定）
3. 从主页进入而不是直接跳 `/auth/login`，减少回调错误概率

---

## 这版脚本新增的保护

- 启动时若检测到 `/api/auth/error`，会自动尝试：清理 cookie -> 访问 logout -> 回到首页
- 启动参数中减少自动化特征（降低被异常风控拦截概率）
- 保留人工确认步骤：你可以先手工完成验证/登录，再回终端继续

## 安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 常用参数

- `--browser auto|chrome|msedge|chromium`：浏览器选择（推荐 `chrome`）
- `--reset-profile`：清空 `--profile-dir`，强制重新登录
- `--profile-dir .playwright-profile`：登录态目录（多账号可用多个目录）
- `--url https://chatgpt.com/`：启动地址（推荐主页）
- `--start-timeout 300`：启动阶段等待输入框出现的最大秒数
- `--wait-seconds 180`：每题等待回复秒数

## 如果还报 auth/error

1. 先关掉脚本
2. 改用全新 profile 目录再试：

```bash
python batch_chatgpt_export.py --input questions.txt --output replies.txt --browser chrome --profile-dir .profile-fresh
```

3. 在打开的浏览器里手动刷新，再登录
4. 如仍失败，先在你平时手动使用的浏览器中确认 ChatGPT 登录正常，再回到脚本
