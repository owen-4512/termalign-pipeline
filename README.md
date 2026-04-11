# 用 ChatGPT 网页版批量提问并导出 TXT

你现在这个情况（已登录，但终端仍提示“页面暂未就绪”）通常是因为：
- ChatGPT 新版输入框不是 `textarea`，而是 `contenteditable div`
- 页面虽然在首页，但输入框还没真正激活（需要先点左侧 `New chat`）

这个版本已经针对以上两点修复。

## 推荐运行方式

```bash
python batch_chatgpt_export.py \
  --input questions.txt \
  --output replies.txt \
  --browser chrome \
  --reset-profile \
  --url https://chatgpt.com/
```

## questions.txt 支持两种格式

### 格式 A：每行一个问题（简单模式）

```txt
问题1
问题2
问题3
```

### 格式 B：复杂多行问题，用“至少 3 个换行”分隔（推荐复杂场景）

```txt
请帮我写一份短视频运营计划。
要求：
1) 先给 30 天计划
2) 按周拆解


请把下面这段英文翻译成中文，并给术语表：
...


帮我比较 A 方案和 B 方案，给出利弊和推荐结论。
```

> 也就是说：当问题本身有多行内容时，在两个问题中间空至少两行（形成 3 个及以上换行）即可。

## 使用步骤（关键）

1. 浏览器打开后，先完成登录
2. 点击左侧 **New chat / Neuer Chat**，确保底部输入框出现
3. 回到终端按回车
4. 脚本开始批量发送并导出结果

---

## 这版修复点

- 增加新版 ChatGPT 输入框选择器（`contenteditable` / `ProseMirror` 等）
- 输入文本时支持 `fill` 失败自动回退到键盘输入
- `questions.txt` 新增复杂多行格式（3+ 换行分隔）
- 终端提示中明确要求先打开新聊天并确认输入框可见

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
