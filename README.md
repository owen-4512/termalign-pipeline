# 用 ChatGPT 网页版批量提问并导出 TXT

你提的需求是：**用 ChatGPT 网页 App**，把多个问题逐个贴进去，每个问题拿一个回复，最后导出 `txt`。

这个脚本就是按这个流程做的：
1. 打开 ChatGPT 网页（可复用登录态）
2. 你手动登录一次
3. 程序逐条发送问题
4. 自动抓取每条问题对应的最新回复
5. 导出到 txt

---

## 1) 安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 2) 准备问题文件

创建 `questions.txt`（每行一个问题，空行会自动忽略）：

```txt
什么是向量数据库？
如何用通俗的话解释 RAG？
帮我列一个一周健身计划。
```

## 3) 运行脚本

```bash
python batch_chatgpt_export.py --input questions.txt --output replies.txt
```

首次运行会打开浏览器：
- 先在 ChatGPT 网页中登录
- 进入可以正常聊天的页面
- 回终端按一次回车，脚本开始逐条提问

## 常用参数

- `--wait-seconds 180`：每题最多等待 180 秒
- `--headless`：无头模式（一般不建议首次使用）
- `--profile-dir .playwright-profile`：登录态目录
- `--url https://chatgpt.com/`：网页地址

## 输出格式

`replies.txt` 会按下面格式保存：

```txt
===== 问题 1 =====
...

----- 回复 1 -----
...
```

## 注意事项

- ChatGPT 网页结构偶尔会改版，若按钮/输入框定位失败，可更新选择器。
- 如果账号有验证码/风控，建议手动完成后再按回车开始批量发送。
- 本脚本用于个人效率工具，请遵守相关平台条款。
