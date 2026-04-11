#!/usr/bin/env python3
"""通过 ChatGPT 网页版批量提问并导出结果。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

from playwright.sync_api import Page, TimeoutError, sync_playwright


COMPOSER_SELECTORS = [
    'textarea[data-testid="prompt-textarea"]',
    'textarea#prompt-textarea',
    'textarea[placeholder*="Message"]',
    'textarea[placeholder*="发送"]',
]

SEND_BUTTON_SELECTORS = [
    'button[data-testid="send-button"]',
    'button[aria-label*="Send"]',
    'button[aria-label*="发送"]',
]

ASSISTANT_MSG_SELECTORS = [
    '[data-message-author-role="assistant"]',
    'article[data-testid="conversation-turn-assistant"]',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 ChatGPT 网页版批量提问并导出 TXT。")
    parser.add_argument("--input", required=True, help="问题文件：每行一个问题。")
    parser.add_argument("--output", default="chatgpt_web_replies.txt", help="输出文件路径。")
    parser.add_argument(
        "--url", default="https://chatgpt.com/", help="ChatGPT 网页地址（默认 https://chatgpt.com/）。"
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="每个问题等待回复的最大秒数（默认 120）。",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式运行（默认关闭，便于你手动登录）。",
    )
    parser.add_argument(
        "--profile-dir",
        default=".playwright-profile",
        help="浏览器用户数据目录，用于复用登录态。",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions:
        raise ValueError("输入文件为空，请至少保留一个非空问题。")
    return questions


def first_visible(page: Page, selectors: Iterable[str]):
    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() > 0 and loc.first.is_visible():
            return loc.first
    return None


def wait_for_composer(page: Page, timeout_ms: int = 120_000):
    end_time = time.time() + timeout_ms / 1000
    while time.time() < end_time:
        composer = first_visible(page, COMPOSER_SELECTORS)
        if composer:
            return composer
        time.sleep(0.3)
    raise TimeoutError("未找到输入框，请确认你已登录并进入可提问页面。")


def latest_assistant_text(page: Page) -> str:
    for sel in ASSISTANT_MSG_SELECTORS:
        items = page.locator(sel)
        if items.count() > 0:
            text = items.nth(items.count() - 1).inner_text().strip()
            if text:
                return text
    return ""


def send_question_and_wait(page: Page, question: str, wait_seconds: int) -> str:
    composer = wait_for_composer(page)
    before = latest_assistant_text(page)

    composer.click()
    composer.fill(question)

    send_btn = first_visible(page, SEND_BUTTON_SELECTORS)
    if send_btn:
        send_btn.click()
    else:
        composer.press("Enter")

    deadline = time.time() + wait_seconds
    last_text = ""
    stable_rounds = 0

    while time.time() < deadline:
        current = latest_assistant_text(page)
        if current and current != before:
            if current == last_text:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_text = current

            if stable_rounds >= 3:
                return current
        time.sleep(1)

    if last_text:
        return f"[可能未完整生成，已超时] {last_text}"
    return "[未获取到回复：等待超时]"


def export_results(path: Path, results: list[tuple[str, str]]) -> None:
    blocks = []
    for i, (q, a) in enumerate(results, 1):
        blocks.append(
            f"===== 问题 {i} =====\n{q}\n\n"
            f"----- 回复 {i} -----\n{a}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(Path(args.input))
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(args.url)

        print("请在打开的浏览器中完成登录并进入聊天页，然后回到终端按回车继续...", flush=True)
        input()

        results: list[tuple[str, str]] = []
        total = len(questions)
        for idx, q in enumerate(questions, 1):
            print(f"[{idx}/{total}] 发送问题中...", flush=True)
            try:
                ans = send_question_and_wait(page, q, args.wait_seconds)
            except Exception as exc:  # noqa: BLE001
                ans = f"[抓取失败] {type(exc).__name__}: {exc}"
            results.append((q, ans))

        context.close()

    output = Path(args.output)
    export_results(output, results)
    print(f"完成：已导出 {len(results)} 组问答到 {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
