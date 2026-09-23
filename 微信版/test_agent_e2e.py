# -*- coding: utf-8 -*-
"""端到端验证：Agent 工具调用 + 记忆 + 画像 + 状态 全链路"""

import sys

sys.path.insert(0, ".")

import wechat_bot as w

SESSION = "测试会话_agent"

cases = [
    "现在几点了？",
    "帮我算一下 128 乘以 37 再加 456 等于多少",
    "你们专业版多少钱？",
    "你好呀，今天心情不错",
]

print("=" * 60)
print("Agent 端到端测试（会话：%s）" % SESSION)
print("=" * 60)

for q in cases:
    ans = w.ask_bot(SESSION, q, SESSION)
    ans = (ans or "").replace("\n", " ")
    print("\n用户：%s" % q)
    print("机器人：%s" % ans[:80])
