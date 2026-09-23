# -*- coding: utf-8 -*-
"""验证工具集与 deepseek-flash 的 Function Calling 能力"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

import 工具

load_dotenv()   # 读取同目录 .env 里的 DEEPSEEK_API_KEY

print("=" * 60)
print("1) 工具注册表")
print("=" * 60)
for t in 工具.TOOLS:
    print("  · %-18s %s" % (t.name, (t.description or "").splitlines()[0][:45]))

print()
print("=" * 60)
print("2) 本地执行测试（不经模型）")
print("=" * 60)
print("  时间 ->", 工具.get_current_time.invoke({}))
print("  计算 ->", 工具.calculate.invoke({"expression": "(1+2)**3*5"}))
print("  计算(恶意) ->", 工具.calculate.invoke({"expression": "__import__('os').system('dir')"}))

print()
print("=" * 60)
print("3) 模型自主调用测试（关键：deepseek-flash 是否支持 Function Calling）")
print("=" * 60)
model = ChatDeepSeek(model="deepseek-flash")
model_with_tools = model.bind_tools(工具.TOOLS)

cases = [
    "现在几点了？",
    "帮我算一下 128 * 37 + 456 等于多少",
    "北京今天天气怎么样？",
    "你们产品专业版多少钱？",
    "你好呀，今天心情不错",
]

for q in cases:
    try:
        r = model_with_tools.invoke([HumanMessage(content=q)])
        calls = getattr(r, "tool_calls", None) or []
        if calls:
            desc = ", ".join("%s(%s)" % (c["name"], c["args"]) for c in calls)
            print("  %-28s -> 调用工具: %s" % (q, desc))
        else:
            content = (r.content or "").replace("\n", " ")[:36]
            print("  %-28s -> 直接回答: %s" % (q, content))
    except Exception as e:
        print("  %-28s -> 失败: %s" % (q, e))
