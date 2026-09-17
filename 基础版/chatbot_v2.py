"""带记忆的聊天机器人 v2（DeepSeek 版）

相比 chatbot.py 新增：
    ④ 多会话：启动时输入会话名称，不同名称 = 不同对话，互不干扰
    ⑥ 历史管理：trim_messages 自动修剪超长历史，防止撑爆上下文窗口
    ⑦ 流式输出：逐字打印模型回复（打字机效果）

运行：
    python3 chatbot_v2.py
退出：
    输入 quit 或 退出
"""

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek

# 读取项目根目录下的 .env（用环境变量设置密钥也可以）
load_dotenv()

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("未检测到 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入密钥。")

# 1. 模型（推理题多可换成 model="deepseek-reasoner"）
model = ChatDeepSeek(model="deepseek-chat")

# 2. 提示词模板：人设 + 历史占位 + 用户输入
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的中文助手，回答简洁准确。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 3. 历史修剪器（⑥）：只保留最近 max_tokens 的部分
#    - token_counter：用自定义函数按字符数估算 token 数
#      （不能用 token_counter=model：langchain-deepseek 对 deepseek-chat
#        没有实现 get_num_tokens_from_messages，会直接报错）
#    - include_system=True：系统人设始终保留
#    - start_on="human"：以用户消息开头，避免从半句开始
def count_tokens(messages) -> int:
    """粗略 token 计数：按字符数估算（中文约 1 字 ≈ 1 token）。"""
    return sum(len(getattr(m, "content", "") or "") for m in messages)


trimmer = trim_messages(
    max_tokens=500,        # 保留上限（按字符估算），可按需调大/调小
    strategy="last",       # 保留最后的部分
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# 4. 组装链：先修剪历史，再填进提示词，最后交给模型
#    RunnablePassthrough.assign 把 history 替换为修剪后的版本
chain = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    | prompt
    | model
)

# 5. 历史存储：session_id -> 该会话的消息记录（内存中）
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 6. 包上"自动记忆"外壳
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def sanitize(text: str) -> str:
    """去掉无法编码为 UTF-8 的非法字符（孤立代理对等），正常 emoji 保留。"""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


# 7. 交互循环（④：会话名称可切换）
print("=" * 50)
print("🤖 聊天机器人 v2 已启动")
print("=" * 50)
session_id = input("会话名称（回车使用默认 demo-session）: ").strip() or "demo-session"
print(f"当前会话：{session_id}（换一个名称 = 开启新对话，互不干扰）")
print("输入 quit 或 退出 结束\n")

config = {"configurable": {"session_id": session_id}}

while True:
    try:
        user_input = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 再见！")
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "退出"):
        print("👋 再见！")
        break

    user_input = sanitize(user_input)

    try:
        # ⑦ 流式输出：逐字打印，打字机效果
        print("🤖 助手: ", end="", flush=True)
        for r in chatbot.stream({"input": user_input}, config=config):
            print(r.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"⚠️ 调用出错（已跳过，可继续对话）：{e}")
        continue
