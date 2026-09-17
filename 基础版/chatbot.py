"""带记忆的聊天机器人（DeepSeek 版）

功能：
    - 自动保存对话历史：多轮对话不再"失忆"
    - 会话隔离：session_id 不同的人互不干扰
    - 人设指令：通过 system 消息设定机器人角色

运行：
    python3 chatbot.py
退出：
    输入 quit 或 退出
"""

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

chain = prompt | model

# 3. 历史存储：session_id -> 该会话的消息记录（内存中）
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 4. 包上"自动记忆"外壳
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def sanitize(text: str) -> str:
    """去掉无法编码为 UTF-8 的非法字符（如终端粘贴进来的孤立代理对），正常 emoji 会保留。"""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


# 5. 交互循环
session_id = "demo-session"  # 不同 id 相当于不同用户/不同对话
config = {"configurable": {"session_id": session_id}}

print("🤖 聊天机器人已启动（输入 quit 或 退出 结束）\n")

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

    # 清洗非法字符，避免编码错误导致崩溃
    user_input = sanitize(user_input)

    try:
        response = chatbot.invoke({"input": user_input}, config=config)
    except Exception as e:
        # 单次调用出错只提示，不退出程序
        print(f"⚠️ 调用出错（已跳过，可继续对话）：{e}")
        continue

    print(f"🤖 助手: {response.content}")
