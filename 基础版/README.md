# 对话机器人（DeepSeek 版）

基于 [LangChain](https://www.langchain.com/) + [DeepSeek](https://www.deepseek.com/) 的带记忆聊天机器人。

## 版本

| 文件 | 说明 |
| --- | --- |
| `chatbot.py` | **基础版**：自动记忆多轮对话 + 编码容错（emoji/特殊字符不崩溃） |
| `chatbot_v2.py` | **增强版**：在基础版上增加多会话隔离、历史修剪（trim_messages）、流式输出 |

## 快速开始

```bash
# 1. 创建虚拟环境并激活
python -m venv venv
# Windows: venv\Scripts\activate    Linux/WSL: source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env      # Windows: copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-你的密钥

# 4. 运行
python chatbot.py         # 基础版
python chatbot_v2.py      # 增强版（推荐）
```

## 使用

- 直接输入文字即可多轮对话，机器人自动记住上下文
- 输入 `quit` 或 `退出` 结束
- v2 增强版：启动时可输入"会话名称"（不同名称 = 不同对话）；回复为流式输出；历史自动修剪到 `max_tokens` 上限（默认 500 字符估算值，可调）

## 说明

- 密钥只放在 `.env`（已被 `.gitignore` 忽略），不要写进代码或提交到版本库
- 记忆默认存内存，重启程序即清空；需要持久化可改用 `SQLChatMessageHistory`
- 详细的环境配置与故障排查见 `环境配置与使用说明.txt`
