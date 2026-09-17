# 对话机器人交付包

基于 **LangChain + DeepSeek V4.1 Flash** 的对话机器人，两个版本：

| 目录 | 说明 | 运行环境 |
| --- | --- | --- |
| `基础版/` | 纯命令行聊天机器人：记忆、多会话、历史修剪、流式输出 | 任意平台（含 WSL） |
| `微信版/` | 接入微信的完整版：记忆、画像、情绪、RAG、识图、语音、表情包 | Windows（需微信桌面端） |

## 快速开始

### 基础版（命令行）
```bash
cd 基础版
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # 填入 DEEPSEEK_API_KEY
python chatbot_v2.py            # 增强版（推荐）
python chatbot.py               # 基础版
```
详细说明见 `基础版/环境配置与使用说明.txt`。

### 微信版（接入微信）
```powershell
cd 微信版
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-wechat.txt
copy .env.example .env          # 填入 DEEPSEEK_API_KEY
venv\Scripts\python.exe wechat_bot.py
```
详细说明见 `微信版/使用说明.txt`。

## 微信版功能一览

- **对话记忆**：SQLite 持久化，重启后依然记得
- **用户画像**：每 6 轮自动抽取长期事实（在群里按发言人分别记录）
- **情绪状态**：好感度 / 心情 / 精力，持续演化并影响语气
  - 查看方式：`venv\Scripts\python.exe 查看状态.py --facts`
- **知识库 RAG**：把资料写进 `知识库.txt` 即可被引用
- **图片理解**：DeepSeek V4.1 Flash 原生多模态，能看懂图片和表情包
  - 自检：`venv\Scripts\python.exe test_vision.py`
- **语音回复**：edge-tts 合成（免费、无需 GPU）；可选 GPT-SoVITS 音色克隆
- **表情包**：按当前心情自动挑图发送
- **拟人化行为**：分条发送、打字延迟、去掉括号动作、群里被 @ 才回应

## 注意事项

- **不要外发 `.env` 与 `memory.db`**：前者是你的密钥，后者含聊天记录与用户画像
- 首次运行会在 `微信版/` 下自动生成 `memory.db`、`voice_cache/`、`media_cache/`
- 个人微信自动化存在**封号风险**，建议用小号测试、控制频率
- 微信版要求 **Python 3.12**（3.14 无法安装 wechatauto-replica 的 winsdk 依赖）
