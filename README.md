# 对话机器人交付包

基于 **LangChain + DeepSeek V4.1 Flash** 的对话机器人，两个版本：

| 目录 | 说明 | 运行环境 |
| --- | --- | --- |
| `基础版/` | 纯命令行聊天机器人：记忆、多会话、历史修剪、流式输出 | 任意平台（含 WSL） |
| `微信版/` | 接入微信的完整版：记忆、画像、情绪、RAG、识图、语音、表情包 | Windows（需微信桌面端） |

## ⚠️ 免责声明（请先阅读）

- 本项目**仅供学习与技术研究**。请遵守所在地法律法规，以及你所使用的第三方服务
  （DeepSeek、微信等）的服务条款。
- **微信版基于对个人微信桌面客户端的自动化操作**，这**违反《微信个人账号使用规范》**，
  存在**账号被限制、封禁**的风险。请务必使用小号测试、控制操作频率，
  **不要用于营销、群发或任何商业用途**。
- 使用本项目所产生的任何直接或间接后果（包括但不限于账号封禁、数据丢失、API 费用、
  法律纠纷），**由使用者自行承担**，作者不承担任何责任。
- 使用前请确认你已获得相关账号与数据的合法授权；`.env` 中的密钥与
  `memory.db` 中的聊天记录属于敏感数据，请自行妥善保管，不要外发或提交到版本库。

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

## 许可

本项目采用 [MIT License](LICENSE) 开源。
