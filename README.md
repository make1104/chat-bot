# 🤖 多模态长期记忆 Agent 系统

> 一个能在真实 IM 场景里长期运行的对话 Agent：**有记忆、有情绪、能看懂图和表情、会调工具办事**。
> 基于 LangChain + DeepSeek V4.1 Flash 构建，Python 实现。

**核心能力**：Agent 工具调用 · 三层记忆架构 · 多模态理解（图片/表情包）· RAG 检索 · 情绪状态机 · 语音与表情包输出

---

## 项目亮点

| 能力 | 实现 | 关键指标 |
| --- | --- | --- |
| **Agent 工具调用** | 模型自主决策调用 5 类工具，多轮"模型→工具→模型"循环 | 工具调用正确率 **80%**，任务达成率 **100%** |
| **三层记忆架构** | 会话历史（SQLite 持久化）+ 用户画像（LLM 自动抽取）+ 情绪状态（好感度/心情/精力） | 跨进程重启记忆不丢 |
| **多模态理解** | 图片（解密/UIA 截图/界面触发下载三通道）+ 表情包（屏幕截图）→ 视觉模型转译入对话 | 端到端延迟 P95 **3.2s** |
| **Agentic RAG** | 对比"预注入"与"工具检索"两种策略并实测 | 工具检索使调用正确率 **+15pp**，token **-4.4%** |
| **可观测性** | 每轮记录工具调用链、token 用量、延迟 | 平均 **1828 token/轮** |

---

## 系统架构

```mermaid
graph TB
    subgraph 接入层
        WX[微信客户端] -->|本地加密数据库| LISTEN[DB 监听器<br/>1s 轮询 + sort_seq 水位线]
    end

    subgraph 理解层
        LISTEN --> FILTER[五层消息过滤<br/>自己发的→回声去重→群聊@唤醒→类型分流]
        FILTER --> KIND{消息类型}
        KIND -->|图片| IMG[图片获取<br/>①解密缓存 ②点图触发下载]
        KIND -->|表情包| EMO[屏幕截图捕获]
        KIND -->|文本| CTX[文本直通]
        IMG --> VLM[多模态模型识别]
        EMO --> VLM
        VLM --> CTX
    end

    subgraph 记忆层
        CTX --> MEM[(SQLite<br/>messages 历史 / profiles 画像+情绪)]
    end

    subgraph 决策层
        MEM --> BUILD[上下文组装<br/>用户画像 + 情绪状态 + RAG 片段]
        BUILD --> AGENT{{Agent 执行循环}}
        AGENT -->|tool_calls| TOOLS[工具集<br/>时间 / 天气 / 计算 / 知识库 / 历史检索]
        TOOLS -->|ToolMessage| AGENT
        AGENT --> REPLY[最终回答]
    end

    subgraph 表达层
        REPLY --> SPLIT[分条拆分 + 打字延迟]
        SPLIT --> WXOUT[发送回微信]
        REPLY --> TTS[edge-tts 语音合成]
        REPLY --> ST[按情绪选表情包]
    end
```

---

## 核心能力详解

### 1. Agent 工具调用（与"聊天机器人"的分界线）

模型不再只是生成文本，而是能**自主判断并执行动作**：

```python
model_with_tools = model.bind_tools(TOOLS)

def run_agent_loop(prompt_value):
    messages = prompt_value.to_messages()
    ai = model_with_tools.invoke(messages)
    for _ in range(MAX_TOOL_ROUNDS):
        if not ai.tool_calls: break
        messages.append(ai)
        for call in ai.tool_calls:
            result = TOOL_MAP[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai = model_with_tools.invoke(messages)
    return ai
```

已注册工具：`get_current_time` · `get_weather` · `calculate`（AST 白名单求值，非 eval）· `search_knowledge` · `search_history`

实测行为：
> 用户「现在几点？顺便帮我算一下 25 乘以 4」
> → 模型一次调用 `get_current_time()` + `calculate({'expression':'25*4'})` → 汇总作答

### 2. 三层记忆架构

| 层次 | 存储 | 更新策略 |
| --- | --- | --- |
| **会话历史** | `messages` 表 | 每轮自动写入，按会话隔离，载入最近 N 条 |
| **用户画像** | `profiles.facts` | 每 N 轮由 LLM 抽取长期事实，去重 + 限长（防膨胀） |
| **情绪状态** | `profiles.affinity/mood_score/energy` | 情感词典即时评分；心情按离线时长衰减，精力随时间恢复 |

群聊场景下**画像与情绪按发言人分别记录**，避免不同人互相污染。

### 3. 多模态理解（三条通道）

异构媒体的获取方式完全不同，这是踩坑最多的部分：

| 类型 | 通道 | 约束 |
| --- | --- | --- |
| 图片 | ① 解密本地 `.dat` 缓存 | 需图片已被查看过 |
| 图片 | ② 模拟点击消息触发微信下载原图 | 兜底路径，需窗口可见 |
| 表情包 | 屏幕截图（数据库内为加密数据，无法解密） | 需窗口可见 |

获取到图片后交给多模态模型转译为文本描述，再作为对话内容进入记忆与情绪链路。

### 4. Agentic RAG 策略对比

同一套评测集，切换 RAG 策略实测（20 条用例，6 大类别）：

| 指标 | 预注入 inject | 工具检索 tool（Agentic RAG） |
| --- | --- | --- |
| 任务达成率 | **100%** | **100%** |
| 工具调用正确率 | 65.0% | **80.0%** ↑15pp |
| 需要工具召回率 | 56.2% | **75.0%** ↑18.8pp |
| 无需工具不误调率 | 100% | 100% |
| 平均每轮 token | 1912 | **1828** ↓4.4% |
| 延迟 P50 / P95 | 0.94s / 4.35s | 1.27s / **3.17s** |
| 平均延迟 | 1.35s | 1.41s |

**结论**：把检索交给模型自主决策，工具调用质量显著提升、token 略降、P95 尾延迟更低；代价是 P50 略升（部分请求多一轮模型往返）。

---

## 评测体系

项目自带可复现的评测：

```powershell
venv\Scripts\python.exe 评测.py                 # 预注入模式
venv\Scripts\python.exe 评测.py --mode=tool     # Agentic RAG 模式
```

- `评测集.json`：20 条用例，覆盖 工具-时间/计算/天气/知识库、无需工具、多工具组合 六大类
- `评测知识库.txt`：可验证事实的评测专用知识库（保证期望答案可判定）
- 产出 `评测报告.md`：汇总指标 + 分类表现 + 逐条明细

**两个指标要分清**：
- **工具调用正确率** —— Agent 行为质量（该调的调了、不该调的不调）
- **任务达成率** —— 用户侧结果质量（回答是否真的对）

> 注：模型判断"不用工具也能答对"时未调用工具，属合理的自主决策，只要结果正确即计入任务达成（实测有 4~6 条属此类）。

---

## 技术难点与解法

| 难点 | 现象 | 解法 |
| --- | --- | --- |
| **消息回声自循环** | 机器人回复自己的回复，无限对话 | `is_self` 判定 + 60 秒内容去重双保险 |
| **群聊消息 sender 不可靠** | 无法判断谁在说话 | 从内容前缀 `wxid: 内容` 解析发送者并剥离 |
| **表情包无法解密** | 数据库内为加密数据 | 改用 UIA 屏幕截图通道 |
| **图片需已缓存** | 未查看过的图取不到 | 模拟点击消息触发微信下载原图 |
| **Windows GBK 控制台崩溃** | emoji 打印抛 `UnicodeEncodeError` | 输出流 `errors="replace"` 兜底 |
| **孤立代理对崩溃** | 特殊字符导致 API 序列化失败 | 输入统一 UTF-8 清洗 |
| **上下文无限增长** | 长对话撑爆窗口 | `trim_messages` 字符估算修剪 |
| **跨会话隐私** | 群成员记忆互相污染 | 画像/情绪按发言人分键存储 |

---

## 快速开始

### 基础版（命令行，跨平台）
```bash
cd 基础版
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # 填入 DEEPSEEK_API_KEY
python chatbot_v2.py        # 记忆 + 多会话 + 历史修剪 + 流式输出
```

### 微信版（Windows + 微信桌面端）
```powershell
cd 微信版
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-wechat.txt
copy .env.example .env      # 填入 DEEPSEEK_API_KEY
venv\Scripts\python.exe wechat_bot.py
```

配套工具：
```powershell
venv\Scripts\python.exe 查看状态.py --facts   # 查看好感度/心情/精力与用户画像
venv\Scripts\python.exe 评测.py               # 跑 Agent 能力评测
venv\Scripts\python.exe test_vision.py        # 多模态识图自检
```

---

## 项目结构

```
├─ 基础版/                  命令行版（记忆/多会话/修剪/流式）
└─ 微信版/
   ├─ wechat_bot.py         主程序：消息过滤 → 记忆 → Agent → 表达
   ├─ 工具.py               Agent 工具集（可扩展）
   ├─ 语音.py               语音合成（edge-tts / GPT-SoVITS）
   ├─ 评测.py               评测执行器
   ├─ 评测集.json           20 条评测用例
   ├─ 查看状态.py            记忆与情绪状态查看器
   ├─ 知识库.txt            RAG 知识库（可替换）
   └─ 待开发_语音理解.md     语音消息理解设计文档
```

---

## ⚠️ 免责声明

- 本项目**仅供学习与技术研究**。请遵守所在地法律法规，以及 DeepSeek、微信等第三方服务的条款。
- **微信版基于对个人微信桌面客户端的自动化操作**，**违反《微信个人账号使用规范》**，存在**账号被限制或封禁**的风险。请务必使用小号测试、控制操作频率，**不要用于营销、群发或任何商业用途**。
- 使用本项目产生的任何直接或间接后果（账号封禁、数据丢失、API 费用、法律纠纷等）**由使用者自行承担**。
- `.env`（密钥）与 `memory.db`（聊天记录与用户画像）属敏感数据，请勿外发或提交到版本库。

## 许可

[MIT License](LICENSE)
