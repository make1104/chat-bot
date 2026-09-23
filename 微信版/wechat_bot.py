# -*- coding: utf-8 -*-
"""微信聊天机器人（DeepSeek 版）—— 基于 wechatauto-replica（微信 4.x）+ LangChain

运行环境（必须在 Windows 上运行）：
    - Windows 10/11 + 已登录的微信桌面客户端（4.x）
    - Python 3.12（wechatauto-replica 需要；3.14 暂无 winsdk 预编译包）

⚠️ 风险提示：
    个人微信自动化违反《微信个人账号使用规范》，存在封号风险！
    建议用小号测试，控制回复频率，不要用于营销/群发。

快速开始：
    1) venv\\Scripts\\python.exe -m pip install wechatauto-replica langchain langchain-core langchain-deepseek python-dotenv
    2) 复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY
    3) 登录微信桌面版，保持窗口打开
    4) venv\\Scripts\\python.exe wechat_bot.py
    5) 先用"文件传输助手"发消息测试
"""

import base64
import collections
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 控制台编码兜底：GBK 控制台下打印 emoji 会抛 UnicodeEncodeError，这里改为替换字符
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 语音模块（同目录的 语音.py；没装 edge-tts 时自动降级为不发声）
try:
    from 语音 import cleanup as voice_cleanup
    from 语音 import synthesize

    VOICE_IMPORT_OK = True
except Exception as _e:
    VOICE_IMPORT_OK = False
    print(f"⚠️ 语音模块不可用（{_e}），语音功能会自动跳过")

# 工具集（同目录的 工具.py）：让机器人能"办事"而不只是聊天
try:
    from 工具 import TOOLS, TOOL_MAP, set_history_search, set_knowledge_search

    TOOLS_IMPORT_OK = True
except Exception as _e:
    TOOLS, TOOL_MAP, TOOLS_IMPORT_OK = [], {}, False
    print(f"⚠️ 工具模块不可用（{_e}），将退化为纯对话模式")

# 规划与反思模块（同目录的 规划.py）
try:
    import 规划 as QUESTION_PLANNER

    PLANNER_IMPORT_OK = True
except Exception as _e:
    PLANNER_IMPORT_OK = False
    print(f"⚠️ 规划模块不可用（{_e}），将跳过规划与反思")

    class _NullPlanner:
        """兜底实现：模块缺失时保证主流程无需判空。"""

        @staticmethod
        def need_plan(text):
            return False

        @staticmethod
        def make_plan(model, question):
            return ""

        @staticmethod
        def detect_risk(*args, **kwargs):
            return False, []

        @staticmethod
        def verify(*args, **kwargs):
            return True, ""

        @staticmethod
        def revise(model, question, draft, feedback, tools):
            return draft

    QUESTION_PLANNER = _NullPlanner()

load_dotenv()

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("未检测到 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入密钥。")

# ==================== RAG：知识库检索（关键词版，零额外依赖） ====================
# 想要真正的向量语义检索（bge 模型 + FAISS），参考 WSL 项目的 chatbot_rag.py，
# 但需要额外安装 sentence-transformers / faiss-cpu（Windows 上体积较大）
KB_FILE = "知识库.txt"   # 知识库文件（放本项目目录，UTF-8 编码的 txt）
TOP_K = 3                # 每次检索返回的片段数
RAG_ALWAYS_INJECT = True  # True=每次对话都预注入知识片段（稳，费 token）
                          # False=不预注入，由模型自主决定是否调用 search_knowledge 工具（Agentic RAG）

if os.path.exists(KB_FILE):
    with open(KB_FILE, encoding="utf-8") as f:
        _kb_text = f.read()
    _splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    _chunks = _splitter.split_text(_kb_text)
    print(f"📚 知识库已加载：{len(_chunks)} 个片段")
else:
    _chunks = []
    print("⚠️ 未找到知识库.txt，本次运行不带 RAG（只靠模型自身知识）")


def retrieve(query: str) -> str:
    """关键词检索：按字符重合度打分，取最相关的 TOP_K 个片段。"""
    if not _chunks:
        return "（知识库为空）"
    docs = sorted(_chunks, key=lambda c: -len(set(query) & set(c)))[:TOP_K]
    return "\n\n".join(f"[片段{i+1}] {d}" for i, d in enumerate(docs))


def search_history(keyword: str) -> str:
    """在**当前会话**的历史记录里搜索关键词（供 Agent 的 search_history 工具调用）。

    只查当前会话，避免把别的聊天内容串进来。
    """
    kw = (keyword or "").strip()
    who = getattr(_current_session, "who", None)
    if not kw:
        return "（请提供要搜索的关键词）"
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE who=? AND content LIKE ? "
            "ORDER BY id DESC LIMIT 6",
            (who, f"%{kw}%"),
        ).fetchall() if who else []
        conn.close()
    except Exception as e:
        return f"历史检索失败：{e}"
    if not rows:
        return f"当前会话的历史记录里没有找到「{kw}」"
    lines = []
    for role, content in rows:
        speaker = "对方" if role == "human" else "我"
        lines.append(f"{speaker}：{content[:60]}")
    return "\n".join(reversed(lines))


# 把检索能力注入工具模块（工具写在 工具.py，用注入避免循环依赖）
set_knowledge_search(retrieve)
set_history_search(search_history)


# ==================== LangChain：带记忆的对话核心（同 v2 + RAG） ====================
MODEL_NAME = "deepseek-flash"   # DeepSeek V4.1 Flash：原生多模态，文字与识图共用同一个模型
model = ChatDeepSeek(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的中文助手，回答简洁准确。\n\n"
                   "【关于对方】\n{profile}\n\n"
                   "【你当前的状态】\n{state}\n"
                   "让上面的状态自然地影响你的语气、称呼和回复长短（心情差就冷淡些，"
                   "好感度高就更亲近随意，精力低就更短），但绝对不要说出这些数值。\n\n"
                   "【参考知识】\n{context}\n\n"
                   "回答时优先依据参考知识，并结合你对对方的了解自然交流；"
                   "如果参考知识里没有相关信息，请如实说明，不要编造。\n\n"
                   "【输出要求】\n"
                   "- 只输出可以直接发出去的聊天文字，像真人在微信里打字一样自然、口语化。\n"
                   "- 禁止任何括号或星号包裹的动作、神态、心理、场景描写"
                   "（例如（微笑）、(歪头)、【动作】、*叹气* 一律不许出现），不要写旁白。\n"
                   "- 不要书面语，不要分点罗列，控制在 1~3 句。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def count_tokens(messages) -> int:
    """粗略 token 计数：按字符数估算（DeepSeek 不支持模型数 token）。"""
    return sum(len(getattr(m, "content", "") or "") for m in messages)


trimmer = trim_messages(
    max_tokens=500,        # 历史上限（按字符估算），可按需调大/调小
    strategy="last",
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# ==================== Agent：规划 + 工具调用 + 反思 ====================
TOOLS_ENABLED = True        # 是否启用工具调用（关掉则退化为纯对话机器人）
MAX_TOOL_ROUNDS = 4         # 单轮对话最多几次"模型 → 工具 → 模型"
PLANNING_ENABLED = True     # 复杂请求是否先分解步骤（Plan-and-Execute）
REFLECTION_ENABLED = True   # 是否对草稿做质检与修订（Reflexion）
REFLECT_MODE = "risk"       # "risk"=仅高风险时质检（省成本、避免误伤正确答案）
                            # "always"=只要用过工具或做过规划就质检（旧策略，作为对照）
REFLECT_SAMPLE_RATE = 0.05  # 非高风险场景的抽样质检比例（用于质量监控）

# 把工具绑定到模型：模型会自主决定是否调用、调用哪个、传什么参数
model_with_tools = model.bind_tools(TOOLS) if (TOOLS_ENABLED and TOOLS_IMPORT_OK) else model

# 当前会话（线程局部）：让"检索历史"这类工具知道该查哪个会话，避免串台
_current_session = threading.local()

# 可观测性：记录每轮对话的工具调用与 token 用量（供评测 / 监控读取）
TURN_STATS = collections.deque(maxlen=500)


def _acc_usage(ai, acc: dict) -> None:
    """累计一条模型返回的 token 用量。"""
    u = getattr(ai, "usage_metadata", None) or {}
    acc["in"] = acc.get("in", 0) + int(u.get("input_tokens") or 0)
    acc["out"] = acc.get("out", 0) + int(u.get("output_tokens") or 0)


def _last_human_text(messages) -> str:
    """取出最后一条用户消息的纯文本（用于规划与反思）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):     # 多模态内容块
                return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _tools_digest(calls_made) -> str:
    """把本轮工具调用与结果整理成文本（供质检/修订作为依据）。

    注意：工具输出必须保留足够完整——曾因截断到 120 字符，质检员看不到
    真实检索内容而误判为"编造"，反而把正确答案改坏。
    """
    if not calls_made:
        return ""
    return "\n".join("%s(%s) -> %s" % (c["name"], c["args"], c["result"][:1000])
                     for c in calls_made)


def _evidence(messages, calls_made) -> str:
    """汇总"模型作答时可用的全部依据"，供质检与修订使用。

    必须同时包含系统提示里的参考资料——否则模型合理复述已注入的上下文事实，
    会被质检误判成"编造"并改坏正确答案（实测踩过这个坑）。
    """
    parts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                parts.append("【系统提供的参考资料】\n" + c.strip()[:2500])
            break
    digest = _tools_digest(calls_made)
    if digest:
        parts.append("【本轮工具结果】\n" + digest)
    return "\n\n".join(parts)


def run_agent_loop(prompt_value) -> AIMessage:
    """Agent 执行循环：规划 → 工具调用 → 质检修订 → 最终回答。

    三种模式按需触发，简单闲聊不会增加任何额外开销：
      · Plan：请求包含多步意图时，先分解步骤再执行
      · Act ：模型自主调用工具，结果回喂后继续推理
      · Reflect：用过工具或做过规划时，质检草稿并必要时修订
    """
    messages = prompt_value.to_messages()
    question = _last_human_text(messages)
    calls_made = []
    tool_errors = []
    usage = {"in": 0, "out": 0}
    rounds = 0
    plan_used = False
    reflected = False

    # ---------- ① Plan：复杂请求先分解 ----------
    if PLANNING_ENABLED and QUESTION_PLANNER.need_plan(question):
        plan = QUESTION_PLANNER.make_plan(model, question)
        if plan:
            plan_used = True
            print("🧭 执行计划：")
            for ln in plan.splitlines():
                if ln.strip():
                    print("   " + ln.strip()[:70])
            # 插到 system 之后，作为执行指引
            idx = 1 if messages and not isinstance(messages[0], SystemMessage) else 0
            messages.insert(idx, SystemMessage(content="【执行计划】\n" + plan))

    # ---------- ② Act：工具调用循环 ----------
    ai = model_with_tools.invoke(messages)
    rounds += 1
    _acc_usage(ai, usage)

    for _ in range(MAX_TOOL_ROUNDS):
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            break
        messages.append(ai)
        for call in calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            print(f"🔧 工具调用：{name}({args})")
            fn = TOOL_MAP.get(name)
            try:
                result = fn.invoke(args) if fn else f"未注册的工具：{name}"
            except Exception as e:
                result = f"工具执行失败：{e}"
            res_text = str(result)
            head = res_text[:30]
            if (not res_text.strip() or "失败" in head or "未注册" in head
                    or "未找到" in head or "为空" in head):
                tool_errors.append(f"{name}: {res_text[:40]}")
            print(f"   ↳ {res_text[:60]}")
            calls_made.append({"name": name, "args": args, "result": res_text[:1000]})
            messages.append(ToolMessage(content=res_text, tool_call_id=call.get("id") or ""))
        ai = model_with_tools.invoke(messages)
        rounds += 1
        _acc_usage(ai, usage)

    draft = ai.content or ""

    # ---------- ③ Reflect：风险触发 → 质检 → 修订 ----------
    if REFLECTION_ENABLED:
        ev = _evidence(messages, calls_made)
        risky, reasons = QUESTION_PLANNER.detect_risk(
            draft, ev, len(calls_made), tool_errors, question)
        should = risky
        if not should and REFLECT_MODE == "always" and (calls_made or plan_used):
            should = True
        if not should and random.random() < REFLECT_SAMPLE_RATE:
            should, reasons = True, ["抽样质检"]

        if should:
            if reasons:
                print("⚠️ 质检触发：" + "；".join(reasons))
            ok, feedback = QUESTION_PLANNER.verify(model, question, draft, ev)
            if not ok:
                print(f"🔍 质检未通过：{feedback}")
                revised = QUESTION_PLANNER.revise(model, question, draft, feedback, ev)
                if revised and revised != draft:
                    reflected = True
                    draft = revised
                    print("✍️ 已修订回答")
            else:
                print("✅ 质检通过")
        reflect_triggered = should
    else:
        reflect_triggered = False

    if reflected:
        ai = AIMessage(content=draft)

    TURN_STATS.append({
        "session": getattr(_current_session, "who", None),
        "tools": [c["name"] for c in calls_made],
        "calls": calls_made,
        "rounds": rounds,
        "planned": plan_used,
        "reflected": reflected,
        "reflect_triggered": reflect_triggered,
        "tool_errors": tool_errors,
        "input_tokens": usage["in"],
        "output_tokens": usage["out"],
        "ts": time.time(),
    })
    return ai


chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | trimmer,
        # RAG 两种策略：预注入（默认，稳）或交给模型用工具检索（省 token，更 Agent）
        context=lambda x: (retrieve(x["input"]) if RAG_ALWAYS_INJECT
                           else "（未预注入参考知识；如需查询资料，请调用 search_knowledge 工具）"),
        profile=lambda x: format_profile(x["who"]),   # 长期记忆：用户画像
        state=lambda x: format_state(x["who"]),       # 情绪状态：心情/好感度/精力
    )
    | prompt
    | RunnableLambda(run_agent_loop)                  # Agent：带工具调用的执行循环
)

# ==================== 长期记忆 + 用户画像（SQLite） ====================
DB_FILE = "memory.db"      # 记忆数据库文件（放项目目录，首次运行自动创建）
HISTORY_LOAD = 40          # 每次从数据库载入多少条历史消息
PROFILE_EVERY = 6          # 每多少轮对话抽取一次用户画像
PROFILE_MAX_FACTS = 20     # 每个用户最多保留多少条画像

# ==================== 情绪状态 + 好感度 ====================
STATE_ENABLED = True          # 是否启用情绪/好感度状态机
AFFINITY_MAX = 100            # 好感度上限
MOOD_DECAY_HOURS = 6          # 超过这么久没聊天，心情回归平静
ENERGY_PER_MSG = 2            # 每条消息消耗的精力
ENERGY_RECOVER_PER_HOUR = 12  # 每小时恢复的精力

# 简单的情感词典（命中即加减分，即时生效、不额外消耗 API）
_POSITIVE_WORDS = ("谢谢", "感谢", "厉害", "好棒", "真棒", "喜欢你", "可爱", "哈哈", "嘻嘻",
                   "开心", "爱你", "真好", "有趣", "好玩", "牛", "强", "辛苦")
_NEGATIVE_WORDS = ("讨厌", "无聊", "闭嘴", "烦人", "滚", "傻", "笨", "垃圾", "丑", "恶心", "别烦")


def _migrate(conn) -> None:
    """给已有的 profiles 表补充状态字段（兼容旧数据库）。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    for name, ddl in (
        ("affinity", "ALTER TABLE profiles ADD COLUMN affinity INTEGER NOT NULL DEFAULT 0"),
        ("mood_score", "ALTER TABLE profiles ADD COLUMN mood_score INTEGER NOT NULL DEFAULT 0"),
        ("energy", "ALTER TABLE profiles ADD COLUMN energy INTEGER NOT NULL DEFAULT 100"),
        ("state_ts", "ALTER TABLE profiles ADD COLUMN state_ts REAL"),
    ):
        if name not in cols:
            conn.execute(ddl)
    conn.commit()


def _db() -> sqlite3.Connection:
    """打开数据库并确保表结构存在。"""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        who TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS profiles (
        who TEXT PRIMARY KEY,
        facts TEXT NOT NULL DEFAULT '[]',
        msg_count INTEGER NOT NULL DEFAULT 0,
        first_seen REAL,
        last_seen REAL)""")
    _migrate(conn)
    conn.commit()
    return conn


class SQLiteChatHistory(BaseChatMessageHistory):
    """对话历史存进 SQLite：程序重启后依然记得之前聊过什么。"""

    def __init__(self, who: str, max_load: int = HISTORY_LOAD):
        self.who = who
        self.max_load = max_load

    @property
    def messages(self):
        conn = _db()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE who=? ORDER BY id DESC LIMIT ?",
            (self.who, self.max_load),
        ).fetchall()
        conn.close()
        out = []
        for role, content in reversed(rows):
            out.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
        return out

    def add_messages(self, messages) -> None:
        conn = _db()
        now = time.time()
        humans = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            if not content:
                continue
            role = "human" if isinstance(m, HumanMessage) else "ai"
            humans += role == "human"
            conn.execute(
                "INSERT INTO messages (who, role, content, ts) VALUES (?,?,?,?)",
                (self.who, role, content, now),
            )
        if humans:
            conn.execute("INSERT OR IGNORE INTO profiles (who, first_seen, last_seen) VALUES (?,?,?)",
                         (self.who, now, now))
            conn.execute("UPDATE profiles SET msg_count=msg_count+?, last_seen=? WHERE who=?",
                         (humans, now, self.who))
        conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = _db()
        conn.execute("DELETE FROM messages WHERE who=?", (self.who,))
        conn.commit()
        conn.close()


def load_profile(who: str) -> dict:
    conn = _db()
    row = conn.execute("SELECT facts, msg_count FROM profiles WHERE who=?", (who,)).fetchone()
    conn.close()
    if not row:
        return {"facts": [], "msg_count": 0}
    try:
        facts = json.loads(row[0] or "[]")
    except Exception:
        facts = []
    return {"facts": facts, "msg_count": row[1] or 0}


def save_facts(who: str, new_facts: list) -> None:
    """把新抽取的事实并入该用户的画像（简单去重）。"""
    if not new_facts:
        return
    facts = load_profile(who)["facts"]
    for f in new_facts:
        f = (f or "").strip(" -•·\t。")
        if not f or len(f) > 60:
            continue
        if any(f == old or f in old or old in f for old in facts):
            continue
        facts.append(f)
    facts = facts[-PROFILE_MAX_FACTS:]
    now = time.time()
    conn = _db()
    conn.execute("INSERT OR IGNORE INTO profiles (who, first_seen, last_seen) VALUES (?,?,?)",
                 (who, now, now))
    conn.execute("UPDATE profiles SET facts=?, last_seen=? WHERE who=?",
                 (json.dumps(facts, ensure_ascii=False), now, who))
    conn.commit()
    conn.close()


def format_profile(who: str) -> str:
    """把用户画像格式化成提示词里的一段。"""
    facts = load_profile(who)["facts"]
    if not facts:
        return "（暂时还不了解对方，可以在聊天中自然了解，不要编造关于对方的信息）"
    return "你已知的关于对方的长期信息：\n" + "\n".join(f"- {f}" for f in facts)


# ---------------------- 情绪状态机 ----------------------
def load_state(who: str) -> dict:
    """读取状态，并按离线时长做衰减（心情回落、精力恢复）。"""
    conn = _db()
    row = conn.execute(
        "SELECT affinity, mood_score, energy, state_ts FROM profiles WHERE who=?", (who,)
    ).fetchone()
    conn.close()
    if not row:
        return {"affinity": 0, "mood_score": 0, "energy": 100}

    affinity, mood_score, energy, state_ts = row
    affinity = affinity or 0
    mood_score = mood_score or 0
    energy = 100 if energy is None else energy
    if state_ts:
        hours = max(0.0, (time.time() - state_ts) / 3600.0)
        if hours >= MOOD_DECAY_HOURS:
            mood_score = 0                                    # 心情回归平静
        energy = min(100, energy + int(hours * ENERGY_RECOVER_PER_HOUR))
    return {"affinity": affinity, "mood_score": mood_score, "energy": energy}


def update_state(who: str, text: str) -> dict:
    """按本轮消息更新好感度/心情/精力（启发式，即时生效，不额外调 API）。"""
    st = load_state(who)
    delta = 1                                            # 正常聊天保底 +1
    pos = sum(1 for w in _POSITIVE_WORDS if w in text)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text)
    if pos:
        delta += min(pos * 2, 4)                         # 夸她/示好
    if neg:
        delta -= min(neg * 3, 6)                         # 冒犯/无聊
    if len(text.strip()) <= 2 and text.strip() not in ("在", "嗯嗯"):  # 敷衍的短消息
        delta -= 1

    affinity = max(0, min(AFFINITY_MAX, st["affinity"] + delta))
    mood_score = max(-10, min(10, st["mood_score"] + delta))
    energy = max(0, st["energy"] - ENERGY_PER_MSG)
    now = time.time()

    conn = _db()
    conn.execute("INSERT OR IGNORE INTO profiles (who, first_seen, last_seen, state_ts) VALUES (?,?,?,?)",
                 (who, now, now, now))
    conn.execute("UPDATE profiles SET affinity=?, mood_score=?, energy=?, state_ts=?, last_seen=? WHERE who=?",
                 (affinity, mood_score, energy, now, now, who))
    conn.commit()
    conn.close()
    return {"affinity": affinity, "mood_score": mood_score, "energy": energy, "delta": delta}


def _mood_label(score: int) -> str:
    if score >= 5:
        return "心情超好、兴致很高"
    if score >= 2:
        return "心情不错"
    if score >= -1:
        return "平静"
    if score >= -4:
        return "有点不耐烦"
    return "明显生气了"


def _relation_label(affinity: int) -> str:
    if affinity >= 80:
        return "非常亲密的老朋友"
    if affinity >= 50:
        return "关系不错的熟人"
    if affinity >= 20:
        return "聊得来的普通朋友"
    if affinity >= 1:
        return "刚认识不久"
    return "完全的陌生人"


def format_state(who: str) -> str:
    """把情绪/好感度格式化成提示词里的一段。"""
    if not STATE_ENABLED:
        return "（未启用状态）"
    st = load_state(who)
    lines = [
        f"心情：{_mood_label(st['mood_score'])}",
        f"好感度：{st['affinity']}/{AFFINITY_MAX}（{_relation_label(st['affinity'])}）",
        f"精力：{st['energy']}/100",
    ]
    if st["energy"] < 30:
        lines.append("（精力很低：回复要更短、更敷衍一点）")
    return "\n".join(lines)


_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "请从下面的对话中提取关于【用户】值得长期记住的事实：称呼、喜好、正在做的事、重要经历、情绪偏好等。"
               "只提取用户本人的稳定信息，忽略寒暄和临时闲聊。"
               "每行输出一条，不要编号、不要解释；如果没有任何值得记住的，只输出：无"),
    ("human", "{convo}"),
])
_extract_chain = _extract_prompt | model


def extract_facts(convo: str) -> list:
    """调用模型从最近对话里抽取用户画像事实。"""
    try:
        res = _extract_chain.invoke({"convo": convo})
        text = (res.content or "").strip()
    except Exception as e:
        print(f"⚠️ 画像抽取失败：{e}")
        return []
    if not text or text.strip() in ("无", "（无）", "None"):
        return []
    facts = [ln.strip(" -•·\t。") for ln in text.splitlines() if ln.strip()]
    return [f for f in facts if f and f != "无"]


_history_cache = {}


def get_session_history(session_id: str) -> SQLiteChatHistory:
    """每个会话一个持久化历史对象（数据实际存在 SQLite 里）。"""
    if session_id not in _history_cache:
        _history_cache[session_id] = SQLiteChatHistory(session_id)
    return _history_cache[session_id]


chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def sanitize(text: str) -> str:
    """去掉无法编码为 UTF-8 的非法字符（孤立代理对等），正常 emoji 保留。"""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def ask_bot(session_id: str, text: str, profile_key: str = None) -> str:
    """对一个会话调用一次 DeepSeek，返回回复文本。profile_key 用于取用户画像。"""
    _current_session.who = session_id   # 供 search_history 等工具定位当前会话
    try:
        response = chatbot.invoke(
            {"input": sanitize(text), "who": profile_key or session_id},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content
    except Exception as e:
        return f"（机器人暂时出错了：{e}）"


# ==================== 微信收发：wechatauto-replica（微信 4.x 复刻版） ====================
# 注意：发行包名是 wechatauto-replica，但导入模块名是 wechatauto（不是 wechatauto_replica！）
try:
    from wechatauto import WeChat  # noqa: E402
    from wechatauto import MediaDownloader, WeChatDB  # noqa: E402
except ImportError:
    raise SystemExit("未安装 wechatauto-replica，请先执行：\n"
                     "venv\\Scripts\\python.exe -m pip install wechatauto-replica")

# 要监听的联系人昵称；留空 [] 表示监听所有聊天（含群聊）
# 支持：联系人昵称、群聊名称（显示名，自动映射到 username）
# 测试时先用"文件传输助手"，确认正常后再改成你想对话的对象
LISTEN_NAMES = ["107", "文件传输助手"]

# 群聊唤醒设置：只有被 @ 时才回答
GROUP_AT_ONLY = True   # True=必须"@+名字"才回答；False=群里提到名字就回答（不要求@符号）
BOT_NAMES = []         # 机器人在群里的称呼（群内昵称/微信昵称）；留空则自动用当前微信昵称

# 发给机器人这些词，机器人就停止运行
EXIT_WORDS = ("quit", "退出")

# ==================== 分条发送 + 打字延迟（拟人化）====================
SPLIT_REPLY = True        # 是否把回复拆成多条发送（像真人一样）
MAX_SEGMENTS = 3          # 最多拆成几条
MIN_LEN_TO_SPLIT = 40     # 回复短于这个字数就整条发，不拆
TYPING_SPEED = 0.08       # 模拟打字速度（秒/字）
FIRST_DELAY = (1.0, 3.0)  # 第一条前的延迟范围（秒）——思考 + 打字
NEXT_DELAY = (0.6, 1.8)   # 后续条之间的延迟范围（秒）


def split_reply(text: str) -> list:
    """把回复按句子拆成最多 MAX_SEGMENTS 条，模拟真人分条发送。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= MIN_LEN_TO_SPLIT:
        return [text]

    # 按中英文句末标点切句（保留标点）
    parts = re.findall(r"[^。！？!?…\n]+[。！？!?…]*", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # 没有句末标点：按长度均分
        n = min(MAX_SEGMENTS, max(1, math.ceil(len(text) / 60)))
        size = math.ceil(len(text) / n)
        return [text[i:i + size] for i in range(0, len(text), size)][:MAX_SEGMENTS]

    if len(parts) <= MAX_SEGMENTS:
        return parts

    # 句子多于上限：均匀合并成 MAX_SEGMENTS 组
    per = math.ceil(len(parts) / MAX_SEGMENTS)
    groups = ["".join(parts[i:i + per]) for i in range(0, len(parts), per)]
    return groups[:MAX_SEGMENTS]


def typing_delay(text: str, first: bool) -> None:
    """按字数模拟打字耗时：首条前等久一点（思考+打字），后续间隔短一些。"""
    lo, hi = FIRST_DELAY if first else NEXT_DELAY
    wait = max(lo, min(hi, len(text) * TYPING_SPEED))
    time.sleep(wait)


# ==================== 输出净化：去掉括号与动作描写 ====================
NO_ACTION = True   # True = 去掉回复里的括号内容和动作/神态描写，只保留纯聊天文本

_ACTION_RE = re.compile(
    r"（[^（）]*）"            # 中文括号内容
    r"|\([^()]*\)"            # 英文括号内容
    r"|【[^【】]*】"          # 中文方括号内容
    r"|\[[^\[\]]*\]"          # 半角方括号内容
    r"|\*[^*\n]{1,30}\*"      # *动作*
    r"|（[^（）\n]*$"         # 未闭合的中文括号（到行尾）
    r"|\([^()\n]*$"           # 未闭合的英文括号（到行尾）
)


def strip_actions(text: str) -> str:
    """去掉括号/星号包裹的动作、神态、旁白，返回纯聊天文本。"""
    if not text:
        return ""
    t = _ACTION_RE.sub("", text)
    t = re.sub(r"[ \t]{2,}", " ", t)          # 压缩多余空格
    t = re.sub(r"[ \t]+([，。！？、；：])", r"\1", t)  # 去掉标点前的空格
    t = re.sub(r"\n{2,}", "\n", t)            # 压缩多余空行
    t = "\n".join(line.strip() for line in t.splitlines() if line.strip())
    return t.strip()   # 整条都是动作描写时返回空字符串，由调用方跳过发送


# ==================== 语音 + 表情包 ====================
VOICE_ENABLED = True       # 是否发送语音
VOICE_PROB = 0.25          # 每条回复转成语音的概率
VOICE_MAX_CHARS = 60       # 回复超过这么多字就不发语音（太长听着累）
STICKER_ENABLED = True     # 是否发表情包
STICKER_PROB = 0.3         # 每条回复后附一张表情包的概率
STICKER_DIR = "表情包"      # 表情包目录（自行放图片，见目录内说明）

_STICKER_TAGS = {
    "happy": ("开心", "高兴", "笑", "哈哈", "赞", "喜欢", "好耶"),
    "angry": ("生气", "怒", "火", "无语", "嫌弃"),
}


def pick_sticker(mood_score: int) -> str:
    """按当前心情挑一张表情包；没有匹配的就从整个目录随机挑。"""
    if not STICKER_ENABLED or not os.path.isdir(STICKER_DIR):
        return ""
    exts = (".gif", ".png", ".jpg", ".jpeg", ".webp")
    files = [
        os.path.join(STICKER_DIR, f)
        for f in os.listdir(STICKER_DIR)
        if f.lower().endswith(exts)
    ]
    if not files:
        return ""
    tags = ()
    if mood_score >= 4:
        tags = _STICKER_TAGS["happy"]
    elif mood_score <= -3:
        tags = _STICKER_TAGS["angry"]
    if tags:
        matched = [f for f in files if any(t in os.path.basename(f) for t in tags)]
        if matched:
            return random.choice(matched)
    return random.choice(files)


# ==================== 图片 / 表情包理解 ====================
# 主模型 deepseek-flash 本身是多模态的，识图直接复用它，不需要第二个模型实例。
VISION_ENABLED = True
MEDIA_DIR = "media_cache"         # 图片下载缓存目录
GROUP_REPLY_IMAGE = False         # 群里收到图片是否也回应（默认否，避免刷屏）
IMAGE_GUI_FALLBACK = True         # 本地无原图缓存时，是否模拟点击消息触发微信下载原图
                                  # （需微信窗口可见；关掉则只认已缓存的图）
VISION_PROMPT = ("用一两句中文描述这张图片或表情包的内容；如果是表情包，"
                 "说明它表达的情绪和图中的文字。直接描述，不要客套。")

_media_downloader = None
_media_db = None


def _msg_local_id(msg):
    """从消息对象取 local_id（不同版本字段名不同，逐个尝试）。"""
    raw = str(
        getattr(msg, "id", None)
        or getattr(msg, "msg_id", None)
        or getattr(msg, "local_id", "")
        or ""
    )
    lid = raw.replace("db-", "").strip()
    return int(lid) if lid.isdigit() else None


def _downloader() -> "MediaDownloader":
    """懒加载媒体下载器（全局复用一个实例）。"""
    global _media_downloader, _media_db
    if _media_downloader is None:
        os.makedirs(MEDIA_DIR, exist_ok=True)
        _media_db = WeChatDB()
        _media_downloader = MediaDownloader(_media_db, save_dir=MEDIA_DIR)
    return _media_downloader


def download_message_media(msg, who: str) -> str:
    """解密微信本地 .dat 缓存拿图片（快，不需要界面）。"""
    local_id = _msg_local_id(msg)
    if local_id is None:
        print("⚠️ 拿不到消息 local_id，无法解密图片")
        return ""
    try:
        return _downloader().download_media(who, local_id, MEDIA_DIR) or ""
    except Exception as e:
        print(f"⚠️ 图片解密失败：{e}")
        return ""


def download_image_original(msg, who: str) -> str:
    """界面路径：模拟点击图片消息 → 触发微信下载原图 → 解密保存。

    专治"原图从未在微信里点开过、本地没有 .dat 缓存"的情况。
    代价：需要微信窗口可见，并等待微信下载原图（几秒~几十秒）。
    """
    local_id = _msg_local_id(msg)
    if local_id is None:
        return ""
    try:
        return _downloader().download_image_original(who, local_id, MEDIA_DIR) or ""
    except Exception as e:
        print(f"⚠️ 界面触发原图下载失败：{e}")
        return ""


def get_message_image(msg, who: str) -> str:
    """把图片/表情包消息变成可读的图片文件路径；失败返回空字符串。

    三条路（这是之前收不到表情包/图片回复的关键）：
      · 表情包(emotion) → 数据库里是加密数据无法解密，改用屏幕截图 msg.capture()
      · 图片(image)     → ① 先解密本地 .dat 缓存（快、无需界面）
                          ② 没缓存则模拟点击消息触发微信下载原图（慢、需窗口可见）
    """
    mtype = str(getattr(msg, "type", "") or "").lower()
    cls = type(msg).__name__
    if mtype == "emotion" or "Emoji" in cls:
        if not hasattr(msg, "capture"):
            print(f"⚠️ 该消息没有 capture 方法（{cls}），无法获取表情包图片")
            return ""
        try:
            os.makedirs(MEDIA_DIR, exist_ok=True)
            path = msg.capture(save_dir=MEDIA_DIR)
            if not path:
                print("⚠️ 表情包截图失败：微信窗口是否可见？是否锁屏？")
            return path or ""
        except Exception as e:
            print(f"⚠️ 表情包截图失败：{e}")
            return ""

    # 普通图片：先试本地缓存解密，没有缓存再走界面触发下载
    path = download_message_media(msg, who)
    if not path and IMAGE_GUI_FALLBACK:
        print("ℹ️ 本地无原图缓存，改用界面点击触发微信下载原图（较慢，请稍候）…")
        path = download_image_original(msg, who)
        if path:
            print("✅ 界面触发下载原图成功")
    return path


def describe_image(path: str) -> str:
    """用多模态主模型描述图片/表情包内容。"""
    if not VISION_ENABLED or not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
        if ext == "jpg":
            ext = "jpeg"
        message = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
        ])
        return (model.invoke([message]).content or "").strip()
    except Exception as e:
        print(f"⚠️ 图片识别失败：{e}")
        return ""


def main():
    wx = WeChat()

    # 群聊中被 @ 时识别的名字：手动配置 + 自动加当前微信昵称
    bot_names = [n for n in BOT_NAMES if n]
    if getattr(wx, "nickname", None):
        bot_names.append(wx.nickname)
    bot_names = list(dict.fromkeys(bot_names))  # 去重

    # 记录最近发出的回复，用于识别"自己的回声"（库对群消息的 self 判定不可靠，用内容去重兜底）
    recent_sent = collections.deque(maxlen=100)

    # 统计每个人的对话轮数，用于"每 N 轮抽取一次画像"
    turn_count = {}

    def on_message(msg, chat):
        """收到新消息时触发（回调式监听）。msg=消息对象, chat=会话对象。"""
        try:
            text = (msg.content or "").strip()
        except Exception:
            text = ""
        who = str(getattr(chat, "who", ""))

        # 关键：忽略自己发的消息（is_self 判定；群聊里可能失效，靠下面的内容去重兜底）
        if getattr(msg, "is_self", False):
            return

        # ---- 图片 / 表情包：先用视觉模型看懂，再把描述当成"对方说的话" ----
        # 用 msg.type 判断（库里的取值：'image'=图片，'emotion'=动画表情），类名做兜底
        mtype = str(getattr(msg, "type", "") or "").lower()
        cls = type(msg).__name__
        is_image = mtype in ("image", "emotion") or "Image" in cls or "Emoji" in cls

        if is_image and VISION_ENABLED:
            if who.endswith("@chatroom") and not GROUP_REPLY_IMAGE:
                print(f"⏭️ 群聊图片默认不回应（GROUP_REPLY_IMAGE=False）：[{who}] type={mtype or cls}")
                return
            path = get_message_image(msg, who)
            desc = describe_image(path) if path else ""
            if not desc:
                print(f"⏭️ 图片未能识别（type={mtype or cls}，图片获取或识别失败），已跳过")
                return
            print(f"🖼️ [{who}] 图片内容：{desc}")
            text = f"[对方发来一张图片或表情包，内容是：{desc}]"

        if not text:
            if mtype and mtype != "system":
                print(f"⏭️ 跳过未处理的消息类型：{mtype}（{cls}）")
            return

        # 先剥掉群消息的发送者前缀（格式 "wxid_xxx: 内容"），后面的比较和提问都用干净文本
        m = re.match(r"^(wxid_[^\s:：]+)\s*[:：]\s*", text)
        sender_wxid = m.group(1) if m else None
        if m:
            text = text[m.end():]
        if not text:
            return

        # 内容去重：短时间窗内收到与最近发送的某条相同/包含该条的消息 → 是自己的回声，跳过
        now = time.time()
        for who0, content0, ts0 in recent_sent:
            same = (content0 == text) or (len(content0) >= 10 and content0 in text)
            if who0 == who and same and now - ts0 < 60:
                print(f"⏭️ 已跳过自己的回声：{text[:30]}…")
                return

        # 群聊唤醒：只有被 @（@+名字）时才回答
        if who.endswith("@chatroom"):
            mentioned = any(n in text for n in bot_names)
            if GROUP_AT_ONLY:
                if not ("@" in text and mentioned):
                    return  # 没被 @ → 不理会
            else:
                if not mentioned:
                    return

        print(f"📩 [{chat.who}] {text}")
        if text.lower() in EXIT_WORDS:
            print("收到退出指令，机器人停止。")
            os._exit(0)

        # 用户画像的键：群聊里按发言人区分，私聊直接按联系人
        profile_key = sender_wxid if (who.endswith("@chatroom") and sender_wxid) else who

        # 先按这条消息更新情绪状态（好感度/心情/精力），让本轮回复就带上新状态
        mood_now = 0
        if STATE_ENABLED:
            st = update_state(profile_key, text)
            mood_now = st["mood_score"]
            print(f"💗 好感度 {st['affinity']}/{AFFINITY_MAX} | 心情 {mood_now} | "
                  f"精力 {st['energy']} | 本条变化 {st['delta']:+d}")

        reply = ask_bot(who, text, profile_key)  # 会话记忆挂在 who 上，画像挂在 profile_key 上
        if NO_ACTION:
            reply = strip_actions(reply)  # 去掉括号动作/神态描写，只留纯聊天文本
            if not reply:
                print("⏭️ 回复全是动作描写，已跳过不发送")
                return

        # 分条发送 + 打字延迟（像真人一样一条条发，而不是甩一大段）
        segments = split_reply(reply) if SPLIT_REPLY else [reply]
        for i, seg in enumerate(segments):
            typing_delay(seg, first=(i == 0))
            recent_sent.append((chat.who, seg, time.time()))  # 逐条记录回声，供去重
            if who.endswith("@chatroom") and sender_wxid and i == 0:
                chat.SendMsg(seg, at=sender_wxid)  # 群聊只在第一条 @ 提问者
            else:
                chat.SendMsg(seg)
            print(f"📤 已发送 [{chat.who}] ({i + 1}/{len(segments)}): {seg[:40]}")

        # 语音：按概率把这条回复合成语音发出去（微信语音气泡发不了，发的是音频文件）
        if (VOICE_ENABLED and VOICE_IMPORT_OK and random.random() < VOICE_PROB
                and len(reply) <= VOICE_MAX_CHARS):
            audio = synthesize(reply)
            if audio:
                time.sleep(random.uniform(0.6, 1.6))
                chat.SendFiles(audio)
                print(f"🔊 已发送语音：{os.path.basename(audio)}")
                voice_cleanup()

        # 表情包：按概率附一张图（按当前心情挑）
        if STICKER_ENABLED and random.random() < STICKER_PROB:
            sticker = pick_sticker(mood_now)
            if sticker:
                time.sleep(random.uniform(0.4, 1.2))
                chat.SendFiles(sticker)
                print(f"🖼️ 已发送表情包：{os.path.basename(sticker)}")

        # 每 PROFILE_EVERY 轮抽取一次用户画像，写入长期记忆
        turn_count[profile_key] = turn_count.get(profile_key, 0) + 1
        if PROFILE_EVERY and turn_count[profile_key] % PROFILE_EVERY == 0:
            history = get_session_history(who).messages[-6:]
            convo = "\n".join(
                f"{'用户' if isinstance(msg_i, HumanMessage) else '助手'}：{msg_i.content}"
                for msg_i in history
            )
            facts = extract_facts(convo)
            save_facts(profile_key, facts)
            if facts:
                print(f"🧠 已更新 [{profile_key}] 的画像：{facts}")

    if LISTEN_NAMES:
        for name in LISTEN_NAMES:
            wx.AddListenChat(name, callback=on_message)
            print(f"📡 已监听：{name}")
    else:
        wx.AddListenAll(on_message)
        print("📡 已监听所有聊天")

    print("🤖 微信机器人已启动（Ctrl+C 停止）\n")
    wx.KeepRunning()  # 阻塞运行，消息回调在后台线程触发


if __name__ == "__main__":
    main()
