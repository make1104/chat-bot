# -*- coding: utf-8 -*-
"""规划与反思：让 Agent 从"会调工具"进化到"会做任务"

两个能力，都是**按需触发**（简单闲聊不增加任何额外调用，不拖慢响应）：

1. Planner（规划）：把复杂请求拆成有序步骤，作为执行指引注入上下文
        → 解决"多步任务只做一半"的问题
2. Reflector（反思）：对草稿回答做质检（答非所问 / 编造 / 不像人话），不合格则修订
        → 解决"工具调了但结论错"的问题

依赖注入：本模块不直接持有模型，所有函数接受 model 参数，便于替换与单测。
"""

import re

from langchain_core.prompts import ChatPromptTemplate

# ==================== 一、复杂度判定（规则，零成本）====================

# 出现这些词通常意味着多步任务
COMPLEX_HINTS = (
    "先", "然后", "再", "接着", "最后", "分别", "对比", "比较",
    "顺便", "并且", "同时", "一起", "整理", "总结", "步骤", "流程",
)
_MULTI_QUESTION = re.compile(r"[?？].+[?？]")


def need_plan(text: str) -> bool:
    """是否需要先做规划。短消息、单一意图一律返回 False。"""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if _MULTI_QUESTION.search(t):          # 一句话里两个问号 → 多问题
        return True
    return any(h in t for h in COMPLEX_HINTS) and len(t) >= 15


# ==================== 二、Planner ====================

_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是任务规划器。把用户的请求拆解成 2~5 个可执行步骤，每步一行、以数字开头，"
     "只写要做什么，不要解释、不要输出其它内容。\n"
     "原则：能用工具解决的步骤要写明用哪个工具；如果任务其实一步就能完成，只输出一行。"),
    ("human", "{question}"),
])


def make_plan(model, question: str) -> str:
    """生成执行计划；失败返回空字符串（调用方应容错）。"""
    try:
        res = (_PLAN_PROMPT | model).invoke({"question": question})
        return (res.content or "").strip()
    except Exception as e:
        print(f"⚠️ 规划失败：{e}")
        return ""


# ==================== 三、Reflector ====================

_VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是回答质检员，只做判定，不要重写答案。判断【草稿回答】是否满足全部要求：\n"
     "1) 正面回答了用户的问题，没有答非所问、没有漏掉其中一部分；\n"
     "2) 没有编造信息：涉及事实与数字时，必须能在【可用依据】中找到出处"
     "（依据包含系统提供的参考资料与本轮工具结果，请逐字核对，找不到再下结论）；\n"
     "3) 像真人在聊天，口语化、简短，没有书面语和括号动作描写。\n"
     "只输出一行：全部满足输出 PASS；"
     "仅当发现**明确的编造**（依据中完全不存在的信息被当成事实写出）"
     "或**严重答非所问**时，才输出 FAIL|一句话说明具体问题。"
     "只要不确定，一律输出 PASS。"),
    ("human", "用户问题：{question}\n\n【可用依据】\n{evidence}\n\n草稿回答：{draft}"),
])

_REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "根据质检意见修订回答。要求：口语化、简短（1~3 句）、不写括号动作描写、"
     "涉及数字必须与【可用依据】一致。若依据充分则保留原有正确信息，"
     "不要因为保守而删掉有出处的关键数字。只输出修订后的回答本身。"),
    ("human", "用户问题：{question}\n\n【可用依据】\n{evidence}\n\n原草稿：{draft}\n\n质检意见：{feedback}"),
])


def verify(model, question: str, draft: str, evidence: str):
    """质检草稿。返回 (是否通过, 问题描述)。

    质检本身出错时按"通过"处理，避免阻塞正常回复。
    """
    try:
        res = (_VERIFY_PROMPT | model).invoke(
            {"question": question, "evidence": evidence or "（本轮无额外依据）", "draft": draft})
        out = (res.content or "").strip()
    except Exception as e:
        print(f"⚠️ 质检失败（跳过）：{e}")
        return True, ""
    if out.upper().startswith("PASS"):
        return True, ""
    return False, out.split("|", 1)[-1].strip()[:120] or "回答质量未达标"


def revise(model, question: str, draft: str, feedback: str, evidence: str) -> str:
    """按质检意见修订；失败则返回原稿。"""
    try:
        res = (_REVISE_PROMPT | model).invoke(
            {"question": question, "evidence": evidence or "（本轮无额外依据）",
             "draft": draft, "feedback": feedback})
        new = (res.content or "").strip()
        return new or draft
    except Exception as e:
        print(f"⚠️ 修订失败（保留原稿）：{e}")
        return draft


# ==================== 四、反思触发条件 ====================


def need_reflect(draft: str, tool_calls: int, plan_used: bool,
                 sample_rate: float, rnd: float) -> bool:
    """是否需要质检：用了工具/做过规划必查，其余按比例抽查。"""
    if not draft or len(draft.strip()) < 8:
        return False
    if tool_calls > 0 or plan_used:
        return True
    return rnd < sample_rate


if __name__ == "__main__":
    # 自测：python 规划.py
    samples = [
        "你好呀",
        "现在几点了？",
        "先查一下专业版价格，再帮我算买三年一共多少钱",
        "现在几点？顺便算一下 25 乘以 4",
        "帮我看看北京天气怎么样",
    ]
    print("复杂度判定（是否需要规划）：")
    for s in samples:
        print("  %-34s -> %s" % (s, "需要规划" if need_plan(s) else "直接回答"))
