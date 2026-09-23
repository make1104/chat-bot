# -*- coding: utf-8 -*-
"""Agent 工具集：让机器人从"只会聊天"变成"能办事"

每个工具都是 LangChain Tool（带类型标注 + 文档字符串），
模型会根据用户意图**自主决定**是否调用、调用哪个、传什么参数——
这正是「Agent」与「聊天机器人」的分界线。

新增工具只需三步：
    1. 用 @tool 装饰一个函数，写清参数类型和用途（文档字符串会被喂给模型）
    2. 把函数加进文件末尾的 TOOLS 列表
    3. 重启机器人（无需改主程序）
"""

import ast
import datetime
import json
import operator
import urllib.parse
import urllib.request

from langchain_core.tools import tool

# ==================== 工具 1：时间 ====================


@tool
def get_current_time() -> str:
    """查询当前的日期、时间和星期。当用户问"现在几点""今天几号""今天星期几"时使用。"""
    now = datetime.datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return f"当前时间：{now:%Y-%m-%d %H:%M:%S}，星期{weekday}"


# ==================== 工具 2：天气（wttr.in，免费无需 Key）====================


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为城市名，例如 "北京"、"上海"、"Tokyo"。"""
    city = (city or "").strip()
    if not city:
        return "请提供城市名，例如：北京"
    url = "https://wttr.in/%s?format=j1&lang=zh" % urllib.parse.quote(city)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        cur = data["current_condition"][0]
        desc = ""
        for key in ("lang_zh", "weatherDesc"):
            arr = cur.get(key) or []
            if arr and arr[0].get("value"):
                desc = arr[0]["value"]
                break
        return ("%s当前天气：%s，气温 %s°C（体感 %s°C），湿度 %s%%，风速 %s km/h"
                % (city, desc or "未知", cur.get("temp_C"), cur.get("FeelsLikeC"),
                   cur.get("humidity"), cur.get("windspeedKmph")))
    except Exception as e:
        return "天气查询失败：%s" % e


# ==================== 工具 3：计算器（AST 安全求值，不用 eval）====================

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    """只允许数字与基础运算符的白名单求值。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("只支持数字与 + - * / // %% ** 运算")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。参数 expression 是算式，例如 "23*47+15"、"(1+2)**3"。
    涉及任何算术、百分比、复利等精确计算时都必须使用本工具，不要自己心算。"""
    try:
        expr = (expression or "").strip()
        value = _safe_eval(ast.parse(expr, mode="eval"))
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return "%s = %s" % (expr, value)
    except Exception as e:
        return "计算失败：%s" % e


# ==================== 工具 4/5：由主程序注入检索能力的工具 ====================
# 用"注入"而不是 import，避免主程序与本模块循环依赖。

_KB_SEARCH = None
_HISTORY_SEARCH = None


def set_knowledge_search(fn) -> None:
    """注入知识库检索函数（由主程序在启动时调用）。"""
    global _KB_SEARCH
    _KB_SEARCH = fn


def set_history_search(fn) -> None:
    """注入历史对话检索函数（由主程序在启动时调用）。"""
    global _HISTORY_SEARCH
    _HISTORY_SEARCH = fn


@tool
def search_knowledge(query: str) -> str:
    """在本地知识库中检索资料。当用户询问产品介绍、价格套餐、使用方法、售后等
    知识库内的信息时使用。参数 query 是检索关键词或问题。"""
    if _KB_SEARCH is None:
        return "（知识库尚未就绪）"
    return _KB_SEARCH(query)


@tool
def search_history(keyword: str) -> str:
    """在历史聊天记录中搜索关键词，用于回忆之前聊过的内容。
    当用户问"我之前说过什么""上次提到的那件事"时使用。参数 keyword 是搜索关键词。"""
    if _HISTORY_SEARCH is None:
        return "（历史记录检索尚未就绪）"
    return _HISTORY_SEARCH(keyword)


# ==================== 工具注册表 ====================
# 往这里加函数即可扩展 Agent 能力

TOOLS = [
    get_current_time,
    get_weather,
    calculate,
    search_knowledge,
    search_history,
]

TOOL_MAP = {t.name: t for t in TOOLS}


if __name__ == "__main__":
    # 自测：python 工具.py
    print("已注册工具：")
    for t in TOOLS:
        print("  · %-18s %s" % (t.name, (t.description or "").splitlines()[0][:40]))
    print()
    print("时间工具 ->", get_current_time.invoke({}))
    print("计算工具 ->", calculate.invoke({"expression": "(1+2)**3*5"}))
    print("计算工具（非法）->", calculate.invoke({"expression": "__import__('os')"}))
