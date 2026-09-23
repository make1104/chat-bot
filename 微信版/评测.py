# -*- coding: utf-8 -*-
"""Agent 能力评测：工具调用行为 + 任务完成率 + 延迟 + token 成本

用法：
    venv\\Scripts\\python.exe 评测.py                    # 默认 RAG 预注入模式
    venv\\Scripts\\python.exe 评测.py --mode=tool        # Agentic RAG：交给模型决定是否检索
    venv\\Scripts\\python.exe 评测.py 5                  # 只跑前 5 条（调试）
    venv\\Scripts\\python.exe 评测.py --mode=tool 5

产出：
    · 控制台指标表
    · 评测报告.md / 评测报告_tool模式.md

两个核心指标要分清：
    · 工具调用正确率 —— Agent 行为质量（该调的调了、不该调的不调）
    · 任务达成率     —— 用户侧结果质量（回答是否真的对）
    注：模型认为"不需要工具也能答对"时未调用工具，属于合理的自主决策，
        只要结果正确即计入任务达成。
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, ".")

import wechat_bot as w

EVAL_SET = "评测集.json"
EVAL_KB = "评测知识库.txt"
EVAL_SESSION = "__eval__"


def parse_args():
    args = sys.argv[1:]
    mode = "inject"
    limit = None
    for a in args:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1].strip()
        elif a.isdigit():
            limit = int(a)
    return mode, limit


def load_eval_set(limit=None):
    with open(EVAL_SET, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    return cases[:limit] if limit else cases


def use_eval_knowledge():
    """换成评测专用知识库，让期望答案可验证。"""
    if not os.path.exists(EVAL_KB):
        print("警告：未找到 %s，沿用现有知识库" % EVAL_KB)
        return
    with open(EVAL_KB, encoding="utf-8") as f:
        text = f.read()
    w._chunks = w.RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_text(text)
    print("评测知识库已加载：%d 个片段" % len(w._chunks))


def run_case(case):
    w.TURN_STATS.clear()
    t0 = time.time()
    answer = w.ask_bot(EVAL_SESSION, case["输入"], EVAL_SESSION)
    elapsed = time.time() - t0

    stats = w.TURN_STATS[-1] if w.TURN_STATS else {}
    expect_tools = set(case.get("expected_tools") or [])
    got_tools = set(stats.get("tools", []))
    tool_ok = expect_tools == got_tools

    kws = case.get("expect_keywords") or []
    if kws:
        outcome = all(k in (answer or "") for k in kws)
    else:
        outcome = tool_ok          # 无可验证关键词时，以工具调用是否得当作为达成判据

    return {
        "id": case["id"], "类别": case["类别"], "输入": case["输入"],
        "期望工具": sorted(expect_tools), "实际工具": sorted(got_tools),
        "工具正确": tool_ok, "任务达成": outcome,
        "未调工具但答对": (not got_tools) and outcome and bool(expect_tools),
        "回答": (answer or "").replace("\n", " ")[:50],
        "延迟秒": round(elapsed, 2),
        "输入token": stats.get("input_tokens", 0),
        "输出token": stats.get("output_tokens", 0),
    }


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def main():
    mode, limit = parse_args()
    w.RAG_ALWAYS_INJECT = (mode != "tool")     # 运行时切换 RAG 策略
    report_file = "评测报告.md" if mode != "tool" else "评测报告_tool模式.md"

    cases = load_eval_set(limit)
    use_eval_knowledge()

    print("=" * 82)
    print("Agent 评测开始｜RAG 模式：%s｜用例：%d 条" %
          ("预注入(inject)" if mode != "tool" else "工具检索(tool)", len(cases)))
    print("=" * 82)

    results = [run_case(c) for c in cases]
    for r in results:
        print("[%2d] %-12s 工具:%-4s 达成:%-4s 期望=%-26s 实际=%-26s %5.2fs"
              % (r["id"], r["类别"], "OK" if r["工具正确"] else "MISS",
                 "OK" if r["任务达成"] else "MISS",
                 ",".join(r["期望工具"]) or "(无)", ",".join(r["实际工具"]) or "(无)",
                 r["延迟秒"]))

    total = len(results)
    tool_correct = sum(1 for r in results if r["工具正确"])
    achieved = sum(1 for r in results if r["任务达成"])
    no_tool_cases = [r for r in results if not r["期望工具"]]
    no_tool_ok = sum(1 for r in no_tool_cases if not r["实际工具"])
    tool_cases = [r for r in results if r["期望工具"]]
    tool_hit = sum(1 for r in tool_cases if set(r["期望工具"]).issubset(set(r["实际工具"])))
    saved = sum(1 for r in results if r["未调工具但答对"])

    lat = sorted(r["延迟秒"] for r in results)
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    tok_in = sum(r["输入token"] for r in results)
    tok_out = sum(r["输出token"] for r in results)
    tool_calls = sum(len(r["实际工具"]) for r in results)

    summary = [
        ("任务达成率（用户侧结果）", "%.1f%% (%d/%d)" % (pct(achieved, total), achieved, total)),
        ("工具调用完全正确率（行为）", "%.1f%% (%d/%d)" % (pct(tool_correct, total), tool_correct, total)),
        ("需要工具的召回率", "%.1f%% (%d/%d)" % (pct(tool_hit, len(tool_cases)), tool_hit, len(tool_cases))),
        ("无需工具的不误调率", "%.1f%% (%d/%d)" % (pct(no_tool_ok, len(no_tool_cases)), no_tool_ok, len(no_tool_cases))),
        ("模型自主判断免工具并答对", "%d 条" % saved),
        ("平均延迟 / P50 / P95", "%.2fs / %.2fs / %.2fs" % (sum(lat) / total, p50, p95)),
        ("平均每轮工具调用", "%.2f 次" % (tool_calls / total)),
        ("平均每轮 token", "%.0f（输入 %d / 输出 %d）" % ((tok_in + tok_out) / total, tok_in, tok_out)),
    ]

    print()
    print("=" * 82)
    print("汇总指标")
    print("=" * 82)
    for k, v in summary:
        print("  %-26s %s" % (k, v))

    by_cat = {}
    for r in results:
        d = by_cat.setdefault(r["类别"], [0, 0, 0])
        d[0] += 1
        d[1] += 1 if r["任务达成"] else 0
        d[2] += 1 if r["工具正确"] else 0
    print()
    print("  分类（达成率 / 工具正确率）：")
    for cat, (n, ok, tok) in by_cat.items():
        print("    %-14s %5.1f%% / %5.1f%%  (%d 条)" % (cat, pct(ok, n), pct(tok, n), n))

    # ---------- 报告 ----------
    lines = [
        "# Agent 评测报告",
        "",
        "- 评测时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "- RAG 模式：**%s**" % ("预注入 inject" if mode != "tool" else "工具检索 tool（Agentic RAG）"),
        "- 用例总数：%d" % total,
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    lines += ["| %s | %s |" % (k, v) for k, v in summary]
    lines += ["", "## 分类表现", "", "| 类别 | 任务达成率 | 工具正确率 | 用例数 |", "| --- | --- | --- | --- |"]
    for cat, (n, ok, tok) in by_cat.items():
        lines.append("| %s | %.1f%% | %.1f%% | %d |" % (cat, pct(ok, n), pct(tok, n), n))
    lines += ["", "## 用例明细", "",
              "| # | 类别 | 输入 | 期望工具 | 实际工具 | 工具正确 | 任务达成 | 延迟 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %.2fs |"
                     % (r["id"], r["类别"], r["输入"],
                        ",".join(r["期望工具"]) or "-", ",".join(r["实际工具"]) or "-",
                        "Y" if r["工具正确"] else "N", "Y" if r["任务达成"] else "N", r["延迟秒"]))
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print()
    print("报告已写入：%s" % report_file)

    try:
        conn = w._db()
        conn.execute("DELETE FROM messages WHERE who=?", (EVAL_SESSION,))
        conn.execute("DELETE FROM profiles WHERE who=?", (EVAL_SESSION,))
        conn.commit()
        conn.close()
        print("已清理评测会话数据（不影响真实记忆）")
    except Exception as e:
        print("清理失败：%s" % e)


if __name__ == "__main__":
    main()
