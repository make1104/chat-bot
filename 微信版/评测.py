# -*- coding: utf-8 -*-
"""Agent 能力评测：工具调用行为 + 任务完成率 + 延迟 + token 成本

用法：
    venv\\Scripts\\python.exe 评测.py                          # 默认：预注入 + 规划 + 风险式反思
    venv\\Scripts\\python.exe 评测.py --mode=tool                # Agentic RAG
    venv\\Scripts\\python.exe 评测.py --mode=tool --plan=off     # 关闭规划与反思（基线）
    venv\\Scripts\\python.exe 评测.py --reflect=always          # 反思全触发（旧策略，作对照）
    venv\\Scripts\\python.exe 评测.py --repeat=3                # 每条用例重复 3 次，看方差
    venv\\Scripts\\python.exe 评测.py --no-isolate              # 关闭用例隔离（旧行为）

评测方法学（重要）：
    · **用例隔离**：每条用例使用独立会话，避免前一条的历史影响后一条的工具决策（默认开启）
    · **重复实验**：单次 23 条用例的随机方差可能大于策略差异，用 --repeat 多次取均值 ± 标准差
    · 两个核心指标分开看：
        工具调用正确率 —— Agent 行为质量（该调的调了、不该调的不调）
        任务达成率     —— 用户侧结果质量（回答是否真的对）
      模型判断"不用工具也能答对"而未调用工具，属合理自主决策，只要结果正确即计入达成。
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
    mode, plan, reflect = "inject", "on", "risk"
    limit, repeat, isolate = None, 1, True
    for a in args:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1].strip()
        elif a.startswith("--plan="):
            plan = a.split("=", 1)[1].strip()
        elif a.startswith("--reflect="):
            reflect = a.split("=", 1)[1].strip()
        elif a.startswith("--repeat="):
            repeat = max(1, int(a.split("=", 1)[1]))
        elif a == "--no-isolate":
            isolate = False
        elif a.isdigit():
            limit = int(a)
    return mode, plan, reflect, limit, repeat, isolate


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


def run_case(case, isolate=True):
    """用例隔离：每条用例用独立会话，避免历史串扰影响工具决策。"""
    session = "%s/%d" % (EVAL_SESSION, case["id"]) if isolate else EVAL_SESSION
    w.TURN_STATS.clear()
    t0 = time.time()
    answer = w.ask_bot(session, case["输入"], session)
    elapsed = time.time() - t0

    stats = w.TURN_STATS[-1] if w.TURN_STATS else {}
    expect_tools = set(case.get("expected_tools") or [])
    got_tools = set(stats.get("tools", []))
    tool_ok = expect_tools == got_tools

    kws = case.get("expect_keywords") or []
    outcome = all(k in (answer or "") for k in kws) if kws else tool_ok

    return {
        "id": case["id"], "类别": case["类别"], "输入": case["输入"],
        "期望工具": sorted(expect_tools), "实际工具": sorted(got_tools),
        "工具正确": tool_ok, "任务达成": outcome,
        "未调工具但答对": (not got_tools) and outcome and bool(expect_tools),
        "规划": bool(stats.get("planned")),
        "质检触发": bool(stats.get("reflect_triggered")),
        "反思修订": bool(stats.get("reflected")),
        "回答": (answer or "").replace("\n", " ")[:50],
        "延迟秒": round(elapsed, 2),
        "输入token": stats.get("input_tokens", 0),
        "输出token": stats.get("output_tokens", 0),
    }


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def round_metrics(rs):
    """一轮的指标。"""
    total = len(rs)
    achieved = sum(1 for r in rs if r["任务达成"])
    tool_ok = sum(1 for r in rs if r["工具正确"])
    no_tool = [r for r in rs if not r["期望工具"]]
    no_tool_ok = sum(1 for r in no_tool if not r["实际工具"])
    lat = [r["延迟秒"] for r in rs]
    toks = [r["输入token"] + r["输出token"] for r in rs]
    return {
        "达成率": pct(achieved, total),
        "工具正确率": pct(tool_ok, total),
        "无工具误调率": pct(len(no_tool) - no_tool_ok, len(no_tool)),
        "平均延迟": sum(lat) / total,
        "平均token": sum(toks) / total,
        "质检触发": sum(1 for r in rs if r["质检触发"]),
        "规划触发": sum(1 for r in rs if r["规划"]),
        "修订": sum(1 for r in rs if r["反思修订"]),
    }


def agg(values):
    """均值 ± 标准差（单次时标准差为 0）。"""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def main():
    mode, plan, reflect, limit, repeat, isolate = parse_args()
    w.RAG_ALWAYS_INJECT = (mode != "tool")
    w.PLANNING_ENABLED = (plan != "off")
    w.REFLECTION_ENABLED = (plan != "off")
    w.REFLECT_MODE = reflect

    tag = ""
    if mode == "tool":
        tag += "_tool模式"
    if plan == "off":
        tag += "_无规划"
    elif reflect == "always":
        tag += "_反思全触发"
    report_file = "评测报告%s.md" % tag

    cases = load_eval_set(limit)
    use_eval_knowledge()

    print("=" * 84)
    print("Agent 评测｜RAG：%s｜规划：%s｜反思：%s｜用例：%d 条 × %d 轮｜隔离：%s"
          % ("预注入" if mode != "tool" else "工具检索",
             "开" if plan != "off" else "关",
             reflect if plan != "off" else "关",
             len(cases), repeat, "开" if isolate else "关"))
    print("=" * 84)

    # ---------- 执行 ----------
    all_results, per_round = [], []
    for i in range(repeat):
        rs = [run_case(c, isolate) for c in cases]
        all_results += rs
        per_round.append(round_metrics(rs))
        m = per_round[-1]
        print("第 %d 轮：达成率 %.1f%%｜工具正确率 %.1f%%｜平均延迟 %.2fs｜平均 token %.0f"
              % (i + 1, m["达成率"], m["工具正确率"], m["平均延迟"], m["平均token"]))

    if repeat == 1:
        for r in all_results:
            print("  [%2d] %-12s 工具:%-4s 达成:%-4s 期望=%-26s 实际=%-26s %5.2fs"
                  % (r["id"], r["类别"], "OK" if r["工具正确"] else "MISS",
                     "OK" if r["任务达成"] else "MISS",
                     ",".join(r["期望工具"]) or "(无)", ",".join(r["实际工具"]) or "(无)",
                     r["延迟秒"]))

    # ---------- 汇总（多轮取均值 ± 标准差）----------
    def col(key):
        return [m[key] for m in per_round]

    summary = []
    for label, key, fmt in (
        ("任务达成率（用户侧结果）", "达成率", "%.1f%%"),
        ("工具调用完全正确率（行为）", "工具正确率", "%.1f%%"),
        ("无需工具时误调率（越低越好）", "无工具误调率", "%.1f%%"),
        ("平均延迟", "平均延迟", "%.2fs"),
        ("平均每轮 token", "平均token", "%.0f"),
    ):
        mean, sd = agg(col(key))
        summary.append((label, fmt % mean, ("± %.1f" % sd) if repeat > 1 else ""))

    mean_plan, _ = agg(col("规划触发"))
    mean_trig, _ = agg(col("质检触发"))
    mean_rev, _ = agg(col("修订"))
    summary.append(("规划 / 质检 / 修订（次）",
                    "%.0f / %.0f / %.0f" % (mean_plan, mean_trig, mean_rev), ""))

    print()
    print("=" * 84)
    print("汇总指标%s" % ("（%d 轮均值 ± 标准差）" % repeat if repeat > 1 else ""))
    print("=" * 84)
    for label, val, sd in summary:
        print("  %-28s %-12s %s" % (label, val, sd))

    # ---------- 报告 ----------
    lines = [
        "# Agent 评测报告",
        "",
        "- 评测时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "- 配置：RAG=%s｜规划=%s｜反思=%s" % (
            "预注入" if mode != "tool" else "工具检索(Agentic RAG)",
            "开" if plan != "off" else "关", reflect if plan != "off" else "关"),
        "- 用例：%d 条 × %d 轮｜用例隔离：%s" % (len(cases), repeat, "开" if isolate else "关"),
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 | 标准差 |",
        "| --- | --- | --- |",
    ]
    for label, val, sd in summary:
        lines.append("| %s | %s | %s |" % (label, val, sd or "-"))

    lines += ["", "## 逐轮明细", "", "| 轮次 | 达成率 | 工具正确率 | 平均延迟 | 平均 token | 质检触发 |", "| --- | --- | --- | --- | --- | --- |"]
    for i, m in enumerate(per_round, 1):
        lines.append("| %d | %.1f%% | %.1f%% | %.2fs | %.0f | %d |"
                     % (i, m["达成率"], m["工具正确率"], m["平均延迟"], m["平均token"], m["质检触发"]))

    if repeat == 1:
        lines += ["", "## 用例明细", "",
                  "| # | 类别 | 输入 | 期望工具 | 实际工具 | 工具正确 | 任务达成 | 质检 | 延迟 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in all_results:
            lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %.2fs |"
                         % (r["id"], r["类别"], r["输入"],
                            ",".join(r["期望工具"]) or "-", ",".join(r["实际工具"]) or "-",
                            "Y" if r["工具正确"] else "N", "Y" if r["任务达成"] else "N",
                            "Y" if r["质检触发"] else "-", r["延迟秒"]))

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print()
    print("报告已写入：%s" % report_file)

    # ---------- 清理评测会话 ----------
    try:
        conn = w._db()
        conn.execute("DELETE FROM messages WHERE who LIKE ?", (EVAL_SESSION + "%",))
        conn.execute("DELETE FROM profiles WHERE who LIKE ?", (EVAL_SESSION + "%",))
        conn.commit()
        conn.close()
        print("已清理评测会话数据（不影响真实记忆）")
    except Exception as e:
        print("清理失败：%s" % e)


if __name__ == "__main__":
    main()
