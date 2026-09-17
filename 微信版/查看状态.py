# -*- coding: utf-8 -*-
"""查看微信机器人的记忆与情绪状态（读取 memory.db，不需要 API Key）

用法：
    python 查看状态.py            # 列出所有人的状态
    python 查看状态.py 小明        # 只看某个人的（支持 wxid 或昵称关键字）
    python 查看状态.py --facts    # 顺便列出每个人的画像事实
"""

import json
import os
import sqlite3
import sys
import time

DB_FILE = "memory.db"


def _encodable(ch: str) -> bool:
    """当前控制台能否显示这个字符（GBK 控制台下部分方块字符不可用）。"""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        ch.encode(enc)
        return True
    except Exception:
        return False


# 兜底：万一还有字符编不出来，替换而不是崩溃
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def _pick_bar_chars():
    for full, empty in (("█", "░"), ("■", "□"), ("#", "-")):
        if _encodable(full) and _encodable(empty):
            return full, empty
    return "#", "-"


FULL, EMPTY = _pick_bar_chars()
LINE1, LINE2 = "=", "-"


def mood_label(score: int) -> str:
    if score >= 5:
        return "心情超好、兴致很高"
    if score >= 2:
        return "心情不错"
    if score >= -1:
        return "平静"
    if score >= -4:
        return "有点不耐烦"
    return "明显生气了"


def relation_label(affinity: int) -> str:
    if affinity >= 80:
        return "非常亲密的老朋友"
    if affinity >= 50:
        return "关系不错的熟人"
    if affinity >= 20:
        return "聊得来的普通朋友"
    if affinity >= 1:
        return "刚认识不久"
    return "完全的陌生人"


def bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """把数值画成进度条。"""
    ratio = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(ratio * width))
    return FULL * filled + EMPTY * (width - filled)


def ago(ts) -> str:
    if not ts:
        return "未知"
    delta = max(0, int(time.time() - ts))
    if delta < 60:
        return f"{delta} 秒前"
    if delta < 3600:
        return f"{delta // 60} 分钟前"
    if delta < 86400:
        return f"{delta // 3600} 小时前"
    return f"{delta // 86400} 天前"


def main():
    if not os.path.exists(DB_FILE):
        print(f"还没找到 {DB_FILE} —— 机器人还没开始记东西呢（先运行一次 wechat_bot.py）")
        return

    keyword = ""
    show_facts = "--facts" in sys.argv or "-f" in sys.argv
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            keyword = a
            break

    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT who, affinity, mood_score, energy, msg_count, first_seen, last_seen, facts "
        "FROM profiles ORDER BY affinity DESC, msg_count DESC"
    ).fetchall()
    msg_rows = dict(conn.execute("SELECT who, COUNT(*) FROM messages GROUP BY who").fetchall())
    conn.close()

    if keyword:
        rows = [r for r in rows if keyword.lower() in str(r[0]).lower()]
    if not rows:
        print("没有匹配的记录。提示：who 是联系人的 username（群聊是 xxx@chatroom）")
        return

    print(LINE1 * 56)
    print("  微信机器人 · 记忆与状态总览")
    print(LINE1 * 56)

    for who, affinity, mood_score, energy, msg_count, first_seen, last_seen, facts_raw in rows:
        affinity = affinity or 0
        mood_score = mood_score or 0
        energy = 100 if energy is None else energy
        print(f"\n【{who}】  最近互动：{ago(last_seen)}")
        print(f"  好感度  {bar(affinity, 0, 100)}  {affinity:3d}/100  {relation_label(affinity)}")
        print(f"  心情    {bar(mood_score, -10, 10)}  {mood_score:+3d}      {mood_label(mood_score)}")
        print(f"  精力    {bar(energy, 0, 100)}  {energy:3d}/100")
        print(f"  消息数  {msg_count or 0} 条（数据库共 {msg_rows.get(who, 0)} 条，首次互动：{ago(first_seen)}）")

        if show_facts:
            try:
                facts = json.loads(facts_raw or "[]")
            except Exception:
                facts = []
            if facts:
                print("  画像：")
                for f in facts:
                    print(f"        - {f}")
            else:
                print("  画像：（还没抽取出长期事实）")

    print("\n" + LINE2 * 56)
    print("提示：加 --facts 可显示画像事实；带名字参数可只看某人")


if __name__ == "__main__":
    main()
