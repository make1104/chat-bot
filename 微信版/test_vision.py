# -*- coding: utf-8 -*-
"""实测 DeepSeek V4.1 Flash 的多模态识图能力（自制一张红色正方形测试图）"""

import base64
import os
import struct
import zlib

from dotenv import load_dotenv

load_dotenv()
KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def make_png(path: str, size: int = 200) -> str:
    """纯 Python 生成一张白底红色正方形的 PNG（不依赖 Pillow）。"""

    def px(x, y):
        if size * 0.3 <= x < size * 0.7 and size * 0.3 <= y < size * 0.7:
            return (220, 30, 30)     # 红色方块
        return (255, 255, 255)       # 白色背景

    raw = b""
    for y in range(size):
        raw += b"\x00"
        for x in range(size):
            raw += bytes(px(x, y))

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def main():
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    img = make_png("vision_test.png")
    print(f"测试图已生成：{img} ({os.path.getsize(img)} bytes)")

    model = ChatOpenAI(model="deepseek-flash", api_key=KEY, base_url="https://api.deepseek.com")

    # 1) 纯文本能力
    r1 = model.invoke([HumanMessage(content="只回复两个字：收到")])
    print("文本能力：", (r1.content or "").strip()[:30])

    # 2) 识图能力（base64 内联）
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r2 = model.invoke([HumanMessage(content=[
        {"type": "text", "text": "这张图里有什么？背景是什么颜色？中间的图形是什么形状、什么颜色？简洁回答。"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])])
    print("识图结果：", (r2.content or "").strip())


if __name__ == "__main__":
    main()
