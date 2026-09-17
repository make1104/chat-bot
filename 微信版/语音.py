# -*- coding: utf-8 -*-
"""语音合成模块：支持 edge-tts（默认，轻量免费）与 GPT-SoVITS（音色克隆）

两种后端：
  1) edge = edge-tts —— 微软在线语音，无需 GPU、无需部署，装完即用（推荐先用这个）
  2) gpt_sovits = 本地 GPT-SoVITS 服务 —— 可以克隆任意音色（例如角色原声），
     需要先自行部署 GPT-SoVITS 并启动它的 api 服务（默认 http://127.0.0.1:9880）

用法：
    from 语音 import synthesize
    path = synthesize("你好呀~")     # 返回音频文件路径，失败返回 ""
"""

import asyncio
import os
import time
import urllib.parse
import urllib.request

# ==================== 配置 ====================
TTS_BACKEND = "edge"       # "edge" | "gpt_sovits" | "none"
TTS_VOICE = "zh-CN-XiaoyiNeural"   # edge 音色：
#   zh-CN-XiaoyiNeural   少女音（推荐，活泼）
#   zh-CN-XiaoxiaoNeural 温柔女声
#   zh-CN-YunxiNeural    少年音
#   zh-CN-YunyangNeural  沉稳男声
TTS_RATE = "+8%"           # 语速，如 "+10%" / "-5%"
TTS_VOLUME = "+0%"         # 音量
OUT_DIR = "voice_cache"    # 语音缓存目录（自动创建）

# ---- GPT-SoVITS 配置（TTS_BACKEND = "gpt_sovits" 时生效）----
GS_API = "http://127.0.0.1:9880"                      # GPT-SoVITS api_v2.py 服务地址
GS_REF_AUDIO = r"E:\GPT-SoVITS\参考音频\sample.wav"    # 参考音频路径（3~10 秒，越干净越好）
GS_PROMPT_TEXT = "参考音频里说的那句话"                  # 参考音频对应的文本
GS_PROMPT_LANG = "zh"                                 # 参考音频语言
GS_TEXT_LANG = "zh"                                   # 待合成文本语言


def _edge_tts(text: str, out_path: str) -> bool:
    """用 edge-tts 合成（免费在线，约 1~2 秒）。"""
    import edge_tts

    async def _run():
        await edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME).save(out_path)

    asyncio.run(_run())
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _gpt_sovits(text: str, out_path: str) -> bool:
    """调用本地 GPT-SoVITS 的 /tts 接口合成（音色克隆）。"""
    params = urllib.parse.urlencode({
        "text": text,
        "text_lang": GS_TEXT_LANG,
        "ref_audio_path": GS_REF_AUDIO,
        "prompt_text": GS_PROMPT_TEXT,
        "prompt_lang": GS_PROMPT_LANG,
        "text_split_method": "cut5",
        "media_type": "wav",
    })
    with urllib.request.urlopen(f"{GS_API}/tts?{params}", timeout=180) as resp:
        data = resp.read()
    if not data:
        return False
    with open(out_path, "wb") as f:
        f.write(data)
    return True


def synthesize(text: str) -> str:
    """把文本合成为语音文件，返回文件绝对路径；失败返回空字符串。"""
    text = (text or "").strip()
    if not text or TTS_BACKEND == "none":
        return ""
    os.makedirs(OUT_DIR, exist_ok=True)
    ext = ".mp3" if TTS_BACKEND == "edge" else ".wav"
    out_path = os.path.abspath(os.path.join(OUT_DIR, f"tts_{int(time.time() * 1000)}{ext}"))
    try:
        ok = _edge_tts(text, out_path) if TTS_BACKEND == "edge" else _gpt_sovits(text, out_path)
    except Exception as e:
        print(f"⚠️ 语音合成失败（{TTS_BACKEND}）：{e}")
        return ""
    if not ok:
        print("⚠️ 语音合成失败：没有生成音频文件")
        return ""
    return out_path


def cleanup(keep: int = 30) -> None:
    """只保留最近 keep 个语音文件，其余的删掉（避免缓存目录无限膨胀）。"""
    if not os.path.isdir(OUT_DIR):
        return
    files = sorted(
        (os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.startswith("tts_")),
        key=os.path.getmtime,
        reverse=True,
    )
    for old in files[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass


if __name__ == "__main__":
    # 自测：python 语音.py "你好呀，我是你的机器人"
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "你好呀，我是你的机器人，测试一下语音功能~"
    print("后端：", TTS_BACKEND, "音色：", TTS_VOICE)
    p = synthesize(sample)
    print("生成结果：", p or "失败")
