# -*- coding: utf-8 -*-
"""验证图片获取的三条通道与 local_id 解析"""

import sys

sys.path.insert(0, ".")

import wechat_bot as w

print("IMAGE_GUI_FALLBACK =", w.IMAGE_GUI_FALLBACK)
print()
print("函数就绪检查：")
for name in ("_msg_local_id", "_downloader", "download_message_media",
             "download_image_original", "get_message_image", "describe_image"):
    print(f"  {name:26} -> {callable(getattr(w, name, None))}")


class Msg:
    def __init__(self, mtype, lid):
        self.type = mtype
        self.local_id = lid


print()
print("local_id 解析：")
for lid in ("db-123", "456", 789, "", None):
    m = Msg("image", lid)
    print(f"  原始 {str(lid)!r:10} -> {w._msg_local_id(m)}")

print()
print("说明：")
print("  图片(image)     -> download_message_media() 解密缓存；失败则 download_image_original() 点图触发下载")
print("  表情包(emotion) -> msg.capture() 屏幕截图")
print("（本条测试不发真实下载请求，避免干扰正在运行的微信窗口）")
