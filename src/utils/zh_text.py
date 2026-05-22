"""
中文文字渲染：用 Pillow + NotoSansCJK 绕过 OpenCV 不支持 Unicode 的限制。
模块加载时缓存字体对象，避免每帧重复 IO。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONT_CACHE[size]


def put_text_zh(frame: np.ndarray, text: str, pos: tuple[int, int],
                color: tuple = (0, 255, 0), font_size: int = 20) -> np.ndarray:
    """
    在 BGR numpy 帧上绘制中文（或任意 Unicode）文字。
    直接修改传入的 frame 并返回。
    color 格式为 BGR，与 OpenCV 一致。
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    font    = _get_font(font_size)
    # Pillow 颜色为 RGB
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame
