"""公共工具函数"""
import math
import numpy as np
import cv2
from typing import Optional, Tuple


def dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angle_deg(p1, p2) -> float:
    """两点连线与水平轴的夹角（度）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))


def iou(boxA, boxB) -> float:
    """xyxy 格式的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def box_center(box) -> Tuple[float, float]:
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def kp_xy(kps_xy: np.ndarray, kps_conf: np.ndarray, idx: int, conf_thr=0.3) \
        -> Optional[Tuple[float, float]]:
    """安全取关键点坐标，置信度不足返回 None"""
    if kps_conf is None or kps_conf[idx] < conf_thr:
        return None
    return (float(kps_xy[idx, 0]), float(kps_xy[idx, 1]))


def draw_label(frame: np.ndarray, text: str, pos: Tuple[int, int],
               color=(0, 255, 0), font_size: int = 20):
    from utils.zh_text import put_text_zh
    put_text_zh(frame, text, pos, color=color, font_size=font_size)


def draw_event_overlay(frame: np.ndarray, events: list[dict]):
    """在左上角叠加当前帧所有激活事件（支持中文）"""
    from utils.zh_text import put_text_zh
    y = 8
    for ev in events:
        color = ev.get("color", (0, 200, 255))
        label = f"[{ev['id']}] {ev['name']}" if ev.get("id") is not None \
                else ev["name"]
        put_text_zh(frame, label, (10, y), color=color, font_size=20)
        y += 26
