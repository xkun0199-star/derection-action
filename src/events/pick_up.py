"""
拿起物品检测：
  手腕关键点与 YOLO 检测到的可拾取物体（瓶子/杯子/书等）距离
  小于阈值，且手腕高度明显上升（物品被抬起）。
"""
import numpy as np
from events.base import BaseEventDetector
from utils.geometry import kp_xy, dist
from utils.track_history import TrackHistory
import config

_KP = config.KP

# 可拾取物体 COCO 类别
_PICKABLE_CLASSES = {
    39,   # bottle
    41,   # cup
    42,   # fork
    43,   # knife
    44,   # spoon
    45,   # bowl
    46,   # banana
    47,   # apple
    63,   # laptop
    64,   # mouse
    67,   # cell phone
    73,   # book
    76,   # scissors
    77,   # teddy bear
}
_PROXIMITY_PX  = 80   # 手腕到物体中心最大距离
_LIFT_PX       = 20   # y方向上移幅度（像素）
_WINDOW        = 6


class PickUpDetector(BaseEventDetector):

    def __init__(self):
        self._wrist_y_hist = TrackHistory(maxlen=_WINDOW + 2)

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._wrist_y_hist.clean_stale(active_ids)

        pickable = [d for d in det_list if d["cls"] in _PICKABLE_CLASSES]
        if not pickable:
            return events

        for person in pose_list:
            tid      = person["track_id"]
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]

            lw = kp_xy(kps_xy, kps_conf, _KP["left_wrist"],  0.4)
            rw = kp_xy(kps_xy, kps_conf, _KP["right_wrist"], 0.4)
            wrists = [w for w in (lw, rw) if w is not None]
            if not wrists:
                continue

            # 最近手腕 y 坐标均值
            avg_wy = sum(w[1] for w in wrists) / len(wrists)
            self._wrist_y_hist.push(tid, avg_wy)
            buf = list(self._wrist_y_hist.get(tid))

            # 手腕上移
            if len(buf) >= _WINDOW:
                rising = buf[0] - buf[-1] > _LIFT_PX   # y 减小 = 上移
            else:
                rising = False

            # 手腕与可拾取物距离
            near_obj = None
            for obj in pickable:
                cx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                cy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                if any(dist(w, (cx, cy)) < _PROXIMITY_PX for w in wrists):
                    near_obj = obj
                    break

            if near_obj and rising:
                events.append({
                    "name"    : "拿起物品",
                    "id": tid,
                    "color"   : (0, 255, 128),
                    "detail"  : near_obj["label"],
                })
        return events
