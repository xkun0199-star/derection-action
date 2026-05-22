"""
眼部状态检测：闭眼 / 打哈欠。
优先使用 SeetaFace2 81点关键点计算精确 EAR；
没有81点时退化到 5点版本估算。
（眨眼持续约150ms，0.3s推理间隔无法可靠采样，已移除）
"""
import math
from collections import deque
from events.base import BaseEventDetector
from utils.geometry import dist
from utils.track_history import TrackHistory
import config

# 81点模型中眼部关键点索引（SeetaFace2 pt81 layout）
# 左眼：上眼睑中点=37，下眼睑中点=41，左眼角=36，右眼角=39
# 右眼：上眼睑中点=44，下眼睑中点=46，左眼角=42，右眼角=45
_LEFT_EAR_IDX  = (36, 37, 38, 39, 40, 41)  # 6点EAR
_RIGHT_EAR_IDX = (42, 43, 44, 45, 46, 47)

_BLINK_FRAMES = 2   # 快速眨眼帧数（保留常量但不再使用）
_CLOSE_FRAMES = 3   # 持续闭眼帧数（0.3s×3 = 0.9s）
_YAWN_FRAMES  = 5   # 打哈欠持续帧数（0.3s×5 = 1.5s）


def _ear6(pts, idxs) -> float:
    """6点眼纵横比"""
    p = [pts[i] for i in idxs]
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h  = dist(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def _ear_from_pts5(pts5) -> float:
    """用5点粗估EAR：仅区分睁/闭眼趋势"""
    if len(pts5) < 3:
        return 1.0
    leye, reye = pts5[0], pts5[1]
    nose       = pts5[2]
    eye_dist = dist(leye, reye)
    # 粗估：鼻到眼线距离 / 眼间距 越小 → 越可能闭眼
    eye_mid = ((leye[0]+reye[0])/2, (leye[1]+reye[1])/2)
    vdist = abs(nose[1] - eye_mid[1])
    return min(vdist / (eye_dist + 1e-6), 1.0)


class EyeStateDetector(BaseEventDetector):

    def __init__(self):
        self._ear_history = TrackHistory(maxlen=20)

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._ear_history.clean_stale(active_ids)

        for person in pose_list:
            tid = person["track_id"]
            face = self._match_face(person["bbox"], face_list)
            if face is None:
                continue

            if face["pts81"]:
                ear = (_ear6(face["pts81"], _LEFT_EAR_IDX)
                       + _ear6(face["pts81"], _RIGHT_EAR_IDX)) / 2
            else:
                ear = _ear_from_pts5(face["pts5"])

            self._ear_history.push(tid, ear)
            buf = list(self._ear_history.get(tid))

            ev = self._classify(buf, ear)
            if ev:
                events.append({
                    "name"    : ev,
                    "id": tid,
                    "color"   : (0, 0, 255),
                    "detail"  : f"EAR={ear:.3f}",
                })
        return events

    @staticmethod
    def _match_face(person_bbox, face_list):
        """在人体框内找置信度最高的人脸"""
        px1,py1,px2,py2 = person_bbox
        best, best_score = None, -1
        for f in face_list:
            fx1,fy1,fx2,fy2 = f["bbox"]
            if fx1 >= px1-20 and fx2 <= px2+20 and fy1 >= py1-20:
                if f["score"] > best_score:
                    best, best_score = f, f["score"]
        return best

    @staticmethod
    def _classify(buf, cur_ear) -> str | None:
        thr = config.EAR_CLOSE_THRESH
        if len(buf) < 3:
            return None

        closed = [e < thr for e in buf]
        consec = sum(closed[-_CLOSE_FRAMES:])

        if consec >= _YAWN_FRAMES:
            return "打哈欠"
        if consec >= _CLOSE_FRAMES:
            return "闭眼"
        return None
