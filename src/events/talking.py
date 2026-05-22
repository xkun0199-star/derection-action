"""
交谈/倾听检测：
  - 嘴部开合（用 SeetaFace2 81点或5点推算）持续波动 → 交谈
  - 头部微转（注视他人方向）且未说话 → 倾听
  两人以上在同一帧均检测到说话/倾听信号时确认"交谈"事件。
"""
from events.base import BaseEventDetector
from utils.geometry import dist
from utils.track_history import TrackHistory
import config

# 81点中嘴部关键点索引（上唇中点=62，下唇中点=66）
_MOUTH_TOP_IDX    = 51   # 上唇中点
_MOUTH_BOTTOM_IDX = 57   # 下唇中点
_MAR_OPEN_THRESH  = 0.04  # 嘴部纵横比阈值
_TALK_FRAMES      = 5


def _mar_from_pts81(pts81) -> float:
    """嘴部纵横比"""
    top    = pts81[_MOUTH_TOP_IDX]
    bottom = pts81[_MOUTH_BOTTOM_IDX]
    left   = pts81[48]
    right  = pts81[54]
    vdist = dist(top, bottom)
    hdist = dist(left, right)
    return vdist / (hdist + 1e-6)


class TalkingDetector(BaseEventDetector):

    def __init__(self):
        self._mar_history  = TrackHistory(maxlen=_TALK_FRAMES + 3)
        self._mouth_open_count = {}  # tid -> rolling open count

    def detect(self, frame, pose_list, det_list, face_list):
        events  = []
        talking = []
        active_ids = {p["track_id"] for p in pose_list}
        self._mar_history.clean_stale(active_ids)

        for person in pose_list:
            tid  = person["track_id"]
            face = self._match_face(person["bbox"], face_list)
            if face is None:
                continue

            if face["pts81"] and len(face["pts81"]) > 67:
                mar = _mar_from_pts81(face["pts81"])
            else:
                mar = self._mar_from_pts5(face["pts5"])

            is_open = mar > _MAR_OPEN_THRESH
            self._mar_history.push(tid, is_open)
            buf = list(self._mar_history.get(tid))

            # 嘴部开合波动 → 说话
            if len(buf) >= _TALK_FRAMES:
                open_ratio = sum(buf[-_TALK_FRAMES:]) / _TALK_FRAMES
                if 0.3 < open_ratio < 0.9:   # 有开有合，不是一直张嘴
                    talking.append(tid)

        # 2人以上说话 → 交谈；1人说话且附近有人 → 倾听
        if len(talking) >= 2:
            for tid in talking:
                events.append({
                    "name"    : "交谈",
                    "id": tid,
                    "color"   : (200, 255, 0),
                })
        elif len(talking) == 1 and len(pose_list) >= 2:
            events.append({
                "name"    : "交谈/倾听",
                "id": talking[0],
                "color"   : (150, 255, 0),
            })
        return events

    @staticmethod
    def _match_face(person_bbox, face_list):
        px1,py1,px2,py2 = person_bbox
        best, best_score = None, -1
        for f in face_list:
            fx1,fy1,fx2,fy2 = f["bbox"]
            if fx1 >= px1-20 and fx2 <= px2+20 and fy1 >= py1-20:
                if f["score"] > best_score:
                    best, best_score = f, f["score"]
        return best

    @staticmethod
    def _mar_from_pts5(pts5) -> float:
        """5点粗估：嘴部两点为pts5[3], pts5[4]"""
        if len(pts5) < 5:
            return 0.0
        ml, mr = pts5[3], pts5[4]
        # 嘴角间距与整脸宽度比
        return dist(ml, mr) * 0.05   # 粗略映射到 MAR 量纲
