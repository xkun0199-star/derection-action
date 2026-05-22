"""
摔倒检测：
  1. 躯干（肩-髋连线）与水平线夹角 < FALL_ANGLE_THRESH
  2. 或 bbox 宽高比 > FALL_RATIO_THRESH（人体变"扁"）
  3. 需连续 N 帧触发，避免瞬间姿态误报
"""
from events.base import BaseEventDetector
from utils.geometry import kp_xy, angle_deg
from utils.track_history import TrackHistory
import config

_KP            = config.KP
_ANGLE_THR     = config.FALL_ANGLE_THRESH
_RATIO_THR     = config.FALL_RATIO_THRESH
_CONSEC_FRAMES = 3


class FallDownDetector(BaseEventDetector):

    def __init__(self):
        self._history = TrackHistory(maxlen=_CONSEC_FRAMES + 2)

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._history.clean_stale(active_ids)

        for person in pose_list:
            tid      = person["track_id"]
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]
            x1,y1,x2,y2 = person["bbox"]

            # 条件1：躯干倾角
            trunk_fallen = False
            ls = kp_xy(kps_xy, kps_conf, _KP["left_shoulder"],  0.4)
            rs = kp_xy(kps_xy, kps_conf, _KP["right_shoulder"], 0.4)
            lh = kp_xy(kps_xy, kps_conf, _KP["left_hip"],       0.4)
            rh = kp_xy(kps_xy, kps_conf, _KP["right_hip"],      0.4)

            if ls and rs and lh and rh:
                sho_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
                hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
                angle = angle_deg(sho_mid, hip_mid)   # 与水平轴夹角
                trunk_fallen = angle < _ANGLE_THR

            # 条件2：bbox 宽高比
            bw = x2 - x1
            bh = y2 - y1 + 1e-6
            ratio_fallen = (bw / bh) > _RATIO_THR

            is_fallen = trunk_fallen or ratio_fallen
            self._history.push(tid, is_fallen)
            buf = list(self._history.get(tid))

            if len(buf) >= _CONSEC_FRAMES and all(buf[-_CONSEC_FRAMES:]):
                events.append({
                    "name"    : "摔倒",
                    "id": tid,
                    "color"   : (0, 0, 200),
                    "detail"  : f"angle/ratio",
                })
        return events
