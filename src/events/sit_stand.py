"""
就座/离座检测：
  追踪臀部关键点的绝对高度（y坐标）在连续帧的变化：
  - y 坐标增大（下移）超过阈值后稳定 → 就座
  - y 坐标减小（上移）超过阈值后稳定 → 离座
  稳定条件：变化后连续 STABLE_FRAMES 帧高度变化 < STABLE_PX
"""
from events.base import BaseEventDetector
from utils.geometry import kp_xy
from utils.track_history import TrackHistory
import config

_KP           = config.KP
_CHANGE_PX    = 50    # 臀部需要移动的像素距离
_STABLE_FRAMES= 4     # 稳定确认帧数
_STABLE_PX    = 15    # 稳定判定阈值
_WINDOW       = 20


class SitStandDetector(BaseEventDetector):

    def __init__(self):
        self._hip_y_hist = TrackHistory(maxlen=_WINDOW)
        self._last_event: dict[int, str] = {}

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._hip_y_hist.clean_stale(active_ids)

        for person in pose_list:
            tid      = person["track_id"]
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]

            lh = kp_xy(kps_xy, kps_conf, _KP["left_hip"],  0.4)
            rh = kp_xy(kps_xy, kps_conf, _KP["right_hip"], 0.4)
            if lh is None and rh is None:
                continue
            pts = [p for p in (lh, rh) if p is not None]
            hip_y = sum(p[1] for p in pts) / len(pts)

            self._hip_y_hist.push(tid, hip_y)
            buf = list(self._hip_y_hist.get(tid))
            if len(buf) < _STABLE_FRAMES + 2:
                continue

            recent   = buf[-_STABLE_FRAMES:]
            stable   = max(recent) - min(recent) < _STABLE_PX
            baseline = buf[-(2 + _STABLE_FRAMES)]
            delta    = recent[-1] - baseline

            if not stable:
                continue

            if delta > _CHANGE_PX:
                ev_name = "就座"
            elif delta < -_CHANGE_PX:
                ev_name = "离座"
            else:
                continue

            # 去重：同一 track_id 不连续报相同事件
            if self._last_event.get(tid) == ev_name:
                continue
            self._last_event[tid] = ev_name
            events.append({
                "name"    : ev_name,
                "id": tid,
                "color"   : (0, 200, 200),
                "detail"  : f"Δy={delta:.0f}px",
            })
        return events
