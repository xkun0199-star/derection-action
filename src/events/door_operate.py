"""
开门/关门（窗）检测：
  标准 COCO 中无"门"类别，通过以下策略检测：
  1. 手腕关键点在图像边缘区域（门通常靠边）做抓握动作
  2. 手腕 x 方向出现较大位移（推/拉动作）
  3. 可选：若场景有自定义门检测模型，可直接替换 _door_regions 逻辑
"""
from events.base import BaseEventDetector
from utils.geometry import kp_xy
from utils.track_history import TrackHistory
import config

_KP = config.KP
_PUSH_PULL_PX = 40   # 手腕横向位移阈值
_WINDOW       = 8
# 门通常位于画面左右边缘区域（比例）
_DOOR_ZONE_X  = 0.25   # 图像左右各25%区域视为门区


class DoorOperateDetector(BaseEventDetector):

    def __init__(self):
        self._lw_hist = TrackHistory(maxlen=_WINDOW + 2)
        self._rw_hist = TrackHistory(maxlen=_WINDOW + 2)

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        H, W = frame.shape[:2]
        door_x_left  = W * _DOOR_ZONE_X
        door_x_right = W * (1 - _DOOR_ZONE_X)
        active_ids = {p["track_id"] for p in pose_list}
        self._lw_hist.clean_stale(active_ids)
        self._rw_hist.clean_stale(active_ids)

        for person in pose_list:
            tid      = person["track_id"]
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]

            lw = kp_xy(kps_xy, kps_conf, _KP["left_wrist"],  0.4)
            rw = kp_xy(kps_xy, kps_conf, _KP["right_wrist"], 0.4)

            self._lw_hist.push(tid, lw)
            self._rw_hist.push(tid, rw)

            for hist, wrist in ((self._lw_hist, lw), (self._rw_hist, rw)):
                buf = [v for v in hist.get(tid) if v is not None]
                if len(buf) < _WINDOW or wrist is None:
                    continue
                # 手腕是否在门区
                wx = wrist[0]
                in_door_zone = wx < door_x_left or wx > door_x_right
                if not in_door_zone:
                    continue
                # 横向位移
                xs = [v[0] for v in buf]
                total_disp = abs(xs[-1] - xs[0])
                if total_disp > _PUSH_PULL_PX:
                    action = "开门" if xs[-1] < xs[0] else "关门"
                    events.append({
                        "name"    : action,
                        "id": tid,
                        "color"   : (128, 0, 255),
                        "detail"  : f"disp={total_disp:.0f}px",
                    })
                    break   # 同一人同一帧只报一次
        return events
