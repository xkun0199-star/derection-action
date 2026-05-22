"""
招手检测（精细消抖抗误报版）：
1. 空间拦截：手腕高度必须显著高于脖子/肩膀线（过滤普通的垂手摆动）。
2. 幅度防爆：引入横向运动占空比机制。人在抬起手臂时，主要是垂直（Y轴）方向的位移，
   而真正的招手主要是横向（X轴）高频摆动。通过限制幅度和位移比率，彻底拦截“单次抬手”带来的误报。
3. 时序精筛：严格计算速度方向翻转次数（过零点），确保手腕在 X 轴完成了多次完整的周期性往返。
"""
import numpy as np
from events.base import BaseEventDetector
from utils.geometry import kp_xy
from utils.track_history import TrackHistory
import config

_KP    = config.KP
_WIN   = config.WAVE_WINDOW
_SPEED = config.WAVE_WRIST_SPEED


class WavingDetector(BaseEventDetector):

    def __init__(self):
        # 缓存手腕的 (x, y) 完整二维坐标轨迹，限制大小以对齐配置窗口
        self._lwrist_hist = TrackHistory(maxlen=_WIN + 2)
        self._rwrist_hist = TrackHistory(maxlen=_WIN + 2)
        
        # 核心抗误报参数定义
        self._min_flips = 2          # 10帧内至少要有2次方向反转（代表一个半往返周期）
        self._min_x_amplitude = 15.0 # 手腕在 X 轴上总的摆动跨度至少要达到 15 个像素，防止微小抖动误报

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._lwrist_hist.clean_stale(active_ids)
        self._rwrist_hist.clean_stale(active_ids)

        for person in pose_list:
            tid      = person["track_id"]
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]

            # 1. 提取手腕、肩膀和耳朵的骨骼点坐标（YOLOv8-Pose 17点标准）
            lw = kp_xy(kps_xy, kps_conf, _KP["left_wrist"],  conf_thr=0.4)
            rw = kp_xy(kps_xy, kps_conf, _KP["right_wrist"], conf_thr=0.4)
            ls = kp_xy(kps_xy, kps_conf, _KP["left_shoulder"], conf_thr=0.4)
            rs = kp_xy(kps_xy, kps_conf, _KP["right_shoulder"], conf_thr=0.4)

            # 2. 🛡️ 空间第一道防御线：手有没有举起来？
            # 图像坐标系中 Y 越小代表物理高度越高。手腕的 Y 必须小于肩膀的 Y 轴
            lw_raised = (lw and ls and lw[1] < ls[1])
            rw_raised = (rw and rs and rw[1] < rs[1])

            # 将有效的举手坐标压入时序队列；未举手则注入 None，破坏连续性缓存
            self._lwrist_hist.push(tid, lw if lw_raised else None)
            self._rwrist_hist.push(tid, rw if rw_raised else None)

            lbuf = [pt for pt in self._lwrist_hist.get(tid) if pt is not None]
            rbuf = [pt for pt in self._rwrist_hist.get(tid) if pt is not None]

            # 3. 🔬 驱动二级核心抗误报算法
            is_left_wave = self._analyze_waving_trajectory(lbuf)
            is_right_wave = self._analyze_waving_trajectory(rbuf)

            if is_left_wave or is_right_wave:
                events.append({
                    "name"    : "招手",
                    "id": tid,
                    "color"   : (0, 255, 0), # 切换为充满朝气的亮绿色高亮展示
                    "detail"  : f"ID #{tid} 稳定挥手中",
                })
        return events

    def _analyze_waving_trajectory(self, pts_list) -> bool:
        """
        精细运动特征过滤器：区分“单纯抬手臂”与“高频横向招手”
        """
        if len(pts_list) < _WIN:
            return False

        # 提取滑动窗口内的 X 和 Y 轴坐标序列
        xs = np.array([p[0] for p in pts_list])
        ys = np.array([p[1] for p in pts_list])

        # 🚀 策略一：横向总跨度检查（防止就地微弱抖动）
        x_amplitude = np.max(xs) - np.min(xs)
        if x_amplitude < self._min_x_amplitude:
            return False

        # 🚀 策略二：核心拦截线——运动主轴倾角过滤（消灭单纯的“抬起手臂”）
        # 抬起手臂时，Y轴的变化量剧烈，X轴变化量微弱；而招手时，X轴变化极其显著。
        x_delta = np.sum(np.abs(np.diff(xs)))
        y_delta = np.sum(np.abs(np.diff(ys)))
        
        # 如果 Y 轴位移总和远大于 X 轴（即主要是纵向高频位移），判定为抬手、伸懒腰或指点，直接拦截
        if y_delta > x_delta * 1.5:
            return False

        # 🚀 策略三：高频过零点速度反转检测
        diff_xs = np.diff(xs)
        avg_speed = np.mean(np.abs(diff_xs))

        # 计算横向速度方向的翻转频次
        flips = 0
        for i in range(len(diff_xs) - 1):
            # 加小门限 -0.5 彻底消除背景像素因 YOLO 框抖动带来的微弱白噪声
            if diff_xs[i] * diff_xs[i+1] < -0.5:
                flips += 1

        # 终审：只有平均横向速度达标，且完成了至少 2 次折返，才算是真正的招手！
        if avg_speed > _SPEED and flips >= self._min_flips:
            return True

        return False