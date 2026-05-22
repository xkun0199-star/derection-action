"""
火灾/烟雾检测：
  1. 颜色阈值法：HSV 空间检测火焰橙红色区域
     - 需同时满足：色相在火焰范围 + 高饱和度 + 高亮度
     - 用梯度/纹理排除皮肤等纯色区域（火焰边缘变化丰富）
  2. 烟雾：局部灰度方差低 + 亮度适中（排除纯白/纯黑）
"""
import cv2
import numpy as np
from events.base import BaseEventDetector
import config

# 火焰 HSV 范围（严格高饱和高亮的橙红）
_FIRE_H_LO, _FIRE_H_HI = 0,   20
_FIRE_S_LO, _FIRE_S_HI = 180, 255   # 饱和度要求更高，排除皮肤
_FIRE_V_LO, _FIRE_V_HI = 200, 255   # 亮度要求更高

# 火焰纹理：Laplacian 方差需超过此值（皮肤平滑，火焰边缘凌乱）
_FIRE_TEXTURE_THR = 120.0

# 烟雾：局部区域灰度方差偏低且亮度在合理范围（不是纯白背景）
_SMOKE_VAR_THR   = 300
_SMOKE_MEAN_LO   = 130    # 排除过暗区域
_SMOKE_MEAN_HI   = 220    # 排除纯白背景

_FIRE_MIN_PIXELS  = 800   # 提高最小面积门槛
_MIN_BLOCK_RATIO  = 0.015 # 占整帧面积的最小比例（1.5%）


class FireSmokeDetector(BaseEventDetector):

    def detect(self, frame, pose_list, det_list, face_list):
        events = []

        fire_ev  = self._detect_fire(frame)
        smoke_ev = self._detect_smoke(frame)

        if fire_ev:
            events.append(fire_ev)
        if smoke_ev and not fire_ev:
            events.append(smoke_ev)
        return events

    @staticmethod
    def _detect_fire(frame) -> dict | None:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           (_FIRE_H_LO, _FIRE_S_LO, _FIRE_V_LO),
                           (_FIRE_H_HI, _FIRE_S_HI, _FIRE_V_HI))

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        H, W   = frame.shape[:2]
        min_px = max(_FIRE_MIN_PIXELS, int(H * W * _MIN_BLOCK_RATIO))
        if cv2.countNonZero(mask) < min_px:
            return None

        # 纹理验证：火焰区域 Laplacian 方差应较高
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap       = cv2.Laplacian(gray, cv2.CV_64F)
        roi_vals  = lap[mask > 0]
        if roi_vals.size == 0 or float(np.var(roi_vals)) < _FIRE_TEXTURE_THR:
            return None

        return {
            "name"    : "火灾",
            "id": None,
            "color"   : (0, 0, 255),
        }

    @staticmethod
    def _detect_smoke(frame) -> dict | None:
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W      = gray.shape
        bh, bw    = H // 4, W // 4
        smoke_cnt = 0
        for r in range(4):
            for c in range(4):
                block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                var   = float(np.var(block))
                mean  = float(np.mean(block))
                if var < _SMOKE_VAR_THR and _SMOKE_MEAN_LO < mean < _SMOKE_MEAN_HI:
                    smoke_cnt += 1
        if smoke_cnt >= 6:    # 提高门槛：至少6/16个块（约37%画面）
            return {
                "name"    : "烟雾",
                "id": None,
                "color"   : (128, 128, 128),
            }
        return None
