"""SeetaFace2 封装：人脸检测 + 5点关键点 + 81点关键点"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SeetaFacePy
import config


class SeetaFaceDetector:
    """
    封装 SeetaFace2 的人脸检测与关键点定位。
    detect() 返回标准化的列表，下游无需感知底层 API。
    """

    def __init__(self):
        det_cfg = SeetaFacePy.ModelSetting(
            config.SEETA_DET_MODEL, SeetaFacePy.ModelSetting.CPU
        )
        self.detector = SeetaFacePy.FaceDetector(det_cfg)
        self.detector.set(SeetaFacePy.FaceDetector.Property.MIN_FACE_SIZE,
                          config.SEETA_MIN_FACE)
        self.detector.set(SeetaFacePy.FaceDetector.Property.THRESHOLD1,
                          config.SEETA_THRESHOLD)

        land5_cfg = SeetaFacePy.ModelSetting(
            config.SEETA_LAND5_MODEL, SeetaFacePy.ModelSetting.CPU
        )
        self.landmarker5 = SeetaFacePy.FaceLandmarker(land5_cfg)

        # 81点模型（可选，用于精细眼部EAR计算）
        self.landmarker81 = None
        if os.path.exists(config.SEETA_LAND81_MODEL):
            land81_cfg = SeetaFacePy.ModelSetting(
                config.SEETA_LAND81_MODEL, SeetaFacePy.ModelSetting.CPU
            )
            self.landmarker81 = SeetaFacePy.FaceLandmarker(land81_cfg)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        返回人脸列表，每个元素：
        {
            "bbox":  (x1,y1,x2,y2),
            "score": float,
            "pts5":  [(x,y), ...] × 5,   # 双眼/鼻/嘴角
            "pts81": [(x,y), ...] × 81 or None
        }
        """
        raw_faces = self.detector.detect(frame)
        results = []
        for f in raw_faces:
            pts5 = [(pt.x, pt.y) for pt in self.landmarker5.mark(frame, f)]
            pts81 = None
            if self.landmarker81 is not None:
                pts81 = [(pt.x, pt.y) for pt in self.landmarker81.mark(frame, f)]
            results.append({
                "bbox" : (f.x, f.y, f.x + f.width, f.y + f.height),
                "score": float(f.score),
                "pts5" : pts5,
                "pts81": pts81,
            })
        return results
