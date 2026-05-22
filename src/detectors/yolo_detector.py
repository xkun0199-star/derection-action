"""YOLO 封装：目标检测 + 姿态估计（带 ByteTrack 追踪）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ultralytics import YOLO
import config


class YOLODetector:
    """
    封装 YOLOv8 目标检测与姿态估计。
    两个模型均带 ByteTrack，保证跨帧 track_id 稳定。
    detect() / pose() 均返回标准化列表，下游无需感知 ultralytics 数据结构。
    """

    def __init__(self):
        self.det_model  = YOLO(config.YOLO_DET_MODEL)
        self.pose_model = YOLO(config.YOLO_POSE_MODEL)

    # ── 目标检测 ──────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        返回目标列表：
        {
            "track_id": int,
            "cls":      int,
            "conf":     float,
            "bbox":     (x1,y1,x2,y2),
            "label":    str,
        }
        """
        results = self.det_model.track(
            frame,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            persist=True,
            verbose=False,
        )[0]
        return self._parse_boxes(results)

    # ── 姿态估计 ──────────────────────────────────────────
    def pose(self, frame: np.ndarray) -> list[dict]:
        """
        返回人体列表：
        {
            "track_id": int,
            "conf":     float,
            "bbox":     (x1,y1,x2,y2),
            "kps_xy":   np.ndarray shape(17,2),
            "kps_conf": np.ndarray shape(17,),
        }
        """
        results = self.pose_model.track(
            frame,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            persist=True,
            verbose=False,
        )[0]
        return self._parse_pose(results)

    # ── 私有解析 ──────────────────────────────────────────
    @staticmethod
    def _parse_boxes(result) -> list[dict]:
        out = []
        if result.boxes is None:
            return out
        ids = result.boxes.id
        for i, box in enumerate(result.boxes):
            track_id = int(ids[i]) if ids is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({
                "track_id": track_id,
                "cls"     : int(box.cls[0]),
                "conf"    : float(box.conf[0]),
                "bbox"    : (x1, y1, x2, y2),
                "label"   : result.names[int(box.cls[0])],
            })
        return out

    @staticmethod
    def _parse_pose(result) -> list[dict]:
        out = []
        if result.boxes is None or result.keypoints is None:
            return out
        ids = result.boxes.id
        for i, box in enumerate(result.boxes):
            track_id = int(ids[i]) if ids is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            kps = result.keypoints[i]
            kps_xy   = kps.xy[0].cpu().numpy()    # (17,2)
            kps_conf = kps.conf[0].cpu().numpy()   # (17,)
            out.append({
                "track_id": track_id,
                "conf"    : float(box.conf[0]),
                "bbox"    : (x1, y1, x2, y2),
                "kps_xy"  : kps_xy,
                "kps_conf": kps_conf,
            })
        return out
