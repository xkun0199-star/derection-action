"""事件检测基类"""
from abc import ABC, abstractmethod
import numpy as np


class BaseEventDetector(ABC):
    """
    所有事件检测器的公共接口。
    子类实现 detect()，返回本帧触发的事件列表。
    每个事件是一个 dict：
    {
        "name"    : str,          # 事件名
        "id"      : int | None,   # 关联的人/目标 ID（YOLO track_id）
        "color"   : (B,G,R),      # 显示颜色
        "detail"  : str,          # 可选，调试信息
    }
    """

    @abstractmethod
    def detect(self, frame: np.ndarray, pose_list: list, det_list: list,
               face_list: list) -> list[dict]:
        """
        frame      : 当前帧 BGR
        pose_list  : YOLODetector.pose() 输出
        det_list   : YOLODetector.detect() 输出
        face_list  : SeetaFaceDetector.detect() 输出
        """
        ...
