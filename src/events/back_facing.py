"""
背身检测：鼻子/双眼关键点置信度普遍低，而肩膀可见 → 判定为背对镜头。
同时融合 SeetaFace2：若 YOLO 检测到人但 SeetaFace2 未检测到人脸，置信度更高。
"""
import numpy as np
from events.base import BaseEventDetector
from utils.geometry import kp_xy
import config

_KP = config.KP


class BackFacingDetector(BaseEventDetector):

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        face_boxes = [f["bbox"] for f in face_list]

        for person in pose_list:
            kps_xy   = person["kps_xy"]
            kps_conf = person["kps_conf"]

            nose_vis  = kps_conf[_KP["nose"]]
            leye_vis  = kps_conf[_KP["left_eye"]]
            reye_vis  = kps_conf[_KP["right_eye"]]
            lsho_vis  = kps_conf[_KP["left_shoulder"]]
            rsho_vis  = kps_conf[_KP["right_shoulder"]]

            # 肩膀可见、面部关键点不可见
            face_invisible = (nose_vis < 0.3 and leye_vis < 0.3 and reye_vis < 0.3)
            shoulders_visible = (lsho_vis > 0.5 and rsho_vis > 0.5)

            if not (face_invisible and shoulders_visible):
                continue

            # 用 SeetaFace2 结果二次验证：bbox 区域内无人脸
            px1,py1,px2,py2 = person["bbox"]
            seeta_confirmed = not any(
                _box_overlap((px1,py1,px2,py2), fb) for fb in face_boxes
            )

            score = (lsho_vis + rsho_vis) / 2 * (1.2 if seeta_confirmed else 1.0)
            if score >= config.THRESHOLDS["back_facing"]:
                events.append({
                    "name"    : "背身",
                    "id": person["track_id"],
                    "color"   : (0, 165, 255),
                    "detail"  : f"score={score:.2f}",
                })
        return events


def _box_overlap(boxA, boxB, iou_thr=0.1) -> bool:
    from utils.geometry import iou
    return iou(boxA, boxB) > iou_thr
