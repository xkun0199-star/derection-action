"""
注视检测：用 solvePnP 估算头部三维朝向（Yaw/Pitch），
当 |Yaw| < YAW_THRESH 且 |Pitch| < PITCH_THRESH 连续 N 帧 → 注视相机。

坐标轴定义（OpenCV相机坐标系，Y朝下）：
  Yaw   (偏航)：绕Y轴旋转 → 左右转头，>0 向右
  Pitch (俯仰)：绕X轴旋转 → 上下点头，>0 抬头
  Roll  (横滚)：绕Z轴旋转 → 头部侧倾，不参与判断
"""
import numpy as np
import cv2
from events.base import BaseEventDetector
from utils.track_history import TrackHistory
import config

# ── 6点3D人脸模型（OpenFace标准，毫米，Y轴朝上数学坐标，鼻尖为原点）─────────────
# 对应 SeetaFace2 pts81 索引：鼻尖30、下巴8、左眼外角36、右眼外角45、左嘴角48、右嘴角54
DIFF_PITCH = 106.0  # 可选：调整基准线，使正常注视时更接近0度
_IDX_6 = (30, 8, 36, 45, 48, 54)
_MODEL_6PTS = np.array([
    (  0.0,    0.0,    0.0),  # 鼻尖     30
    (  0.0,  -63.6,  -12.5),  # 下巴     8
    (-43.3,   32.7,  -26.0),  # 左眼外角 36
    ( 43.3,   32.7,  -26.0),  # 右眼外角 45
    (-28.9,  -28.9,  -24.1),  # 左嘴角   48
    ( 28.9,  -28.9,  -24.1),  # 右嘴角   54
], dtype=np.float64)

# 5点退化模型（pts5：左眼、右眼、鼻尖、左嘴角、右嘴角）
_MODEL_5PTS = np.array([
    (-43.3,   32.7,  -26.0),  # 左眼
    ( 43.3,   32.7,  -26.0),  # 右眼
    (  0.0,    0.0,    0.0),  # 鼻尖
    (-28.9,  -28.9,  -24.1),  # 左嘴角
    ( 28.9,  -28.9,  -24.1),  # 右嘴角
], dtype=np.float64)

YAW_THRESH   = 20.0  # 偏航角阈值（度）
PITCH_THRESH = 20.0  # 俯仰角阈值（度）
_GAZE_FRAMES = 6     # 连续满足条件的推理帧数（0.3s×6=1.8s）


def _build_cam_matrix(frame_shape):
    h, w = frame_shape[:2]
    focal = w  # 无标定时的近似焦距
    return np.array([
        [focal,     0, w / 2],
        [    0, focal, h / 2],
        [    0,     0,     1],
    ], dtype=np.float64)


def _rotation_to_euler(rmat):
    """
    从旋转矩阵提取 Yaw/Pitch/Roll（ZYX顺序：R = Rz*Ry*Rx）。
    返回 (yaw_deg, pitch_deg, roll_deg)。
    Yaw  = 绕Y旋转（左右）
    Pitch= 绕X旋转（上下）
    Roll = 绕Z旋转（侧倾）
    """
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        yaw   = np.degrees(np.arctan2(-rmat[2, 0],  sy))
        pitch = np.degrees(np.arctan2( rmat[2, 1],  rmat[2, 2]))
        roll  = np.degrees(np.arctan2( rmat[1, 0],  rmat[0, 0]))
    else:  # 万向节锁，罕见
        yaw   = np.degrees(np.arctan2(-rmat[2, 0],  sy))
        pitch = np.degrees(np.arctan2(-rmat[1, 2],  rmat[1, 1]))
        roll  = 0.0
    return yaw, pitch, roll


def _estimate_head_pose(face, frame_shape):
    """
    优先使用 pts81 中的6个稳定特征点；无81点时退回 pts5。
    返回 (yaw_deg, pitch_deg) 或 None。
    """
    cam = _build_cam_matrix(frame_shape)

    if face.get("pts81") and len(face["pts81"]) > max(_IDX_6):
        pts81 = face["pts81"]
        img_pts = np.array([pts81[i] for i in _IDX_6], dtype=np.float64)
        model_pts = _MODEL_6PTS
    elif face.get("pts5") and len(face["pts5"]) >= 5:
        img_pts   = np.array(face["pts5"][:5], dtype=np.float64)
        model_pts = _MODEL_5PTS
    else:
        return None

    ok, rvec, _ = cv2.solvePnP(
        model_pts, img_pts, cam, None,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, _ = _rotation_to_euler(rmat)
    pitch = pitch - DIFF_PITCH  # 可选：调整基准线，使正常注视时更接近0
    return yaw, pitch


class GazingDetector(BaseEventDetector):

    def __init__(self):
        self._history = TrackHistory(maxlen=_GAZE_FRAMES + 2)

    def detect(self, frame, pose_list, det_list, face_list):
        events = []
        active_ids = {p["track_id"] for p in pose_list}
        self._history.clean_stale(active_ids)

        for person in pose_list:
            tid  = person["track_id"]
            face = self._match_face(person["bbox"], face_list)
            if face is None:
                self._history.push(tid, False)
                continue

            result = _estimate_head_pose(face, frame.shape)
            if result is None:
                self._history.push(tid, False)
                continue

            yaw, pitch = result
            print(f"Track {tid}: yaw={yaw:.1f} pitch={pitch:.1f}")
            is_gazing  = (abs(yaw) < YAW_THRESH and abs(pitch) < PITCH_THRESH)
            self._history.push(tid, is_gazing)
            buf = list(self._history.get(tid))

            if len(buf) >= _GAZE_FRAMES and sum(buf[-_GAZE_FRAMES:]) >= _GAZE_FRAMES:
                events.append({
                    "name"  : "注视",
                    "id"    : tid,
                    "color" : (255, 200, 0),
                    "detail": f"yaw={yaw:.1f} pitch={pitch:.1f}",
                })
        return events

    @staticmethod
    def _match_face(person_bbox, face_list):
        px1, py1, px2, py2 = person_bbox
        best, best_score = None, -1
        for f in face_list:
            fx1, fy1, fx2, fy2 = f["bbox"]
            if fx1 >= px1 - 20 and fx2 <= px2 + 20 and fy1 >= py1 - 20:
                if f["score"] > best_score:
                    best, best_score = f, f["score"]
        return best
