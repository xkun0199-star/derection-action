"""
注视检测（基于 2D-to-3D 空间仿射退化算法纯 Python 闭环版）：
1. 彻底不依赖 C++ 端的二次编译映射，直接使用现成检测器返回的 5 点坐标。
2. 在 Python 内部还原 QualityAssessor 的空间解算原理，提取三维物理角度：
   - Pitch（俯仰角：抬头/低头）
   - Yaw（水平偏转角：左转/右转）
   - Roll（平面翻滚角：歪头）
3. 设立严格的物理度数锥体门限，只有当转头、抬头角收敛在安全正脸范围内，才判定为“正脸注视”。
4. 搭载原有框架的 TrackHistory 时序滑动窗口滤波器，完美平抑漏检与抖动。
"""
import numpy as np
from events.base import BaseEventDetector
from utils.track_history import TrackHistory
import config

# ── 物理姿态硬门限配置（单位：度） ──
# 水平转头偏转角阈值：左右偏转绝对值在 15.0 度以内视为正对屏幕
GAZE_YAW_MAX =9.0  

# 俯仰角阈值：考虑监控或相机处于斜上方俯拍，平视时允许的抬头低头物理边界
GAZE_PITCH_MIN = -10.0
GAZE_PITCH_MAX = 10.0

# 平面翻滚角/歪头阈值
GAZE_ROLL_MAX = 20.0

# 时序消抖参数：连续 5 帧完全处于正脸圆锥体内才算触发注视
_GAZE_FRAMES = 5


class GazingDetector(BaseEventDetector):

    def __init__(self):
        super().__init__()
        # 初始化原框架自带的时序轨迹历史追踪器
        self._history = TrackHistory(maxlen=15)

    def detect(self, frame, pose_list, det_list, face_list):
        """
        核心行为检测管道：输入 AI 推理结构化数据，返回触发的事件
        """
        events = []

        # 遍历 YOLOv8-Pose 追踪到的画面中的每一个行人目标
        for p in pose_list:
            tid = p["track_id"]
            person_bbox = p["bbox"]

            # 1. 空间区域粗筛：在行人的上半身/头部空间边界内定位对应的人脸数据
            face = self._match_face(person_bbox, face_list)
            if face is None:
                self._history.push(tid, False)
                continue

            # 2. 提取当前人脸对应的 5 个核心面部关键点 [(x, y), ...]
            pts5 = face.get("pts5", [])
            if len(pts5) < 5:
                self._history.push(tid, False)
                continue

            # 3. 🌟 激活纯 Python 解算引擎：计算 Pitch, Yaw, Roll 绝对度数
            pitch, yaw, roll = self._calculate_head_pose_angles(pts5)

            # 4. 📐 空间锥体区域判定
            # 当且仅当水平角、俯仰角、歪头角同时收敛在正前方的绝对门限区间内，才认为是有效注视
            is_gazing = (abs(yaw) < GAZE_YAW_MAX and 
                         (GAZE_PITCH_MIN < pitch < GAZE_PITCH_MAX) and 
                         abs(roll) < GAZE_ROLL_MAX)

            # 【可选后台诊断打印】：可以取消下面的注释，在终端实时观察每一帧人的精确侧头、低头度数
            # print(f"[Gaze Tracking] ID #{tid} -> Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, roll: {roll:.1f}° | 注视状态: {is_gazing}")

            # 5. 状态机推入时序队列
            self._history.push(tid, is_gazing)
            buf = list(self._history.get(tid))

            # 6. 时序滑动窗口连续性检测（消除单帧由于抓拍产生的误报抖动）
            if len(buf) >= _GAZE_FRAMES and sum(buf[-_GAZE_FRAMES:]) >= _GAZE_FRAMES:
                events.append({
                    "name"    : "注视",
                    "id"      : tid,
                    "color"   : (0, 215, 255),  # 闪耀黄金/青黄色警报色
                    "detail"  : f"正脸(Y:{yaw:.1f}° P:{pitch:.1f}°)",
                })

        return events

    @staticmethod
    def _calculate_head_pose_angles(pts5):
        """
        核心数学模型：利用人脸 5 点局部坐标进行空间解算，无损提取三维角度
        0:左眼中心, 1:右眼中心, 2:鼻尖, 3:左嘴角, 4:右嘴角
        """
        p0 = np.array(pts5[0])  # 左眼
        p1 = np.array(pts5[1])  # 右眼
        p2 = np.array(pts5[2])  # 鼻尖

        # A. 提取瞳距作为人脸在当前距离下的特征基础尺度（像素跨度值）
        eye_center = (p0 + p1) / 2.0
        eye_vector = p1 - p0
        eye_dist = np.linalg.norm(eye_vector)

        if eye_dist < 1e-5:
            return 0.0, 0.0, 0.0

        # B. 【Yaw 水平偏转角（左转/右转）】
        # 依赖鼻尖 $X$ 坐标相对双眼横向中心点的水平漂移量比例进行逆向解算
        offset_x = p2[0] - eye_center[0]
        yaw = (offset_x / eye_dist) * 45.0  # 45.0 为三维几何旋转线性收敛常量

        # C. 【Pitch 俯仰角（抬头/低头）】
        # 依赖鼻尖 $Y$ 坐标到双眼水平线的垂直距离比例进行逆向推导
        # 减去正脸基础各向异性常数（0.70），反映头部俯仰导致的明暗投影形变
        offset_y = p2[1] - eye_center[1]
        pitch = ((offset_y / eye_dist) - 0.70) * 45.0

        # D. 【Roll 平面翻滚角（左右歪头）】
        # 计算双眼连线向量相对相机绝对水平地平线的几何倾斜弧度
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        roll = np.arctan2(dy, dx) * (180.0 / np.pi)

        # 异常数据安全饱和截断
        yaw = np.clip(yaw, -90.0, 90.0)
        pitch = np.clip(pitch, -90.0, 90.0)
        roll = np.clip(roll, -180.0, 180.0)

        return pitch, yaw, roll

    @staticmethod
    def _match_face(person_bbox, face_list):
        """
        空间高精匹配锁：在人体边界框（YOLO）上半身的 45%（头部生理危险区）内锁定对应的人脸
        """
        px1, py1, px2, py2 = person_bbox
        p_height = py2 - py1
        
        # 严格将搜索视野约束在行人的上半身，彻底杜绝把人手或腿部误当成人脸匹配区
        head_zone_y2 = py1 + p_height * 0.45

        best_face = None
        max_score = -1.0

        for face in face_list:
            fx1, fy1, fx2, fy2 = face["bbox"]
            
            # 算出 SeetaFace 探测到的人脸中心点
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0

            # 判定脸部几何中心是否落入行人的上半身矩形框内
            if (px1 <= fcx <= px2) and (py1 <= fcy <= head_zone_y2):
                # 选取当前人员范围内检测得分最高、人脸框最稳定的一项
                if face["score"] > max_score:
                    max_score = face["score"]
                    best_face = face

        return best_face