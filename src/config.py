"""全局配置：模型路径、检测阈值、帧率控制"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
LIB_DIR   = os.path.join(BASE_DIR, "..", "lib")

# ── 模型路径 ──────────────────────────────────────────────
YOLO_DET_MODEL   = os.path.join(BASE_DIR, "yolov8s.pt")
YOLO_POSE_MODEL  = os.path.join(BASE_DIR, "yolov8s-pose.pt")
SEETA_DET_MODEL  = os.path.join(MODEL_DIR, "fd_2_00.dat")
SEETA_LAND5_MODEL= os.path.join(MODEL_DIR, "pd_2_00_pts5.dat")
SEETA_LAND81_MODEL=os.path.join(MODEL_DIR, "pd_2_00_pts81.dat")

# ── 推理参数 ──────────────────────────────────────────────
YOLO_IMGSZ        = 640
YOLO_CONF         = 0.40
SEETA_MIN_FACE    = 10          # px
SEETA_THRESHOLD   = 0.75

# 推理前将帧缩小到此尺寸，降低 CPU 负载；None 表示不缩放
INFER_WIDTH  = 640
INFER_HEIGHT = 480

# ── 视频源 ────────────────────────────────────────────────
# 摄像头填 0，视频文件填路径字符串
VIDEO_SOURCE = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# 采集线程最大帧率上限（摄像头/视频均适用）
CAPTURE_MAX_FPS = 30

# ── 推理线程间隔(秒) ─────────────────────────────────────
INFERENCE_INTERVAL = 0.00       # ~3 fps 推理，主线程渲染独立不受影响



# ── 事件置信度阈值 ────────────────────────────────────────
THRESHOLDS = {
    # 人相关
    "back_facing"    : 0.6,     # 背身：鼻子关键点置信度低
    "gazing"         : 0.5,     # 注视
    "eye_state"      : 0.3,     # 闭眼/打哈欠/眨眼（EAR）
    "waving"         : 0.55,    # 招手：腕关节速度
    "fall_down"      : 0.6,     # 摔倒：躯干倾角
    "talking"        : 0.5,     # 交谈/倾听：嘴部开合

    # 人与环境
    "pick_up"        : 0.5,     # 拿起物品：手与物体 IoU
    "door_operate"   : 0.5,     # 开/关门窗
    "sit_stand"      : 0.55,    # 就座/离座：臀部高度变化

    # 环境
    "fire_smoke"     : 0.45,    # 火灾/烟雾：YOLO 类别
}

# ── YOLO COCO 类别索引（常用） ────────────────────────────
COCO_PERSON = 0
COCO_CHAIR  = 56
COCO_BOTTLE = 39
COCO_CUP    = 41
COCO_DOOR_RELATED = []          # 标准 COCO 无门，依赖位置推断

# ── 姿态关键点索引（COCO 17点）────────────────────────────
KP = {
    "nose":0, "left_eye":1, "right_eye":2,
    "left_ear":3, "right_ear":4,
    "left_shoulder":5, "right_shoulder":6,
    "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10,
    "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14,
    "left_ankle":15, "right_ankle":16,
}

# ── 眼睛纵横比(EAR)参数 ───────────────────────────────────
EAR_CLOSE_THRESH  = 0.20        # 低于此值视为闭眼
EAR_CONSEC_FRAMES = 3           # 连续帧判定

# ── 摔倒检测参数 ──────────────────────────────────────────
FALL_ANGLE_THRESH = 45          # 躯干与水平线夹角(度)
FALL_RATIO_THRESH = 0.8         # bbox 宽/高比

# ── 招手检测参数 ──────────────────────────────────────────
WAVE_WRIST_SPEED  = 30          # 像素/帧，腕关节速度阈值
WAVE_WINDOW       = 10          # 滑动窗口帧数

# ── 火焰/烟雾检测（HSV颜色范围） ─────────────────────────
FIRE_HSV_LOWER  = (0,  120, 120)
FIRE_HSV_UPPER  = (30, 255, 255)
SMOKE_GRAY_DIFF = 30            # 灰度均值差阈值
