"""
多线程推理管线：
  线程1（capture_thread）：从摄像头/文件持续采集最新帧
  线程2（inference_thread）：YOLO检测+姿态 + SeetaFace2 + 事件判断
  主线程：渲染结果 + 响应键盘
"""
import threading
import time
import queue
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import config
from detectors import YOLODetector, SeetaFaceDetector
from events    import build_all_detectors
from utils.geometry import draw_event_overlay
from utils.zh_text  import put_text_zh


class DetectionPipeline:

    def __init__(self, source=None):
        self._source  = source if source is not None else config.VIDEO_SOURCE
        self._running = True
        self._lock    = threading.Lock()

        # 共享数据
        self._latest_frame: np.ndarray | None = None
        self._latest_result: dict | None       = None   # {frame, events, det, pose, faces}

        # 模型
        print("[Pipeline] 正在加载 YOLO 模型...")
        self._yolo = YOLODetector()
        print("[Pipeline] 正在加载 SeetaFace2 模型...")
        self._seeta = SeetaFaceDetector()

        # 事件检测器
        self._event_detectors = build_all_detectors()
        print(f"[Pipeline] 已注册 {len(self._event_detectors)} 个事件检测器。")

    # ── 采集线程 ──────────────────────────────────────────
    def _capture_thread(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            print("[Capture] 无法打开视频源，请检查配置。")
            self._running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        # 视频文件按原始帧率限速；摄像头(fps==0)不限速
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = (1.0 / src_fps) if src_fps > 1 else 0.0
        print(f"[Capture] 视频帧率: {src_fps:.1f} fps, 帧间隔: {frame_interval*1000:.1f} ms")

        while self._running:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            with self._lock:
                self._latest_frame = frame

            # 按原始帧率限速，消耗掉本帧剩余时间
            elapsed = time.time() - t0
            wait = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)
        cap.release()

    # ── 推理线程 ──────────────────────────────────────────
    def _inference_thread(self):
        while self._running:
            time.sleep(config.INFERENCE_INTERVAL)

            with self._lock:
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()

            # 1. YOLO 检测 + 姿态
            det_list  = self._yolo.detect(frame)
            pose_list = self._yolo.pose(frame)

            # 2. SeetaFace2 人脸检测
            face_list = self._seeta.detect(frame)

            # 3. 逐一运行事件检测器
            all_events = []
            for detector in self._event_detectors:
                try:
                    evs = detector.detect(frame, pose_list, det_list, face_list)
                    all_events.extend(evs)
                    print(f"[Event] {detector.__class__.__name__} 检测到事件: {evs}")
                except Exception as e:
                    print(f"[Event] {detector.__class__.__name__} 异常: {e}")

            with self._lock:
                self._latest_result = {
                    "frame" : frame,
                    "events": all_events,
                    "det"   : det_list,
                    "pose"  : pose_list,
                    "faces" : face_list,
                }

    # ── 主线程（渲染 + 交互） ─────────────────────────────
    def run(self):
        threading.Thread(target=self._capture_thread,   daemon=True).start()
        threading.Thread(target=self._inference_thread, daemon=True).start()

        win = "Event Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        fps_t   = time.time()
        fps_cnt = 0
        fps_str = "FPS: -"

        while self._running:
            # 每次都取最新原始帧，保证画面流畅不卡顿
            with self._lock:
                raw    = self._latest_frame
                result = self._latest_result

            if raw is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 在最新帧上叠加推理结果（推理结果可能比画面慢几帧，但画面本身不卡）
            display = raw.copy()
            if result is not None:
                self._draw_detections(display, result["det"], result["pose"])
                self._draw_faces(display, result["faces"])
                draw_event_overlay(display, result["events"])

            fps_cnt += 1
            if time.time() - fps_t >= 1.0:
                fps_str = f"FPS: {fps_cnt}"
                fps_cnt = 0
                fps_t   = time.time()
            cv2.putText(display, fps_str, (config.FRAME_WIDTH - 110, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._running = False
        cv2.destroyAllWindows()

    # ── 绘制辅助 ─────────────────────────────────────────
    @staticmethod
    def _draw_detections(frame, det_list, pose_list):
        for d in det_list:
            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
            label = f"{d['label']} {d['conf']:.2f} #{d['track_id']}"
            color = (0, 200, 0) if d["cls"] == config.COCO_PERSON else (200, 200, 0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for p in pose_list:
            # 绘制简化骨架（肩-髋-膝-踝）
            kps_xy   = p["kps_xy"]
            kps_conf = p["kps_conf"]
            _draw_skeleton(frame, kps_xy, kps_conf)

    @staticmethod
    def _draw_faces(frame, face_list):
        for f in face_list:
            x1,y1,x2,y2 = [int(v) for v in f["bbox"]]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 200), 1)
            for pt in f["pts5"]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 100, 255), -1)


    def run_image(self, image_path: str, save_path: str | None = None):
        """
        静态图片推理：同步单帧处理，结果显示并可选保存。
        save_path 为 None 时仅显示，不为 None 时写出结果图。
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Image] 无法读取图片: {image_path}")
            return

        # 统一缩放到配置分辨率
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        det_list  = self._yolo.detect(frame)
        pose_list = self._yolo.pose(frame)
        face_list = self._seeta.detect(frame)

        all_events = []
        for detector in self._event_detectors:
            try:
                evs = detector.detect(frame, pose_list, det_list, face_list)
                all_events.extend(evs)
                if evs:
                    print(f"[Event] {detector.__class__.__name__}: {[e['name'] for e in evs]}")
            except Exception as e:
                print(f"[Event] {detector.__class__.__name__} 异常: {e}")

        display = frame.copy()
        self._draw_detections(display, det_list, pose_list)
        self._draw_faces(display, face_list)
        draw_event_overlay(display, all_events)

        # 左下角打印输入文件名
        name = os.path.basename(image_path)
        put_text_zh(display, name, (8, config.FRAME_HEIGHT - 28),
                    color=(180, 180, 180), font_size=16)

        if save_path:
            cv2.imwrite(save_path, display)
            print(f"[Image] 结果已保存: {save_path}")

        cv2.namedWindow("Event Detection - Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Event Detection - Image", config.FRAME_WIDTH, config.FRAME_HEIGHT)
        cv2.imshow("Event Detection - Image", display)
        print("[Image] 按任意键或关闭窗口退出。")
        while True:
            key = cv2.waitKey(100) & 0xFF
            # 任意键、ESC、q 均退出；窗口被关闭时 getWindowProperty 返回 -1
            if key != 255 or cv2.getWindowProperty(
                    "Event Detection - Image", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def _draw_skeleton(frame, kps_xy, kps_conf):
    """绘制主要骨架连线"""
    LINKS = [
        (5,6),(5,7),(7,9),(6,8),(8,10),   # 上肢
        (5,11),(6,12),(11,12),             # 躯干
        (11,13),(13,15),(12,14),(14,16),   # 下肢
    ]
    for a, b in LINKS:
        if kps_conf[a] > 0.4 and kps_conf[b] > 0.4:
            pa = (int(kps_xy[a,0]), int(kps_xy[a,1]))
            pb = (int(kps_xy[b,0]), int(kps_xy[b,1]))
            cv2.line(frame, pa, pb, (0, 180, 255), 2)
