import cv2
import time
import threading
from ultralytics import YOLO



class RobotVisionPipeline:
    def __init__(self):
        # 加载模型 (部署时可更换为 .engine 格式)
        print("正在加载模型...")
        self.model_det = YOLO('yolov8s.pt')       
        self.model_pose = YOLO('yolov8s-pose.pt') 
        
        # 1. 初始化线程锁与共享变量
        self.lock = threading.Lock()
        self.frame = None          # 存储最新采集的原始帧
        self.results_det = None    # 存储最新检测结果
        self.results_pose = None   # 存储最新姿态结果
        self.running = True        # 控制线程生命周期

        # self.gstreamer_pipeline = (
        #             "v4l2src device=/dev/video0 ! "
        #             "video/x-raw, width=640, height=480, framerate=30/1 ! "
        #             "videoconvert ! "
        #             "video/x-raw, format=BGR ! "
        #             "appsink drop=1"
        #         )

    def capture_thread(self):
        
        # cap = cv2.VideoCapture(self.gstreamer_pipeline, cv2.CAP_GSTREAMER)
        # cap = cv2.VideoCapture("/home/user/下载/3.mp4") 
        cap = cv2.VideoCapture(0)  # 直接使用默认摄像头设备
        if not cap.isOpened():
            print("错误：无法打开 GStreamer 摄像头管道！")
            self.running = False
            return

        while self.running:
            success, f = cap.read()
            if success:
                # 加锁：更新全局最新帧
                with self.lock:
                    # 将画面强制调整为 640x480 像素
                    f = cv2.resize(f, (640, 480))
                    self.frame = f
            else:
                time.sleep(0.01) 
                
        cap.release()

    def inference_thread(self):
        """线程 2: 模型推理线程"""
        last_time = time.time()
        
        while self.running:
            # current_time = time.time()
            
            # # 3. 时间差判断：每隔 0.1 秒检测一次，时间未到就 continue
            # if current_time - last_time < 0.5:
            #     time.sleep(0.01) # 释放 CPU 资源
            #     continue
            # last_time = current_time
            time.sleep(0.5)
            # 安全地取出当前最新画面的副本，用于推理
            with self.lock:
                if self.frame is None:
                    continue
                infer_frame = self.frame.copy()
            
            # 执行推理任务 (在锁外部执行，确保推理计算不会阻塞图像采集和画面渲染)
            res_det = self.model_det(infer_frame, imgsz=320, verbose=False)[0]
            res_pose = self.model_pose(infer_frame, imgsz=320, verbose=False)[0]
            # res_pose = None
            # 获取框和关键点数据，供业务逻辑使用
            # boxes = res_det.boxes
            # keypoints = res_pose.keypoints

            # --- 业务逻辑判断区域 ---
            # if check_eat_medicine(boxes, keypoints):
            #     print("检测到吃药动作！")

            # 加锁：将推理出的最新结果存入共享变量
            with self.lock:
                self.results_det = res_det
                self.results_pose = res_pose

    def run(self):
        """主线程: 负责结果渲染与交互验证"""
        # 启动异步线程
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.inference_thread, daemon=True).start()

        print("系统已启动，按 'q' 键退出。")
        window_name = "Companion Robot Vision"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 设置窗口为可调整大小模式
        cv2.resizeWindow(window_name, 640, 480)          # 强制设定窗口的宽高为 640x480


        # while self.running:
        #     time.sleep(0.1) # 主线程空转，等待用户退出指令
        while self.running:
            # 取出最新的画面和对应的最新检测结果
            with self.lock:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                display_frame = self.frame.copy()
                r_det = self.results_det
                r_pose = self.results_pose

            # 4. 结果渲染：没有检测结果时显示原图；有检测到物体时再画框
            if r_det is not None and len(r_det.boxes) > 0:
                # 在画面上画出物体框
                display_frame = r_det.plot(img=display_frame)
                
                # 如果也检测到了骨骼关键点，叠加上去
                if r_pose is not None and len(r_pose.keypoints) > 0:
                    display_frame = r_pose.plot(img=display_frame)

            # cv2.imshow 必须在主线程中运行

            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = RobotVisionPipeline()
    pipeline.run()
