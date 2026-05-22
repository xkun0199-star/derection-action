import cv2
import threading
import queue
import time
import SeetaFacePy

# 定义一个全局的线程安全队列，用于给算法线程输送最新的图像帧
# 限制最大长度为 1延迟累积
frame_queue = queue.Queue(maxsize=1)

# 定义一个结果队列，用于将算法处理完、画好人脸框和特征点的图像传回主线程
result_queue = queue.Queue(maxsize=1)

# 全局运行标志
running = True

def face_inference_thread():
    """
    独立的后台 AI 推理异步线程
    负责：初始化 SeetaFace 引擎、从队列取图、检测人脸和特征点、绘制图形
    """
    global running
    print("[AI线程] 正在初始化 SeetaFace2 引擎，请稍候...")
    
    try:
        # 1. 使用有参构造函数初始化人脸检测器（激活底层模型ID，修复score恒为1.0的问题）
        det_setting = SeetaFacePy.ModelSetting("../model/fd_2_00.dat", SeetaFacePy.ModelSetting.CPU)
        detector = SeetaFacePy.FaceDetector(det_setting)

        detector.set(SeetaFacePy.FaceDetector.Property.MIN_FACE_SIZE, 40)
        detector.set(SeetaFacePy.FaceDetector.Property.THRESHOLD1, 0.75)
        print("[AI线程] 最小人脸尺寸（MIN_FACE_SIZE）成功设置为: 40 像素")
        # 2. 初始化关键点定位器
        land_setting = SeetaFacePy.ModelSetting("../model/pd_2_00_pts5.dat", SeetaFacePy.ModelSetting.CPU)
        landmarker = SeetaFacePy.FaceLandmarker(land_setting)
        print("[AI线程] SeetaFace2 引擎初始化成功，开始监听输入...")
    except Exception as e:
        print(f"[AI线程] 引擎初始化失败，请检查模型路径！错误信息: {e}")
        running = False
        return

    while running:
        if frame_queue.empty():
            time.sleep(0.005) # 队列为空时稍微休息，防止空转占用过多 CPU
            continue
            
        # 从队列中取出最新的摄像头原始帧 (这是一个普通的 NumPy 数组)
        raw_frame = frame_queue.get()
        
        # 3. 直接触发 C++ 底层高性能人脸检测 (NumPy 缓冲区直连，避免临时变量释放导致点乱跳)
        faces = detector.detect(raw_frame)
        
        # 创建一个用于画图展示的副本，不破坏原始帧数据
        display_frame = raw_frame.copy()
        
        # 4. 遍历检测到的所有人脸并提取关键点
        for face in faces:
            # 绘制人脸矩形框 (绿色，线宽为 2)
            cv2.rectangle(display_frame, 
                          (face.x, face.y), 
                          (face.x + face.width, face.y + face.height), 
                          (0, 255, 0), 2)
            
            # 在人脸框上方打印检测置信度得分
            cv2.putText(display_frame, f"Conf: {face.score:.2f}", 
                        (face.x, face.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1)
            
            # 5. 直接将原始 NumPy 图像和 face 框传入 5 点定位器
            points = landmarker.mark(raw_frame, face)
            
            # 遍历 5 个特征点并在图像上画红色的实心小圆点
            for pt in points:
                cv2.circle(display_frame, (int(pt.x), int(pt.y)), 3, (0, 0, 255), -1)
        
        # 将绘制好人脸框和特征点的结果帧放入传出队列，供主线程显示
        if result_queue.full():
            try:
                result_queue.get_nowait() # 如果队列满了，丢弃旧的一帧结果，保证实时性
            except queue.Empty:
                pass
        result_queue.put(display_frame)


def main():
    """
    主线程
    负责：读取摄像头、将最新帧分发给 AI 线程、高速刷新 OpenCV 窗口显示
    """
    global running
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/user/SeetaFace2/ass/4.mp4")
    if not cap.isOpened():
        print("[主线程] 错误：无法打开摄像头，请检查设备连接或权限。")
        return
        
    # 设置摄像头硬件采集的原始分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 🌟【新增核心】：创建显示窗口并强行锁死大小为 640*480
    window_name = "SeetaFace2 Multi-Thread Camera Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 允许调整窗口大小
    cv2.resizeWindow(window_name, 640, 480)          # 强制指定窗口大小为 640x480

    # 创建并启动后台 AI 算法推理线程
    algo_thread = threading.Thread(target=face_inference_thread)
    algo_thread.daemon = True # 设置为守护线程，主线程退出时自动关闭
    algo_thread.start()

    print("[主线程] 摄像头视频流已启动。按键盘上的 'Q' 键可安全退出程序。")
    
    # 用于计算并显示实时 FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"
    
    # 存储当前正在显示的帧
    current_display_frame = None

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[主线程] 无法从摄像头读取数据。")
            break
            
        # 镜像翻转画面（符合照镜子习惯）
        frame = cv2.flip(frame, 1)

        # 实时将最新获取到的原始画面塞入输入队列供 AI 线程消费
        if frame_queue.full():
            try:
                frame_queue.get_nowait() # 丢弃队列里上一帧来不及处理的旧图，确保 AI 总是拿最新图
            except queue.Empty:
                pass
        frame_queue.put(frame)

        # 检查算法线程是否返回了画好人脸框的最新结果
        if not result_queue.empty():
            current_display_frame = result_queue.get()
        
        # 如果算法还没返回过任何结果，先直接显示摄像头的原始画面
        if current_display_frame is None:
            current_display_frame = frame.copy()

        # 计算并向当前显示帧叠加显示实时帧率 (FPS)
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps_text = f"FPS: {fps_counter}"
            fps_counter = 0
            fps_start_time = time.time()
            
        cv2.putText(current_display_frame, fps_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 🌟【修改点】：确保最终送入的是刚刚命名且锁定了分辨率的窗口
        # 如果画面尺寸与 640x480 不一致，OpenCV 会自动等比例缩放填充该窗口
        cv2.imshow(window_name, current_display_frame)

        # 监听键盘输入，按下 'q' 或 'Q' 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # 清理现场，释放资源
    print("[主线程] 正在关闭程序并释放资源...")
    running = False
    cap.release()
    cv2.destroyAllWindows()
    algo_thread.join()
    print("[主线程] 程序已安全退出。")

if __name__ == "__main__":
    main()
