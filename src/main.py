"""
入口：自动识别输入类型并分流
  python main.py                        # 摄像头实时
  python main.py 0                      # 摄像头 #0
  python main.py /path/video.mp4        # 视频文件
  python main.py /path/image.jpg        # 静态图片（显示+可选保存）
  python main.py /path/image.jpg --save result.jpg
"""
import sys
import os
# 限制 OpenBLAS/OMP 线程数，防止 SeetaFace2/YOLO 底层线程爆炸（放在所有 import 之前）
os.environ.setdefault("OMP_NUM_THREADS",   "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS",   "2")
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DetectionPipeline
import config

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def main():
    args = sys.argv[1:]

    # 解析 --save
    save_path = None
    if "--save" in args:
        idx = args.index("--save")
        if idx + 1 < len(args):
            save_path = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("用法: --save <输出路径>")
            sys.exit(1)

    # 解析输入源
    source = args[0] if args else config.VIDEO_SOURCE
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    is_image = (isinstance(source, str)
                and os.path.splitext(source)[1].lower() in _IMAGE_EXTS)

    # 图片模式不需要视频源，传 None 跳过摄像头初始化
    pipeline = DetectionPipeline(source=None if is_image else source)

    if is_image:
        print(f"[Main] 图片模式: {source}")
        pipeline.run_image(source, save_path=save_path)
    else:
        print(f"[Main] 视频/摄像头模式: {source}")
        pipeline.run()


if __name__ == "__main__":
    main()
