import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("正在加载 MobileNet 大脑，请稍候...")
# 1. 加载预训练的 MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# 【核心操作】砍掉最后一层分类器，让它只输出 1280 维的特征向量，而不是具体的类别
model.classifier = torch.nn.Identity() 
model.eval() # 设置为推理模式

# 2. 图像预处理流水线 (将摄像头画面变成模型喜欢的格式)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(frame):
    """把摄像头的一帧画面变成一串数字（特征向量）"""
    # OpenCV 默认是 BGR 颜色，需要转成 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    input_tensor = preprocess(pil_img).unsqueeze(0) # 增加 batch 维度
    
    with torch.no_grad():
        features = model(input_tensor)
    return features.numpy()

# 记忆库
memory = {
    "Item_A": None,
    "Item_B": None
}

# 3. 打开摄像头
cap = cv2.VideoCapture(0)
print("\n摄像头已启动！")
print("操作指南：")
print(" - 按 'a' 键：将当前画面记住为【物品 A】")
print(" - 按 'b' 键：将当前画面记住为【物品 B】")
print(" - 按 'q' 键：退出程序\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 提取当前画面的特征
    current_feature = extract_features(frame)
    
    display_text = "Status: Looking..."
    
    # 如果记忆库里有东西，就开始计算相似度
    if memory["Item_A"] is not None or memory["Item_B"] is not None:
        best_match = "Unknown"
        highest_score = 0.0
        
        # 遍历记忆库，计算余弦相似度
        for name, saved_feature in memory.items():
            if saved_feature is not None:
                # 计算余弦相似度 (值在 -1 到 1 之间，越接近 1 越相似)
                score = cosine_similarity(current_feature, saved_feature)[0][0]
                if score > highest_score:
                    highest_score = score
                    best_match = name
        
        # 设定一个阈值（比如 0.85），大于这个值才认为是认识的物品
        if highest_score > 0.85:
            display_text = f"I see: {best_match} (Score: {highest_score:.2f})"
        else:
            display_text = f"Unknown object (Highest score: {highest_score:.2f})"

    # 在画面上显示结果
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Few-Shot Robot Vision Test', frame)

    # 键盘控制逻辑
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        memory["Item_A"] = current_feature
        print(">>> 已记住【物品 A】！")
    elif key == ord('b'):
        memory["Item_B"] = current_feature
        print(">>> 已记住【物品 B】！")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
