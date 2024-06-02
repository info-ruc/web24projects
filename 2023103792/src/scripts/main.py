import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# 解析命令行参数
parser = argparse.ArgumentParser(description='YOLOv8 Video Stream Detection')
parser.add_argument('--source', type=str, required=True, help='URL of the video stream')
parser.add_argument('--weights', type=str, default='weights/best.pt', help='Path to YOLOv8 weights file')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold for danger detection')
args = parser.parse_args()

# 加载训练好的YOLOv8模型
model = YOLO(args.weights)

# 捕获视频流
cap = cv2.VideoCapture(args.source)

# 定义IoU计算函数
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型进行检测
    results = model.predict(frame, conf=args.conf)  # 使用传入的置信度阈值

    # 获取检测框和标签
    boxes = results[0].boxes
    labels = results[0].names

    # 计算IoU并输出报警信息
    num_boxes = len(boxes)
    if num_boxes > 1:
        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                label1 = labels[boxes[i].cls[0].item()]
                label2 = labels[boxes[j].cls[0].item()]
                
                if (label1 == 'child' and label2 == 'window') or (label1 == 'window' and label2 == 'child'):
                    box1 = boxes[i].xyxy[0].tolist()
                    box2 = boxes[j].xyxy[0].tolist()
                    iou = calculate_iou(box1, box2)
                    if iou > args.iou_thresh:
                        print(f'Danger! IoU between child and window: {iou}')

    # 可视化检测结果
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        label = labels[box.cls[0].item()]
        color = (0, 255, 0) if label == 'child' else (0, 0, 255) if label == 'window' else (255, 0, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 显示结果帧
    cv2.imshow('YOLOv8 Object Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
