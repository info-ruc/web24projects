import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from .segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator

sys.path.append("../../..")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


import multiprocessing as mp
# 设置启动方法为 'spawn'
mp.set_start_method('spawn', force=True)
def sam_input(image):
    image_np = np.array(image)
    # OpenCV默认读取格式为BGR，如果原图是RGB，需要转换为BGR
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # image = cv2.imread('test.png')
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()
    sam_checkpoint = "/home/fulongtai/CoCa3D/opencood/data_utils/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    # 用于产生prompt
    mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    '''
    segmentation ：掩码area：掩码的面积（以像素为单位）
    bbox：XYWH 格式的掩码边界框
    predicted_iou：模型自己对掩码质量的预测
    point_coords：生成此掩码的采样输入点
    stability_score：衡量掩码质量的一个附加指标
    crop_box：用于生成此掩码的图像剪裁区域，以 XYWH 格式表示
    '''
    masks2 = mask_generator_2.generate(image)
    # 创建一个单独的图
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # # 遍历掩模并将它们添加到同一个图中
    # for i, key in enumerate(masks2):
    #     show_mask(key["segmentation"], plt.gca(), True)
    #     score = key["stability_score"]
    # plt.title("Masks with Stability Scores", fontsize=18)
    # plt.axis('off')
    # plt.show()
    return masks2


if __name__ == '__main__':
    sam_input(1)