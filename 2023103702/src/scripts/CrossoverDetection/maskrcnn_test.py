import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer=256
    model.roi_heads.mask_predictor=MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_instance_segmentation_model(num_classes=2)
# model.load_state_dict(torch.load('./maskrcnnROIOutput/smaller_roi_model_14.pth')['model'])
# model.load_state_dict(torch.load('./maskrcnnROIOutput/roi_model_14.pth')['model'])
model.load_state_dict(torch.load('./maskrcnnOutput/model_59.pth')['model'])
model.to(device)

COCO_INSTANCE_CATEGORY_NAMES = [
 '__background__', 'crossover', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    print(img.size)
    # img = img.resize((150,150))
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    pred = model([img.to(device)])
    print('pred')
    print(pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    print("masks>0.5")
    # print(pred[0]['masks'] > 0.5)
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # print("this is masks")
    # print(masks)
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class

def random_color_masks(image):
    print(image.shape)
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def instance_segmentation_api(img_path, threshold=0.8, rect_th=2, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    # img = cv2.resize(img,(150,150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])
        print('img',img.shape)
        print('rgb_mask',rgb_mask.shape)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.3, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    # plt.figure(figsize=(20,30))

    # scale_image = Image.open('../images/CERA_079_cy_228_cx_1104.png')

    # new_img = np.zeros((70,70,3),dtype=np.uint8)
    # for j in range(new_img.shape[0]):
    #     for k in range(new_img.shape[1]):
    #         for channel in range(3):
    #             new_img[j][k][channel]=img[j+240][k+240][channel]

    # new_img = Image.fromarray(new_img)
    # new_img = new_img.resize((401,401))
    # new_img = np.array(new_img)
    # for ii in range(new_img.shape[0]):
    #     for jj in range(new_img.shape[1]):
    #         for kk in range(3):
    #             new_img[ii][jj][kk]=int(new_img[ii][jj][kk])
    plt.imshow(img)
    # plt.imshow(new_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # cv2.imshow("1",img)
    # cv2.waitKey()


model.eval()

# instance_segmentation_api("../images/CERA_012_cy_2045_cx_1711.png")
# instance_segmentation_api("./tmpOutput/CERA_009_cy_575_cx_2051.png")
# instance_segmentation_api("../../TaskOne/images/CERA_033_cy_460_cx_1457.png")
# instance_segmentation_api("./tmpInput/padimg.png")
# instance_segmentation_api("../../TaskOne/images/CERA_049_cy_2176_cx_1878.png")
# instance_segmentation_api("../../TaskOne/images/CERA_123_cy_1696_cx_1898.png")
# instance_segmentation_api("../../TaskOne/images/CERA_106_cy_259_cx_1968.png")
# instance_segmentation_api("./junc_coco_data/test/sSTAR 03_OSN_Merged.jpg")
# instance_segmentation_api("./junc_coco_data/train/sSTAR 20_ODC_Merged.jpg")
# instance_segmentation_api("./junc_coco_data/train/25_training.jpg")
# instance_segmentation_api("./junc_coco_data/train/01_test.jpg")
instance_segmentation_api("./junc_coco_data/train/sSTAR 02_ODC_Merged.jpg")
# instance_segmentation_api("./junc_coco_data/test/24_training.jpg")
# instance_segmentation_api("../../TaskOne/images/CERA_101_cy_140_cx_1514.png")
# instance_segmentation_api("../../TaskOne/images/CERA_017_cy_614_cx_1475.png")
# instance_segmentation_api("../../TaskOne/images/CERA_049_cy_2176_cx_1878.png")
# instance_segmentation_api("../../TaskOne/images/CERA_015_cy_582_cx_1436.png")
# instance_segmentation_api("../images/CERA_069_cy_1716_cx_1104.png")
# instance_segmentation_api("../images/CERA_079_cy_228_cx_1104.png")
# instance_segmentation_api("../images/CERA_056_cy_752_cx_1684.png")
# instance_segmentation_api("./hhh.jpg")
# img = Image.open("../../TaskOne/images/CERA_012_cy_2167_cx_1932.png")
# # img = Image.open("./junc_coco_data/test/sSTAR 24_OSC_Merged.jpg")
# # print(img.shape)
# from torchvision import transforms
# img = np.array(img)
# im2 = img.copy()
# img = transforms.ToTensor()(img)
# # img = img.unsqueeze(0)
# print(img.shape)
# with torch.no_grad():
#     prediction=model([img.to(device)])
#
# import matplotlib.pyplot as plt
# import random
# # img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
#
# # img2 = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
# for i in range(len(prediction[0]['masks'])):
#     msk = prediction[0]['masks'][i,0].detach().cpu().numpy()
#     print(msk)
#     showed = prediction[0]['scores'][i].detach().cpu().numpy()
#     if showed>0:
#         im2[:,:,0][msk>0.5] = random.randint(0,255)
#         im2[:,:,1][msk>0.5] = random.randint(0,255)
#         im2[:,:,2][msk>0.5] = random.randint(0,255)
#
# plt.imshow(im2)
# plt.show()