from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
device=torch.device('cuda')
def showbbox(model,img_tensor):
    model.eval()
    with torch.no_grad():
        prediction=model([img_tensor.to(device)])
    print(prediction)
    img_tensor = img_tensor.permute(1, 2, 0)  # C,H,W → H,W,C，用来画图
    img = (img_tensor * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img)  # tensor → ndarray

    img = img.copy()

    scores=prediction[0]['scores']

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())

        label = prediction[0]['labels'][i].item()

        confidence = scores[i].item()

        if confidence>0.15:

            if label == 1:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
                # cv2.putText(img, 'CrossOver', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                #             thickness=1)

            elif label == 2:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=1)
                cv2.putText(img, 'BiffPos', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                            thickness=1)

            elif label == 3:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
                cv2.putText(img, 'EndPoint', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            thickness=1)

    # cv2.imshow('tt',img)
    # cv2.waitKey()
    plt.figure(figsize=(20, 15))
    plt.imshow(img)
    plt.show()

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda")
model.to(device)
model.load_state_dict(torch.load('./result/model_49.pth')['model'])

# img=cv2.imread("./junc_coco_data/test/24_training.jpg")
# img=cv2.imread("../../TaskOne/images/CERA_079_cy_228_cx_1104.png")
img = cv2.imread("../../TaskOne/CERA_012_cy_2167_cx_1932.png")
# img = img[:,:,::-1]
# img = np.ascontiguousarray(img)
# img = img.copy()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = cv2.imread("../../TaskOne/img.png")
# img = cv2.imread("../../TaskOne/ttt2.jpg")
# img = cv2.imread("../../TaskOne/CERA_101_cy_140_cx_1514.png")
# img = cv2.resize(img,(80,80))
img_tensor=transforms.ToTensor()(img)
# img_tensor = img_tensor.unsqueeze(0)

showbbox(model,img_tensor)