import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    ori_model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = ori_model.roi_heads.box_predictor.cls_score.in_features
    ori_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ori_model.load_state_dict(torch.load('F:/PostGraduate/TaskOne/CrossoverDetection/result/model_49.pth')['model'])
    ori_model.to(device)
    ori_model.eval()
    ori_img = Image.open("F:/PostGraduate/TaskOne/images/CERA_012_cy_2167_cx_1932.png")
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(ori_img)
    img = torch.unsqueeze(img,dim=0)
    with torch.no_grad():
        features=ori_model.backbone.forward(img.to(device))
        print(features['0'].shape)
        print(features['1'].shape)
        features=list(features.values())
        print(features)
        for feature in features:
            feature = feature.detach().cpu().numpy()
            im = np.squeeze(feature)
            im = np.transpose(im, [1,2,0])
            plt.figure()
            for i in range(12):
                ax = plt.subplot(3,4,i+1)
                plt.imshow(im[:,:,i])
            plt.show()


if __name__ == '__main__':
    main()
