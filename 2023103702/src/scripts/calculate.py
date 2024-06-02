import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from maskrcnn_resnet50_fpn_model import ori_model,MaskRCNN_Backbone_Grading
model = MaskRCNN_Backbone_Grading(ori_model.backbone)
model.load_state_dict(torch.load("./Everyepoch/frozen_move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_9.pkl"))

# from torchvision.models import resnet18
# model1 = resnet18(pretrained=True)
# model2 = resnet18(pretrained=True)
# classifier = nn.Sequential()
# model1.fc = classifier
# model2.fc = classifier
# model3 = nn.Linear(1024,5,bias=True)
# model1.load_state_dict(torch.load("./parameters_model1_twostreamResnet18_CEKappaLoss.pkl"))
# model2.load_state_dict(torch.load("./parameters_model2_twostreamResnet18_CEKappaLoss.pkl"))
# model3.load_state_dict(torch.load("./parameters_model3_twostreamResnet18_CEKappaLoss.pkl"))
device = torch.device("cuda")
model = model.to(device)
# model1,model2,model3=model1.to(device),model2.to(device),model3.to(device)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

import numpy as np
model.eval()
# model1.eval()
# model2.eval()
# model3.eval()
with torch.no_grad():
    # path = './images/CERA_104_cy_664_cx_913.png'
    # path = './images/CERA_049_cy_2282_cx_1404.png'
    # path = './images/CERA_090_cy_1684_cx_1180.png'
    path = './images/CERA_109_cy_337_cx_1452.png'
    # path1 = 'F:/PostGraduate/TaskOne/images/CERA_056_cy_752_cx_1684.png'
    # path2 = 'F:/PostGraduate/TaskOne/segmentation_images/CERA_056_cy_752_cx_1684.png'
    img = Image.open(path)
    img = img.resize((224,224))
    import torchvision
    # img = torchvision.transforms.CenterCrop(70)(img)
    # img = img.resize((800,800))
    # img1, img2 = Image.open(path1), Image.open(path2)
    # img1, img2 = img1.resize((224,224)) , img2.resize((224,224))
    # axes1 = plt.subplot(1, 3, 1)
    # plt.imshow(img1)
    # axes2 = plt.subplot(1, 3, 2)
    # plt.imshow(img2)
    plt.imshow(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    # img = np.array(img).transpose(2,0,1)
    # img1 = np.array(img1).transpose(2,0,1)
    # img2 = np.array(img2).transpose(2,0,1)
    # img1 = img1/255
    # img2 = img2/255
    # img = img/255
    # img = np.expand_dims(img, axis=0)
    # img1 = np.expand_dims(img1, axis=0)  # (N, Ci, Hi, Wi)
    # img2 = np.expand_dims(img2, axis=0)


    # img = torch.tensor(img, dtype=torch.float32).cuda()


    # img1 = torch.tensor(img1, dtype=torch.float32).cuda()
    # img2 = torch.tensor(img2, dtype=torch.float32).cuda()
    outputs = model(img)
    print(outputs)
    # output1 = model1(img1)
    # output2 = model2(img2)
    # outputs = torch.concat((output1,output2),dim=1)
    # outputs = model3(outputs)
    outputs = softmax(outputs)

    plt.axis('on')
    plt.title('The result is level {}'.format(outputs.argmax()))
    plt.show()
    print(outputs)


from split_dataset import test_imgs
for u in test_imgs:
    print(u)

# from maskrcnn_resnet50_fpn_model import ori_model
# print(ori_model)
