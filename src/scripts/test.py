# 单张图片超分测试
from utils import *
from torch import nn
from models import SRResNet
import time
from PIL import Image
import cv2
import numpy

# 测试图像
imgPath = './results/cat.jpg'
model_epoch = 460

scaling_factor = 2      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 预训练模型
    srresnet_checkpoint = "./results/checkpoint_srresnet-epoch"+str(model_epoch)+".pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint)
    model = SRResNet(scaling_factor=scaling_factor)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
   
    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.jpg')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    start = time.time()
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil') 
        # opencv 双边滤波   
        cv2img = cv2.cvtColor(numpy.array(sr_img), cv2.COLOR_RGB2BGR)
        filtered_image = cv2.bilateralFilter(cv2img, d=9, sigmaColor=25, sigmaSpace=275)
        sr_img = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        sr_img.save('./results/test_srgan-epoch'+str(model_epoch)+'.jpg')
    print('模型为' + srresnet_checkpoint )
    print('SRRNet用时  {:.3f} 秒'.format(time.time()-start))

