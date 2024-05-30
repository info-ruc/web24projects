from torch.utils.data import Dataset
import json
import os
import random
from PIL import Image
from utils import convert_image

# 验证集图像最大大小限制
test_max_img_size = 1000 * 1000

class ImageTransforms(object):
    """
    图像变换.
    """
    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 如果图像过大，则进行裁剪，否则可能会炸显存
            if img.width * img.height > test_max_img_size :
                scale_rate = 1 / img.width / img.height * test_max_img_size # 面积比值
                scale_rate = scale_rate ** 0.5  # 边长比值
                crop_width = int( scale_rate * img.width ) // self.scaling_factor * self.scaling_factor
                crop_height = int( scale_rate * img.height ) // self.scaling_factor * self.scaling_factor
                left = random.randint(1, img.width - crop_width )
                top = random.randint(1, img.height - crop_height )
                right = left + crop_width
                bottom = top + crop_height
                hr_img = img.crop((left, top, right, bottom))   
            else :
                x_remainder = img.width % self.scaling_factor
                y_remainder = img.height % self.scaling_factor
                left = x_remainder // 2
                top = y_remainder // 2
                right = left + (img.width - x_remainder )
                bottom = top + (img.height - y_remainder )
                hr_img = img.crop((left, top, right, bottom))

        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
        return lr_img, hr_img


class SRDataset(Dataset):
    """
    数据集加载器
    """
    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        """
        :参数 data_folder: # Json数据文件所在文件夹路径
        :参数 split: 'train' 或者 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸  （实际训练时不会用原图进行放大，而是截取原图的一个子块进行放大）
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        :参数 test_data_name: 待评估数据集路径
        """
        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("请提供测试数据集路径!")
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        # 如果是测试，则不需要对图像的长宽作限定
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "裁剪尺寸不能被放大比例整除!"

        # 读取图像路径
        if self.split == 'train':
            with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder, self.test_data_name.split("/")[-1] + '_test_images.json'), 'r') as j:
                self.images = json.load(j)

        # 数据处理方式
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 160 or img.height <= 160 :
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.images)