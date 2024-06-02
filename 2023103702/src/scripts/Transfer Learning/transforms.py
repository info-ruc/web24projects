import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def resize_boxes(boxes, original_size, new_size):
    # 将boxes的参数根据图像的缩放情况进行相应缩放
    # 新尺寸/旧尺寸 得到 缩放因子 分别对应h和w方向
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # unbind： 移除指定维度，返回一个元组，包含了沿着指定维度切片后的各个切片。
    # boxes shape: [b, 4]. [b]代表当前图像中有几个boxes的信息 [4]为boxes的左上角、右下角坐标
    # xmin, ymin, xmax, ymax分别存储所有bbox的坐标。shape: [b, ]
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    # 还原坐标信息: [b, 4]
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class RcnnTransforms(nn.Module):
    # 对图像进行预处理
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(RcnnTransforms, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)  # 转为tuple类型

        self.min_size = min_size  # 指定图像的最小边长范围
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std  # 指定图像在标准化处理中的方差

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: mean/std: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # 缩放图像和bbox到指定的范围内
        # shape: [C, H, W]
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        # 获取h、w中的最小值
        min_size = float(torch.min(im_shape))
        # 获取h、w中的最大值
        max_size = float(torch.max(im_shape))

        # 指定输入图片的最小边长 self.min_size = 800
        size = float(self.min_size[-1])
        # 根据指定的最小边长和图片最小边长 计算 缩放比例
        scale_factor = size / min_size
        # 使用该缩放比例 计算图像最大边长。 如果图像最大边长大于指定的最大边长 则重新定义缩放比例
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        # interpolate利用插值的方法缩放图片 根据scale_factor将图像进行缩放
        # image shape:[C, H, W] -> image[None] shape: [1, C, H, W]
        # [0]: shape: [1, C, H, W] -> [C, H, W]
        image = F.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            # 对应验证模式
            return image, target

        # training
        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        # h, w为图像原始尺寸 image.shape[-2:]为缩放后图像的尺寸
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])  # 将坐标信息完成缩放
        target['boxes'] = bbox
        return image, target

    def max_by_axis(self, shape_list):
        # list中分别存储一个batch中所有图像的shape信息
        # 取出第一张图像的shape作为maxes
        maxes = shape_list[0]

        for sublist in shape_list[1:]:
            for index, item in enumerate(sublist):
                # index: 0, 1, 2 item: C, H, W
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # 将多张图像的尺寸缩放到统一尺寸后，需将此打包成一个batch，然后输入到网络中。
        # 分别计算一个batch中所有图片中的最大channel, height, width
        max_size = self.max_by_axis([list(img.shape) for img in images])
        # 将最大的h，w向上取整到离32最近的倍数
        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到最靠近stride的整数倍的值
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到最靠近stride的整数倍的值
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size
        # 创建shape为batch_shape且值全部为0的tensor
        # images[0] 无论是哪张图像效果都一样
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将img复制到pad_img中 img shape: [C, H, W] pad_img shape: [C, H, W]
            # images中的每张图像都复制到batched_imgs中，都是从0开始, 目的是对齐左上角，保证bbox不变
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def restore_bbox(self, pred_result, resize_shape, origin_image_size):
        # 将预测的结果映射回原图像中
        # pred_result 网络的预测结果。 包含bbox信息和对应bbox的类别信息
        # resize_shape 图像resize后，每个图像的h，w
        # origin_image_size 每一张图像在缩放前的原始尺寸
        if self.training:
            return pred_result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度（验证）
        for i, (pred, im_s, o_im_s) in enumerate(zip(pred_result, resize_shape, origin_image_size)):
            # pred：对应输入的batch中每一张图像的预测信息
            # im_s： 对应输入的batch中每一张图像的resize后的尺寸
            # o_im_s对应输入的batch中每一张图像的原始尺寸
            boxes = pred["boxes"] # 获取预测的bbox信息
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            pred_result[i]["boxes"] = boxes
        return pred_result

    def postprocess(self, result, image_shapes, original_image_sizes):
        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        # result为网络的预测结果。 包含 bbox信息和类别预测信息
        # image_shapes 图像resize后，每个图像的h，w
        # original_image_sizes每一张图像在缩放前的原始尺寸
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度（验证）
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            # 将bboxes缩放回原图像尺度上 boxes：坐标信息 im_s: 缩放后的图像尺度 o_im_s：原始的图像尺度
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def forward(self, images, targets=None):
        # 遍历每一张图片 组成一个list
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            # boxes, labels, imag_id, area, is_crowd
            target_index = targets[i] if targets is not None else None
            # 归一化
            image = self.normalize(image)
            # 将图像和对应的bboxes缩放到指定范围
            image, target_index = self.resize(image, target_index)

            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸 每张图像的尺寸可能不同
        image_sizes = [img.shape[-2:] for img in images]
        # 将images打包成一个batch H, W可能会发生变化，使得H, W接近32的倍数
        images = self.batch_images(images)

        image_sizes_list = []
        for image_size in image_sizes:
            image_sizes_list.append((image_size[0], image_size[1]))

        # image_sizes_list需要记录resize之后的尺寸， 原尺寸保存在target中的'boxes'， 但网络最终的预测结果需要在原图像进行标注。因此通过resize后的尺寸与原图像尺寸 将预测结果映射回原图上
        # images：打包之后的tensor。 image_sizes_list：记录的是resize之后图像的H,W信息。
        image_list = ImageList(images, image_sizes_list)

        return image_list, targets # 输入到backbone
