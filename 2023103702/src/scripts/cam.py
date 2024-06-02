import torch.nn as nn
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform(tensor, height=7, width=7):
    '''
    不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
    height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
    比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
    height = width = 224 / 32 = 7
    '''
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        # default='./examples/both.png',
        # default='./images/CERA_106_cy_259_cx_1968.png',
        # default='./CrossoverDetection/junc_coco_data/test/24_training.jpg',
        # default='./CERA_012_cy_2167_cx_1932.png',
        # default='./images/CERA_033_cy_460_cx_1457.png',
        default='./images/CERA_059_cy_704_cx_1334.png',
        # default='./images/CERA_048_cy_228_cx_2054.png',
        # default='./images/CERA_069_cy_1716_cx_1104.png',
        # default='./images/CERA_079_cy_228_cx_1104.png',
        # default='./images/CERA_056_cy_752_cx_1684.png',
        # default='./images/CERA_104_cy_664_cx_913.png',
        # default='./images/CERA_090_cy_1684_cx_1180.png',
        # default='./images/CERA_109_cy_337_cx_1452.png',
        # default='./images/CERA_122_cy_718_cx_1028.png',
        # default='./augment/cls3_aug_060_CERA_012_cy_2045_cx_1711.png',
        # default='img.png',
        # default='./CrossoverDetection/hhh.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # from resnet50_model import model
    from maskrcnn_resnet50_fpn_model import ori_model,MaskRCNN_Backbone_Grading
    model = MaskRCNN_Backbone_Grading(ori_model.backbone)

    # for param in model.parameters():
    #     param.requires_grad = True
    # from fasterrcnn_resnet50_fpn_model import ori_model,FasterRCNN_Backbone_Grading
    # model = FasterRCNN_Backbone_Grading(ori_model.backbone)
    # model.load_state_dict(torch.load("./parameters_FasterRCNNbackbone_oriImage_CEKappaLoss.pkl"))
    # model.load_state_dict(torch.load("./parameters_resnet50_oriImage_CEKappaLoss.pkl"))
    # model.load_state_dict(torch.load("./Everyepoch/resnet50_oriImage_CEKappaLoss_49.pkl"))



    # model.load_state_dict(torch.load("./Everyepoch/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_89.pkl"))

    # model.load_state_dict(torch.load("./frozen_parameters_move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss.pkl"))

    model.load_state_dict(torch.load("./Everyepoch/2_frozen_move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_19.pkl"))


    # model.load_state_dict(torch.load("./Everyepoch/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_6.pkl"))


    # model.load_state_dict(torch.load("./Everyepoch/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_14.pkl"))


    # model.load_state_dict(torch.load("./Everyepoch/move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_29.pkl"))



    # model.load_state_dict(torch.load("./Everyepoch/smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_19.pkl"))
    # model.load_state_dict(torch.load("./Everyepoch/patch_MaskRCNNbackbone_oriImage_CEKappaLoss_19.pkl"))
    # model.load_state_dict(torch.load("./best_MaskRCNNbackbone_oriImage_CEKappaLoss.pkl"))
    print(model)
    # from resnet50_model import model
    # model.load_state_dict(torch.load("F:/PostGraduate/TaskOne/parameters_resnet50_CEKappaLoss.pkl"))
    # print(model)

    # for param in model.parameters():
    #     print(param.requires_grad)
    #     param.requires_grad=True

    # model.load_state_dict(torch.load('F:/PostGraduate/TaskOne/parameters_swintransformer_CEKappaLoss.pkl'))

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.backbone.body.layer4]
    # target_layers = [model.layer4]
    class_map = {0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3"}
    class_id = 3
    class_name = class_map[class_id]

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    from PIL import Image
    rgb_img = Image.open(args.image_path)
    from torchvision import transforms
    # rgb_img = rgb_img.resize((400,400))
    # rgb_img = transforms.CenterCrop(170)(rgb_img)
    # rgb_img = rgb_img.resize((400,400))
    rgb_img = rgb_img.resize((224,224))
    # rgb_img = transforms.CenterCrop(100)(rgb_img)
    # rgb_img = transforms.Pad(padding=62,padding_mode='constant')(rgb_img)
    #
    # rgb_img = transforms.RandomAffine(degrees=30,translate=(0.35,0.35))(rgb_img)
    # rgb_img = transforms.RandomResizedCrop(224,scale=(0.75,1))(rgb_img)
    # rgb_img = transforms.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5, 1.5),
    #                        hue=(-0.5, 0.5),
    #                        brightness=(0.875, 1.125))(rgb_img)


    # rgb_img = rgb_img.resize((224,224))
    print('type',type(rgb_img))
    # rgb_img = rgb_img.resize((400,400))
    # rgb_img = transforms.Pad(padding=672, fill=0, padding_mode='constant')(rgb_img)
    # rgb_img = transforms.Resize([400,400])(rgb_img)
    rgb_img = np.array(rgb_img)

    # rgb_img = cv2.resize(rgb_img,(224,224))

    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = [ClassifierOutputTarget(class_id)]

    # targets = None
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        print('grayscale_cam', grayscale_cam)
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # plt.imshow(rgb_img)
        # plt.title("origin image")

        plt.imshow(cam_image)
        # plt.title(class_name)
        plt.show()

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)