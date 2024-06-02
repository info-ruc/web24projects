# import matplotlib.pyplot as plt
# import torch
# convnext_train_cekappaloss=torch.load("./CEKappaLoss/swintransformer_train_CEKappaLoss")
# convnext_val_cekappaloss=torch.load("./CEKappaLoss/swintransformer_val_CEKappaLoss")
# plt.title('convnext loss',fontsize=15)
# plt.xlabel('epoch',fontsize=15)
# plt.ylabel('loss value',fontsize=15)
# plt.plot(range(1,101),convnext_train_cekappaloss.tolist(),'-')
# plt.plot(range(1,101),convnext_val_cekappaloss.tolist(),'-')
# plt.legend(['train','val'])
# plt.show()

import torch
import matplotlib.pyplot as plt
from thop import profile

def main():

    move_resize_smaller_patch_train_MaskRCNN=torch.load("./CEKappaLoss/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_train_CEKappaLoss")
    move_resize_smaller_patch_val_MaskRCNN=torch.load("./CEKappaLoss/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_val_CEKappaLoss")

    mask_train_ce = torch.load("./CEKappaLoss/patch_MaskRCNNbackbone_oriImage_train_CEKappaLoss")
    mask_val_ce = torch.load("./CEKappaLoss/patch_MaskRCNNbackbone_oriImage_val_CEKappaLoss")
    faster_train_ce = torch.load("./CEKappaLoss/FasterRCNNbackbone_oriImage_train_CEKappaLoss")
    faster_val_ce = torch.load("./CEKappaLoss/FasterRCNNbackbone_oriImage_val_CEKappaLoss")

    res_train_ce = torch.load("./CEKappaLoss/resnet50_oriImage_train_CEKappaLoss")
    res_val_ce = torch.load("./CEKappaLoss/resnet50_oriImage_val_CEKappaLoss")
    # swintransformer_train_cekappaloss=torch.load("./CEKappaLoss/swintransformer_train_CEKappaLoss")
    # convnext_train_cekappaloss=torch.load("./CEKappaLoss/convnext_train_CEKappaLoss")
    # # densenet_train_cekappaloss=torch.load("./CEKappaLoss/densenet_train_CEKappaLoss")
    # inceptionv3_train_cekappaloss=torch.load("./CEKappaLoss/inceptionv3_train_CEKappaLoss")
    # mobilenetv3_train_cekappaloss=torch.load("./CEKappaLoss/mobilenetv3_train_CEKappaLoss")
    # resnet18_train_cekappaloss=torch.load("./CEKappaLoss/resnet18_train_CEKappaLoss")
    # # resnet34_train_cekappaloss=torch.load("./CEKappaLoss/resnet34_train_CEKappaLoss")
    # resnet50_train_cekappaloss=torch.load("./CEKappaLoss/resnet50_train_CEKappaLoss")
    # resnet101_train_cekappaloss=torch.load("./CEKappaLoss/resnet101_train_CEKappaLoss")
    # resnext_train_cekappaloss=torch.load("./CEKappaLoss/resnext50_train_CEKappaLoss")
    # shufflenet_train_cekappaloss=torch.load("./CEKappaLoss/shufflenet_train_CEKappaLoss")
    # # vgg19_train_cekappaloss=torch.load("./CEKappaLoss/vgg19_train_CEKappaLoss")
    # twostreamResnet18_train_cekappaloss=torch.load("./CEKappaLoss/twostreamResnet18_train_CEKappaLoss")
    #
    #
    # swintransformer_train_mseloss = torch.load("./MSELoss/swintransformer_train_MSELoss")
    # convnext_train_mseloss = torch.load("./MSELoss/convnext_train_MSELoss")
    # # densenet_train_mseloss=torch.load("./MSELoss/densenet_train_MSELoss")
    # inceptionv3_train_mseloss = torch.load("./MSELoss/inceptionv3_train_MSELoss")
    # mobilenetv3_train_mseloss = torch.load("./MSELoss/mobilenetv3_train_MSELoss")
    # resnet18_train_mseloss = torch.load("./MSELoss/resnet18_train_MSELoss")
    # # resnet34_train_mseloss=torch.load("./MSELoss/resnet34_train_MSELoss")
    # resnet50_train_mseloss = torch.load("./MSELoss/resnet50_train_MSELoss")
    # resnet101_train_mseloss = torch.load("./MSELoss/resnet101_train_MSELoss")
    # resnext_train_mseloss = torch.load("./MSELoss/resnext_train_MSELoss")
    # shufflenet_train_mseloss = torch.load("./MSELoss/shufflenet_train_MSELoss")
    # # vgg19_train_mseloss = torch.load("./MSELoss/vgg19_train_MSELoss")
    # twostreamResnet18_train_mseloss=torch.load("./MSELoss/twostreamResnet18_train_MSELoss")
    #
    # swintransformer_val_cekappaloss = torch.load("./CEKappaLoss/swintransformer_val_CEKappaLoss")
    # convnext_val_cekappaloss = torch.load("./CEKappaLoss/convnext_val_CEKappaLoss")
    # # densenet_val_cekappaloss=torch.load("./CEKappaLoss/densenet_val_CEKappaLoss")
    # inceptionv3_val_cekappaloss = torch.load("./CEKappaLoss/inceptionv3_val_CEKappaLoss")
    # mobilenetv3_val_cekappaloss = torch.load("./CEKappaLoss/mobilenetv3_val_CEKappaLoss")
    # resnet18_val_cekappaloss = torch.load("./CEKappaLoss/resnet18_val_CEKappaLoss")
    # # resnet34_val_cekappaloss=torch.load("./CEKappaLoss/resnet34_val_CEKappaLoss")
    # resnet50_val_cekappaloss = torch.load("./CEKappaLoss/resnet50_val_CEKappaLoss")
    # resnet101_val_cekappaloss = torch.load("./CEKappaLoss/resnet101_val_CEKappaLoss")
    # resnext_val_cekappaloss = torch.load("./CEKappaLoss/resnext50_val_CEKappaLoss")
    # shufflenet_val_cekappaloss = torch.load("./CEKappaLoss/shufflenet_val_CEKappaLoss")
    # # vgg19_val_cekappaloss=torch.load("./CEKappaLoss/vgg19_val_CEKappaLoss")
    # twostreamResnet18_val_cekappaloss=torch.load("./CEKappaLoss/twostreamResnet18_val_CEKappaLoss")
    #
    #
    # swintransformer_val_mseloss = torch.load("./MSELoss/swintransformer_val_MSELoss")
    # convnext_val_mseloss = torch.load("./MSELoss/convnext_val_MSELoss")
    # # densenet_val_mseloss=torch.load("./MSELoss/densenet_val_MSELoss")
    # inceptionv3_val_mseloss = torch.load("./MSELoss/inceptionv3_val_MSELoss")
    # mobilenetv3_val_mseloss = torch.load("./MSELoss/mobilenetv3_val_MSELoss")
    # resnet18_val_mseloss = torch.load("./MSELoss/resnet18_val_MSELoss")
    # # resnet34_val_mseloss=torch.load("./MSELoss/resnet34_val_MSELoss")
    # resnet50_val_mseloss = torch.load("./MSELoss/resnet50_val_MSELoss")
    # resnet101_val_mseloss = torch.load("./MSELoss/resnet101_val_MSELoss")
    # resnext_val_mseloss = torch.load("./MSELoss/resnext_val_MSELoss")
    # shufflenet_val_mseloss = torch.load("./MSELoss/shufflenet_val_MSELoss")
    # # vgg19_val_mseloss = torch.load("./MSELoss/vgg19_val_MSELoss")
    # twostreamResnet18_val_mseloss=torch.load("./MSELoss/twostreamResnet18_val_MSELoss")
    #
    #
    # plt.title('The CE+Kappa trainLoss of different models',fontsize=15)
    # plt.xlabel('epoch',fontsize=12)
    # plt.ylabel('loss value',fontsize=12)
    #
    # # plt.plot(range(1, 101), vgg19_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1,101),resnet18_train_cekappaloss.tolist(),'-.')
    # # plt.plot(range(1,101),resnet34_train_cekappaloss.tolist(),'-.')
    # plt.plot(range(1,101),resnet50_train_cekappaloss.tolist(),'-.')
    # plt.plot(range(1, 101), resnet101_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1, 101), resnext_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1, 101), mobilenetv3_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1, 101), shufflenet_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1, 101), inceptionv3_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1,101),convnext_train_cekappaloss.tolist(),'-.')
    # # plt.plot(range(1,101),densenet_train_cekappaloss.tolist(),'-.')
    # plt.plot(range(1, 101), swintransformer_train_cekappaloss.tolist(), '-.')
    # plt.plot(range(1, 101), twostreamResnet18_train_cekappaloss.tolist(), '-.')
    #
    # plt.legend(['resnet18','resnet50','resnet101','resnext','mobilenetv3','shufflenet','inceptionv3','convnext','swintransformer','twostream-resnet18'])
    # plt.show()
    #
    #
    #
    #
    # plt.title('The MSE trainLoss of different models', fontsize=15)
    # plt.xlabel('epoch', fontsize=12)
    # plt.ylabel('loss value', fontsize=12)
    #
    # # plt.plot(range(1, 101), vgg19_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), resnet18_train_mseloss.tolist(), '-.')
    # # plt.plot(range(1,101),resnet34_train_mseloss.tolist(),'-.')
    # plt.plot(range(1, 101), resnet50_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), resnet101_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), resnext_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), mobilenetv3_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), shufflenet_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), inceptionv3_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101), convnext_train_mseloss.tolist(), '-.')
    # # plt.plot(range(1,101),densenet_train_mseloss.tolist(),'-.')
    # plt.plot(range(1, 101), swintransformer_train_mseloss.tolist(), '-.')
    # plt.plot(range(1, 101),twostreamResnet18_train_mseloss.tolist(), '-.')
    #
    # plt.legend(['resnet18','resnet50','resnet101','resnext','mobilenetv3','shufflenet','inceptionv3','convnext','swintransformer','twostream-resnet18'])
    # plt.show()

    plt.title('The CE+Kappa trainLoss of different models',fontsize=15)
    plt.xlabel('epoch',fontsize=12)
    plt.ylabel('loss value',fontsize=12)
    # plt.plot(range(1,101),move_resize_smaller_patch_train_MaskRCNN.tolist(),'-.')
    # plt.plot(range(1,101),mask_train_ce.tolist(),'-.')
    # plt.plot(range(1,101),faster_train_ce.tolist(),'-.')
    plt.plot(range(1,101),res_train_ce.tolist(),'-.')
    # plt.legend(['move_resize_smaller_patch','maskrcnn_backbone','resnet50'])
    plt.legend(['resnet50'])
    plt.show()
    #
    #
    plt.title('The CE+Kappa valLoss of different models',fontsize=15)
    plt.xlabel('epoch',fontsize=12)
    plt.ylabel('loss value',fontsize=12)
    # plt.plot(range(1,101),move_resize_smaller_patch_val_MaskRCNN.tolist(),'--')
    # plt.plot(range(1,101),mask_val_ce.tolist(),'--')
    # plt.plot(range(1, 101), faster_val_ce.tolist(), '--')
    plt.plot(range(1,101),res_val_ce.tolist(),'--')
    plt.legend(['resnet50'])
    # plt.legend(['move_resize_smaller_patch','maskrcnn_backbone','resnet50'])
    plt.show()
    #
    # # plt.plot(range(1, 101), vgg19_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1,101),resnet18_val_cekappaloss.tolist(),'--')
    # # plt.plot(range(1,101),resnet34_val_cekappaloss.tolist(),'--')
    # plt.plot(range(1,101),resnet50_val_cekappaloss.tolist(),'--')
    # plt.plot(range(1, 101), resnet101_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1, 101), resnext_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1, 101), mobilenetv3_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1, 101), shufflenet_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1, 101), inceptionv3_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1,101),convnext_val_cekappaloss.tolist(),'--')
    # # plt.plot(range(1,101),densenet_val_cekappaloss.tolist(),'--')
    # plt.plot(range(1, 101), swintransformer_val_cekappaloss.tolist(), '--')
    # plt.plot(range(1, 101), twostreamResnet18_val_cekappaloss.tolist(), '--')
    #
    # plt.legend(['resnet18','resnet50','resnet101','resnext','mobilenetv3','shufflenet','inceptionv3','convnext','swintransformer','twostream-resnet18'])
    # plt.show()
    #
    #
    #
    #
    # plt.title('The MSE valLoss of different models', fontsize=15)
    # plt.xlabel('epoch', fontsize=12)
    # plt.ylabel('loss value', fontsize=12)
    #
    # # plt.plot(range(1, 101), vgg19_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), resnet18_val_mseloss.tolist(), '--')
    # # plt.plot(range(1,101),resnet34_val_mseloss.tolist(),'--')
    # plt.plot(range(1, 101), resnet50_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), resnet101_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), resnext_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), mobilenetv3_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), shufflenet_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), inceptionv3_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), convnext_val_mseloss.tolist(), '--')
    # # plt.plot(range(1,101),densenet_val_mseloss.tolist(),'--')
    # plt.plot(range(1, 101), swintransformer_val_mseloss.tolist(), '--')
    # plt.plot(range(1, 101), twostreamResnet18_val_mseloss.tolist(), '--')
    #
    # plt.legend(['resnet18','resnet50','resnet101','resnext','mobilenetv3','shufflenet','inceptionv3','convnext','swintransformer','twostream-resnet18'])
    # plt.show()





if __name__ == '__main__':
    main()