from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
from models import SRResNet
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval( model , dataloader ):
    # 记录每个样本 PSNR 和 SSIM值
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]
            # 前向传播.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]                
            # 计算 PSNR 和 SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            lr_img_size = lr_imgs.size(0)
            del lr_imgs, hr_imgs, sr_imgs
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                        data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                        data_range=255.)
            PSNRs.update(psnr, lr_img_size)
            SSIMs.update(ssim, lr_img_size)
            del sr_imgs_y , hr_imgs_y
    # 输出平均PSNR和SSIM
    print('PSNR = {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM = {ssims.avg:.3f}'.format(ssims=SSIMs))
    print('平均单张样本用时  {:.3f} 秒'.format((time.time()-start)/len(dataloader)))    
    return 

if __name__ == '__main__':
    # 测试集目录
    data_folder = "/mnt/data/izumihanako/MyCourses/WebInfoProcess/data"
    test_data_names = ["TestSet/·85"]

    # 预训练模型
    srresnet_checkpoint = "./results/checkpoint_srresnet.pth"
    scaling_factor = 2      # 放大比例

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint)
    model = SRResNet(scaling_factor=scaling_factor)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    for test_data_name in test_data_names:
        print("\n数据集 %s:\n" % test_data_name)
        # 定制化数据加载器
        test_dataset = SRDataset(data_folder , split='test' , crop_size=0 , scaling_factor=scaling_factor,
                                lr_img_type='imagenet-norm' , hr_img_type='[-1, 1]',
                                test_data_name=test_data_name )
        test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=1 , shuffle=False , 
                                                  num_workers=4 , pin_memory=True )
        eval( model = model , dataloader = test_loader )