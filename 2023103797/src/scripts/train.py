import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import SRResNet
from datasets import SRDataset
from utils import *
from eval import eval
from tqdm import tqdm

# 数据集参数
data_folder = '/mnt/data/izumihanako/MyCourses/WebInfoProcess/data' # 数据存放路径
crop_size = 210                 # 高分辨率图像裁剪尺寸
test_data_name = "TestSet/·85"  # 训练时验证集

# 学习参数
checkpoint = "./results/checkpoint_srresnet.pth"   # 预训练模型路径
scaling_factor = 2  # 放大比例
batch_size = 50    # batch大小
start_epoch = 1     # 轮数起始位置
epochs = 460        # 迭代轮数
workers = 20        # 工作线程数
# lr = 1e-4         # 前100 ep 学习率
lr = 1e-5           # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True # 对卷积进行加速

def main():
    global checkpoint,start_epoch
    # 初始化
    model = SRResNet(scaling_factor=scaling_factor)
    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        
    train_dataset = SRDataset( data_folder,split='train' , crop_size=crop_size , scaling_factor=scaling_factor ,
                               lr_img_type='imagenet-norm', hr_img_type='[-1, 1]' )
    train_loader = torch.utils.data.DataLoader( train_dataset , batch_size=batch_size ,
                    shuffle=True , num_workers=workers , pin_memory=True )

    if test_data_name is not None:
        test_dataset = SRDataset(data_folder , split='test' , crop_size=0 , scaling_factor=scaling_factor,
                                lr_img_type='imagenet-norm' , hr_img_type='[-1, 1]',
                                test_data_name=test_data_name )
        test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=1 , shuffle=False , 
                                                    num_workers=4 , pin_memory=True )

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):
        model.train()  # 训练模式：允许使用批样本归一化
        loss_epoch = AverageMeter()  # 统计损失函数
        # 按批处理
        for i, (lr_imgs, hr_imgs) in tqdm( enumerate(train_loader) , total = len(train_loader) ):
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式
            sr_imgs = model(lr_imgs)# 前向传播
            loss = criterion(sr_imgs, hr_imgs)  # 计算损失
            optimizer.zero_grad()   # 后向传播
            loss.backward()
            optimizer.step()        # 更新模型
            # 记录损失值
            loss_epoch.update(loss.item(), lr_imgs.size(0))
            # 释放内存
            del lr_imgs, hr_imgs, sr_imgs
        print("第 "+str(epoch)+" epoch结束, 评估函数如下：")           
        print('MSE_Loss = {mseloss:.6f}'.format(mseloss=loss_epoch.val) )
        if test_data_name is not None:
            eval( model = model , dataloader = test_loader )

        # 保存预训练模型
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'results/checkpoint_srresnet.pth')
        if epoch % 10 == 0 :        
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'results/checkpoint_srresnet-epoch' +str(epoch) + '.pth')
            

if __name__ == '__main__':
    main()