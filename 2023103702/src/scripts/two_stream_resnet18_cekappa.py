import torch
import torch.nn as nn
from torchvision.models import resnet18
from loss import loss_func1,loss_func2,loss_func3
from PIL import Image
# from preprocess import setup_seed
model1 = resnet18(pretrained=True)
model2 = resnet18(pretrained=True)
import pprint
classifier = nn.Sequential()
model1.fc = classifier
model2.fc = classifier
model3 = nn.Linear(1024,5,bias=True)

# setup_seed(28)
fh = open('F:/PostGraduate/TaskOne/grading.txt', 'r')
full_imgs = []
flag = 0

for line in fh:
    if flag == 0:
        flag = flag + 1
        continue
    line = line.rstrip()
    words = line.split(", ")
    full_imgs.append(("F:/PostGraduate/TaskOne/images/" + words[1] + ".png", "F:/PostGraduate/TaskOne/segmentation_images/" + words[1] + ".png" , int(words[2])))

import random
numOfData = len(full_imgs)
rate = 0.4
random.seed(42)
# testlist=random.sample(list(range(0,90)),int(numOfData*rate))

test_imgs = random.sample(full_imgs, int(numOfData * rate))
# test_imgs2 = random.sample(full_imgs2, int(numOfData * rate))
# test_imgs1 = [full_imgs1[i] for i in testlist]
# test_imgs2 = [full_imgs2[i] for i in testlist]

length = len(test_imgs)
for i in range(length):
    full_imgs.remove(test_imgs[i])


import os
from torchvision import transforms
aug_img_path = "F:/PostGraduate/TaskOne/augment"
if not os.path.exists(aug_img_path):
    os.mkdir(aug_img_path)

train_val_imgs = full_imgs.copy()
for i in range(len(full_imgs) * 10):
    ori_img = Image.open(full_imgs[i % len(full_imgs)][0])
    train_transform = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
    aug_img = train_transform(ori_img)
    # axes1 = plt.subplot(1,2,1)
    # plt.imshow(ori_img)
    # axes2 = plt.subplot(1,2,2)
    # plt.imshow(aug_img)
    # plt.show()
    ori_img_path = full_imgs[i % len(full_imgs)][0]
    seg_img_path = full_imgs[i % len(full_imgs)][1]
    # print(ori_img_path[0:-3])
    path_token = ori_img_path.split("/")
    new_img_path = aug_img_path + "/aug_" + '%03d' % i + "_" + path_token[-1]
    train_val_imgs.append((new_img_path, seg_img_path, full_imgs[i % len(full_imgs)][2]))
    aug_img.save(new_img_path)


from preprocess import MyDataset,myTransform
from torchvision import transforms
train_val_dataset = MyDataset(train_val_imgs,transform=myTransform['train'])
test_dataset = MyDataset(test_imgs,transform=myTransform['test'])

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(67))
from torch.utils.data import DataLoader
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataLoader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
# print('full_imgs1',len(full_imgs1))
# print(full_imgs1)
# print("full_imgs2",len(full_imgs2))
# print(full_imgs2)
#
# if(full_imgs1==full_imgs2): print("true")
# else: print("false")

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

device = torch.device("cuda")
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)
optimizer1=torch.optim.Adam(model1.parameters(),lr=5e-6)
optimizer2=torch.optim.Adam(model2.parameters(),lr=5e-6)
optimizer3=torch.optim.Adam(model3.parameters(),lr=5e-6)

# torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
epoch_num=20

train_loss_list=[]
val_loss_list=[]

print("start training......")

for epoch in range(epoch_num):
    print(epoch)
    print("start training......")
    model1.train()
    model2.train()
    model3.train()
    train_loss = 0
    for i, data in enumerate(train_dataLoader):
        inputs1, inputs2, labels = data
        # labels=labels.to(torch.float)
        # labels=labels.unsqueeze(1)
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs1 = model1(inputs1)
        outputs2 = model2(inputs2)
        outputs = torch.concat((outputs1,outputs2),dim=1)
        outputs = model3(outputs)
        # print(outputs)
        outputs3 = softmax(outputs)
        # loss = loss_func1(outputs,labels)+loss_func2(outputs,labels)
        # score_tensor=torch.tensor([0.,1.,2.,3.,4.]).cuda()  ***
        # print("outputs")
        # print(outputs)
        # outputs=outputs.mm(score_tensor.view(-1,1))        ***

        #         print("labels")
        #         print(labels)
        #         print("outputs")
        #         print(outputs)

        loss = loss_func1(outputs3, labels) + loss_func2(outputs, labels)

        # print(loss)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        # batch_size = inputs.size(0)
        # print("step={}; loss={};".format(i,loss.item()))
        train_loss += float(loss.item())
    loss_per_epoch = train_loss / len(train_dataLoader)
    train_loss_list.append(loss_per_epoch)
    print("the training loss of {}th epoch is: {}".format(epoch, loss_per_epoch))

    model1.eval()
    model2.eval()
    model3.eval()
    with torch.no_grad():
        val_loss = 0
        for j, data in enumerate(val_dataLoader):
            inputs1, inputs2, labels = data
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            # labels=labels.unsqueeze(1)
            outputs1 = model1(inputs1)
            outputs2 = model2(inputs2)
            outputs = torch.concat((outputs1, outputs2), dim=1)
            outputs = model3(outputs)
            # print(outputs)
            outputs3 = softmax(outputs)
            # loss=loss_func1(outputs,labels)+loss_func2(outputs,labels)
            # score_tensor=torch.tensor([0.,1.,2.,3.,4.]).cuda()
            # outputs=outputs.mm(score_tensor.view(-1,1))

            loss = loss_func1(outputs3, labels) + loss_func2(outputs, labels)
            # print(loss)
            # batch_size = inputs.size(0)
            # print("step={}; loss={};".format(i,loss.item()))
            val_loss += float(loss.item())
            # ret,predictions=torch.max(outputs.data,1)
            # confusion_matrix
            # confusion.update(predictions.cpu().numpy(),labels.cpu().numpy())
        loss_per_epoch = val_loss / len(val_dataLoader)
        val_loss_list.append(loss_per_epoch)
        # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
        print("the validation loss of {}th epoch is: {}".format(epoch, loss_per_epoch))


train_loss_list_toTensor=torch.tensor(train_loss_list)
val_loss_list_toTensor=torch.tensor(val_loss_list)
# torch.save(train_loss_list_toTensor,"./CEKappaLoss/twostreamResnet18_train_CEKappaLoss")
# torch.save(val_loss_list_toTensor,"./CEKappaLoss/twostreamResnet18_val_CEKappaLoss")
torch.save(model1.state_dict(),'F:/PostGraduate/TaskOne/epoch10_parameters_model1_twostreamResnet18_CEKappaLoss.pkl')
torch.save(model2.state_dict(),'F:/PostGraduate/TaskOne/epoch10_parameters_model2_twostreamResnet18_CEKappaLoss.pkl')
torch.save(model3.state_dict(),'F:/PostGraduate/TaskOne/epoch10_parameters_model3_twostreamResnet18_CEKappaLoss.pkl')