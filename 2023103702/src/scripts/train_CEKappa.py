# 留出验证 hold-out
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from vgg19_model import model
# from resnet18_model import model
# from resnet50_model import model
# from resnet34_model import model
# from resnet101_model import model
# from resnext_model import model
# from mobileNet_model import model
# from shufflenet_model import model
# from densenet_model import model
# from inception_model import model
# from convnext_model import model
# from swintransformer_model import model
# from fasterrcnn_resnet50_fpn_model import ori_model
from maskrcnn_resnet50_fpn_model import ori_model

from loss import loss_func1,loss_func2

from plotFigure import ConfusionMatrix

from split_dataset import test_dataset,train_dataset,val_dataset
from torch.utils.data.sampler import WeightedRandomSampler
# weights = []
labels = {}
for img, label in train_dataset:
    if label not in labels.keys():
        labels[label]=1
    else:
        labels[label]=labels[label]+1

print(labels)

weights = [1.0/labels[label] for img, label in train_dataset]

print(weights)

# for img, label in train_dataset:
#     if label == 0:
#         weights.append(1.0/24)
#     elif label == 1:
#         weights.append(1.0/8)
#     elif label == 2:
#         weights.append(1.0/7)
#     elif label == 3:
#         weights.append(1.0/6)

# print(weights)
sampler = WeightedRandomSampler(weights=weights,num_samples=400,replacement=True)

train_dataLoader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler, num_workers=0)
# print(len(train_dataLoader))
val_dataLoader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


def softmax(X):
    # print(X)
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from maskrcnn_resnet50_fpn_model import MaskRCNN_Backbone_Grading
model = MaskRCNN_Backbone_Grading(backbone=ori_model.backbone)
model.to(device)

# model.load_state_dict(torch.load("./Everyepoch/frozen_move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_19.pkl"))

# model.load_state_dict(torch.load("./Everyepoch/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_99.pkl"))

# model.load_state_dict(torch.load('./Everyepoch/patch_MaskRCNNbackbone_oriImage_CEKappaLoss_99.pkl'))
# model.load_state_dict(torch.load("./Everyepoch/resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_9.pkl"))

# for param in model.parameters():
#     param.requires_grad=True    #smaller_roi成功

for name, parameter in model.named_parameters():
    print(name,parameter.requires_grad)


# optimizer=torch.optim.Adam(model.parameters(),lr=7e-5) #smaller_roi成功
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-5) #move_resize_smaller_patch成功
# optimizer = torch.optim.Adam(model.parameters(),lr=5e-6)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
# torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
epoch_num=100

print("start training and validation......")

# MaskRCNNbackbone_oriImage_train_cekappaloss = torch.load("./CEKappaLoss/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_train_CEKappaLoss")
# MaskRCNNbackbone_oriImage_val_cekappaloss = torch.load("./CEKappaLoss/move_resize_smaller_patch_MaskRCNNbackbone_oriImage_val_CEKappaLoss")
# train_loss_list = MaskRCNNbackbone_oriImage_train_cekappaloss.tolist()
# val_loss_list = MaskRCNNbackbone_oriImage_val_cekappaloss.tolist()
train_loss_list = []
val_loss_list = []
min_qk_loss=100
maxKappa=0.0
for epoch in range(epoch_num):
    # epoch = epoch+100
    print(epoch)
    print("start training......")
    model.train()
    train_loss = 0
    for i, data in enumerate(train_dataLoader):
        inputs, labels = data
        # labels=labels.to(torch.float)
        # labels=labels.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # print('outputs.shape:', outputs.shape)
        outputs1 = softmax(outputs)

        # loss = loss_func1(outputs,labels)+loss_func2(outputs,labels)
        # score_tensor=torch.tensor([0.,1.,2.,3.,4.]).cuda()  ***

        # print("outputs")
        # print(outputs)

        # outputs=outputs.mm(score_tensor.view(-1,1))        ***

        #         print("labels")
        #         print(labels)
        #         print("outputs")
        #         print(outputs)

        loss = loss_func1(outputs1, labels) + loss_func2(outputs, labels)

        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        # print("step={}; loss={};".format(i,loss.item()))
        train_loss += float(loss.item())
    loss_per_epoch = train_loss / len(train_dataLoader)
    train_loss_list.append(loss_per_epoch)
    print("the training loss of {}th epoch is: {}".format(epoch, loss_per_epoch))

    print("start validation......")

    # total_accuracy=0.0
    # confusion = ConfusionMatrix(num_classes=5,labels=['level 0','level 1','level 2','level 3','level 4'])
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for j, data in enumerate(val_dataLoader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # labels=labels.unsqueeze(1)

            outputs = model(inputs)
            outputs1 = softmax(outputs)
            # loss=loss_func1(outputs,labels)+loss_func2(outputs,labels)
            # score_tensor=torch.tensor([0.,1.,2.,3.,4.]).cuda()
            # outputs=outputs.mm(score_tensor.view(-1,1))

            loss = loss_func1(outputs1, labels) + loss_func2(outputs, labels)
            # print(loss)

            batch_size = inputs.size(0)
            # print("step={}; loss={};".format(i,loss.item()))
            val_loss += float(loss.item())
            # ret,predictions=torch.max(outputs.data,1)
            # confusion_matrix
            # confusion.update(predictions.cpu().numpy(),labels.cpu().numpy())
        loss_per_epoch = val_loss / len(val_dataLoader)
        val_loss_list.append(loss_per_epoch)
        # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
        print("the validation loss of {}th epoch is: {}".format(epoch, loss_per_epoch))
        # confusion.plot()
        # confusion.summary()

    model.eval()
    with torch.no_grad():
        confusion = ConfusionMatrix(num_classes=4, labels=['level 0', 'level 1', 'level 2', 'level 3'])
        test_loss = 0
        ce_test_loss = 0
        qk_test_loss = 0
        for k, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print('outputs.shape:',outputs.shape)

            # score_tmp = softmax(outputs)
            # score_list.extend(score_tmp.detach().cpu().numpy())
            # label_list.extend(labels.cpu().numpy())

            outputs1 = softmax(outputs)
            loss = loss_func1(outputs1, labels) + loss_func2(outputs, labels)
            qkloss = loss_func1(outputs1, labels)
            celoss = loss_func2(outputs, labels)
            batch_size = inputs.size(0)
            # print("step={}; loss={};".format(i,loss.item()))
            test_loss += float(loss.item())
            qk_test_loss += float(qkloss.item())
            ce_test_loss += float(celoss.item())
            ret, predictions = torch.max(outputs.data, 1)
            # print('outputs', outputs)
            # print('ret', ret)
            # print('predictions', predictions)
            # confusion_matrix
            confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())

        # plot_curve(score_list, label_list)

        test_loss = test_loss / len(test_dataloader)
        qk_test_loss = qk_test_loss / len(test_dataloader)
        ce_test_loss = ce_test_loss / len(test_dataloader)
        # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
        print("the test loss is: {}".format(test_loss))
        print("the ce_test_loss is: {}".format(ce_test_loss))
        print("the qk_test_loss is: {}".format(qk_test_loss))

        # if test_loss < min_qk_loss:
        #     min_qk_loss = test_loss
        #     torch.save(model.state_dict(), './best_resnet50_oriImage_CEKappaLoss.pkl')
        if (epoch+1)%10==0:
            torch.save(model.state_dict(),'./Everyepoch/frozen_move_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_{}.pkl'.format(epoch))
        # torch.save(model.state_dict(),'./Everyepoch/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss_{}.pkl'.format(epoch))

        qkappa = confusion.plot()
        # if qkappa > maxKappa:
        #     maxKappa = qkappa
        #     print("max kappa is",qkappa)
        #     torch.save(model.state_dict(),'./bestnow.pkl'.format(epoch))



print("Finished......")

with open("./CEKappaLoss/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_trainval_CEKappaLoss.txt",'w') as f:
    for i in range(len(train_loss_list)):
        f.write(str(i)+" "+str(train_loss_list[i])+" "+str(val_loss_list[i])+"\n")

plt.title('loss',fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('loss value',fontsize=15)
plt.plot(range(1,epoch_num+1),train_loss_list,'-')
plt.plot(range(1,epoch_num+1),val_loss_list,'-')
plt.legend(['train','val'])
plt.show()

train_loss_list_toTensor=torch.tensor(train_loss_list)
val_loss_list_toTensor=torch.tensor(val_loss_list)
torch.save(train_loss_list_toTensor, "./CEKappaLoss/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_train_CEKappaLoss")
torch.save(val_loss_list_toTensor, "./CEKappaLoss/frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_val_CEKappaLoss")
torch.save(model.state_dict(), './parameters_frozen_move_resize_smaller_patch_MaskRCNNbackbone_oriImage_CEKappaLoss.pkl')