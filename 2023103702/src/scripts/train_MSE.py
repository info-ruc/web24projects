# mean squared error
# 留出验证 hold-out
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import train_dataLoader,val_dataLoader
# from vgg19_model import model
# from resnet18_model import model
# from resnet34_model import model
# from resnet50_model import model
# from resnet101_model import model
# from resnext_model import model
# from mobileNet_model import model
# from shufflenet_model import model
# from densenet_model import model
from inception_model import model
# from convnext_model import model
# from swintransformer_model import model
from loss import loss_func3

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

device = torch.device("cuda")
model = model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
# torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
epoch_num=100

print("start training and validation......")

train_loss_list = []
val_loss_list = []
for epoch in range(epoch_num):
    print(epoch)
    print("start training......")
    model.train()
    train_loss = 0
    for i, data in enumerate(train_dataLoader):
        inputs, labels = data
        labels = labels.to(torch.float)
        # labels=labels.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs,_ = model(inputs)
        outputs = softmax(outputs)

        # loss = loss_func1(outputs,labels)+loss_func2(outputs,labels)
        score_tensor = torch.tensor([0., 1., 2., 3., 4.]).cuda()

        # print("outputs")
        # print(outputs)

        outputs = outputs.mm(score_tensor.view(-1, 1)).squeeze()
        # print(outputs)

        #         print("labels")
        #         print(labels)
        #         print("outputs")
        #         print(outputs)

        # loss = loss_func1(outputs1,labels)+loss_func2(outputs,labels)
        loss = loss_func3(outputs, labels)

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
        for i, data in enumerate(val_dataLoader):
            inputs, labels = data
            labels = labels.to(torch.float)
            inputs, labels = inputs.to(device), labels.to(device)

            # labels=labels.unsqueeze(1)

            outputs = model(inputs)
            outputs = softmax(outputs)
            # loss=loss_func1(outputs,labels)+loss_func2(outputs,labels)
            score_tensor = torch.tensor([0., 1., 2., 3., 4.]).cuda()
            outputs = outputs.mm(score_tensor.view(-1, 1)).squeeze()

            # print('labels ', labels)
            # print('outputs ', outputs)

            # loss=loss_func1(outputs1,labels)+loss_func2(outputs,labels)
            loss = loss_func3(outputs, labels)
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

print("Finished......")

with open("./MSELoss/inceptionv3_trainval_MSELoss.txt",'w') as f:
    for i in range(len(train_loss_list)):
        f.write(str(i)+" "+str(train_loss_list[i])+" "+str(val_loss_list[i])+"\n")

plt.title('loss',fontsize=15)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('loss value',fontsize=15)
plt.plot(range(1,epoch_num+1),train_loss_list,'-',c='g')
plt.plot(range(1,epoch_num+1),val_loss_list,'-',c='purple')
plt.legend(['train','val'])
plt.show()

train_loss_list_toTensor=torch.tensor(train_loss_list)
val_loss_list_toTensor=torch.tensor(val_loss_list)
torch.save(train_loss_list_toTensor,"./CEKappaLoss/inceptionv3_train_MSELoss")
torch.save(val_loss_list_toTensor,"./CEKappaLoss/inceptionv3_val_MSELoss")
torch.save(model.state_dict(),'F:/PostGraduate/TaskOne/parameters_inceptionv3_MSELoss.pkl')