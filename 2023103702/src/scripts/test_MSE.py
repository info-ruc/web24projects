from torch.utils.data import DataLoader
from plotFigure import ConfusionMatrix,plot_curve
import torch
import numpy as np
from loss import loss_func3
from preprocess import test_dataset
from preprocess import train_dataset

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

device = torch.device("cuda")
test_loss_list = []
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
print("start testing......")
confusion = ConfusionMatrix(num_classes=5, labels=['level 0', 'level 1', 'level 2', 'level 3', 'level 4'])

from resnext_model import model
model.load_state_dict(torch.load("./parameters_resnext_MSELoss.pkl"))
model.eval()
model=model.to(device)
score_list = []  # 存储预测得分
label_list = []  # 存储真实标签

with torch.no_grad():
    test_loss = 0
    # ce_test_loss=0
    # qk_test_loss=0
    for i, data in enumerate(test_dataLoader):
        inputs, labels = data
        labels = labels.to(torch.float)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        outputs = softmax(outputs)
        # score_tmp = softmax(outputs)
        # score_list.extend(score_tmp.detach().cpu().numpy())
        # label_list.extend(labels.cpu().numpy())
        score_tensor = torch.tensor([0., 1., 2., 3., 4.]).cuda()

        outputs = outputs.mm(score_tensor.view(-1, 1)).squeeze(dim=1)
        # outputs1 = softmax(outputs)

        print('outputs',outputs)
        print('labels',labels)

        loss = loss_func3(outputs,labels)
        # loss = loss_func1(outputs1, labels) + loss_func2(outputs, labels)
        # qkloss = loss_func1(outputs1,labels)
        # celoss = loss_func2(outputs,labels)
        batch_size = inputs.size(0)
        # print("step={}; loss={};".format(i,loss.item()))
        test_loss += float(loss.item())
        # qk_test_loss += float(qkloss.item())
        # ce_test_loss += float(celoss.item())
        # ret, predictions = torch.max(outputs.data, 1)
        predictions = torch.round(outputs)
        # confusion_matrix
        confusion.update(predictions.to(torch.int).cpu().numpy(), labels.to(torch.int).cpu().numpy())

    # plot_curve(score_list,label_list)

    test_loss = test_loss / len(test_dataLoader)
    # qk_test_loss = qk_test_loss / len(test_dataLoader)
    # ce_test_loss = ce_test_loss / len(test_dataLoader)
    # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
    print("the test loss is: {}".format(test_loss))
    # print("the ce_test_loss is: {}".format(ce_test_loss))
    # print("the qk_test_loss is: {}".format(qk_test_loss))
    confusion.plot()
    # confusion.summary()