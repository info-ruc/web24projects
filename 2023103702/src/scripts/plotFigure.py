from prettytable import PrettyTable
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score
from itertools import cycle

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is: ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = round((po - pe) / (1 - pe), 3)
        print("the model kappa is ", kappa)

        # quadratic weighted kappa
        weighted_error = 0
        weighted_baseline_error = 0
        num_classes = len(self.matrix[0])
        for i in range(num_classes):
            for j in range(num_classes):
                row = np.sum(self.matrix[i, :])
                col = np.sum(self.matrix[:, j])
                weighted_error += ((i - j) ** 2) * self.matrix[i][j] * n
                weighted_baseline_error += ((i - j) ** 2) * row * col

        print('weighted_error ', weighted_error)
        print('weighted_baseline_error ', weighted_baseline_error)

        quadratic_weighted_kappa = round(1 - (weighted_error / weighted_baseline_error), 3)
        print("the quadratic_weighted_kappa of the model is ", quadratic_weighted_kappa)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确率、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0

            table.add_row([self.labels[i], Precision, Recall, Specificity])

        print(table)
        return str(acc),quadratic_weighted_kappa

    def plot(self):  # 绘制混淆矩阵
        plt.rc('font', family='Times New Roman')
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45,fontsize=14)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels,fontsize=14)
        # 显示colorbar
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.xlabel('True Labels',fontsize=16)
        plt.ylabel('Predicted Labels',fontsize=16)
        # plt.title('Confusion matrix (acc=' + self.summary()[0] + ')',fontsize=18)
        plt.title('Confusion Matrix(ResNet50)',fontsize=16)

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center',
                         color='white' if info > thresh else 'black', fontsize=14)
                # plt.text(x,y,info,verticalalignment='center',horizontalalignment='center')
        plt.tight_layout()
        # plt.savefig("clear3.png", dpi=900)
        plt.savefig("clear6.png", dpi=900)
        # plt.savefig("clear6.svg", dpi=600, format="svg")
        plt.show()
        return self.summary()[1]

def plot_curve(score_list,label_list):
    plt.rc('font', family='Times New Roman')
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], 5)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应从fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    # for i in range(5):
    #     fpr_dict[i],tpr_dict[i],_ = roc_curve(label_onehot[:,i],score_array[:,i])
    #     roc_auc_dict[i]=auc(fpr_dict[i],tpr_dict[i])
    #     print(i,' auc: ',roc_auc_dict[i])

    fpr_dict[0], tpr_dict[0], _ = roc_curve(label_onehot[:, 0], score_array[:, 0])
    roc_auc_dict[0] = auc(fpr_dict[0], tpr_dict[0])
    print(0, ' auc: ', roc_auc_dict[0])

    fpr_dict[1], tpr_dict[1], _ = roc_curve(np.sum(label_onehot[:, 1:5], axis=1), np.sum(score_array[:, 1:5], axis=1))
    roc_auc_dict[1] = auc(fpr_dict[1], tpr_dict[1])
    print('1-3', ' auc: ', roc_auc_dict[1])

    my_fpr_dict , my_tpr_dict , _ = roc_curve(np.sum(label_onehot[:, 0:2], axis=1),np.sum(score_array[:, 0:2],axis=1))
    my_roc_auc_dict = auc(my_fpr_dict,my_tpr_dict)
    # print("my_fpr_dict:",my_fpr_dict)
    # print("my_tpr_dict:",my_tpr_dict)
    # print("_:",_)
    # print("my_roc_auc_dict:",my_roc_auc_dict)


    np.save('./dict/FasterRCNNbackbone_oriImage_my_fpr_dict.npy',my_fpr_dict)
    np.save('./dict/FasterRCNNbackbone_oriImage_my_tpr_dict.npy',my_tpr_dict)
    np.save('./dict/FasterRCNNbackbone_oriImage_my_roc_auc_dict.npy',my_roc_auc_dict)
    # print('dict1 = ',dict_load1.item())
    # print('dict2 = ',dict_load2.item())
    # print('dict = ',dict_load.item())


    # fpr_dict[1],tpr_dict[1],_ = roc_curve(label_onehot[:,1:5],score_array[:,1:5])
    # roc_auc_dict[1]=auc(fpr_dict[1],tpr_dict[1])
    # print('1-4',' auc: ',roc_auc_dict[1])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'pink'])
    # for i,color in zip(range(5),colors):
    #     plt.plot(fpr_dict[i],tpr_dict[i],color=color,lw=2,
    #              label='ROC curve of class {0} (area = {1:0.3f})'
    #              ''.format(i,roc_auc_dict[i]))

    plt.plot(fpr_dict[0], tpr_dict[0], color='aqua', lw=2,
             label='ROC curve of class {0} (area = {1:0.3f})'
                   ''.format(0, roc_auc_dict[0]))

    plt.plot(fpr_dict[1], tpr_dict[1], color='green', lw=2,
             label='ROC curve of class {0} (area = {1:0.3f})'
                   ''.format('1-3', roc_auc_dict[1]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title('ROC Curve',fontsize=16)
    plt.legend(loc="lower right",fontsize=14)
    # plt.savefig("clear4.svg", dpi=600, format="svg")
    plt.savefig("clear7.svg", dpi=600, format="svg")
    plt.show()
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    # for i in range(5):
    #     precision_dict[i],recall_dict[i],_ = precision_recall_curve(label_onehot[:,i],score_array[:,i])
    #     average_precision_dict[i]=average_precision_score(label_onehot[:,i],score_array[:,i])
    #     print(precision_dict[i].shape,recall_dict[i].shape,average_precision_dict[i])
    precision_dict[0], recall_dict[0], _ = precision_recall_curve(label_onehot[:, 0], score_array[:, 0])
    average_precision_dict[0] = average_precision_score(label_onehot[:, 0], score_array[:, 0])
    print(precision_dict[0].shape, recall_dict[0].shape, average_precision_dict[0])

    precision_dict[1], recall_dict[1], _ = precision_recall_curve(np.sum(label_onehot[:, 1:5], axis=1),
                                                                  np.sum(score_array[:, 1:5], axis=1))
    average_precision_dict[1] = average_precision_score(np.sum(label_onehot[:, 1:5], axis=1),
                                                        np.sum(score_array[:, 1:5], axis=1))
    print(precision_dict[1].shape, recall_dict[1].shape, average_precision_dict[1])
    print(precision_dict[1])
    print(recall_dict[1])

    # for i,color in zip(range(5),colors):
    #     plt.plot(recall_dict[i],precision_dict[i],color=color,lw=2,
    #              label='PR curve of class {0} (AP = {1:0.3f})'
    #              ''.format(i,average_precision_dict[i]))

    plt.plot(recall_dict[0], precision_dict[0], color='pink', lw=2,
             label='PR curve of class {0} (AP = {1:0.3f})'
                   ''.format(0, average_precision_dict[0]))

    plt.plot(recall_dict[1], precision_dict[1], color='cornflowerblue', lw=2,
             label='PR curve of class {0} (AP = {1:0.3f})'
                   ''.format('1-3', average_precision_dict[1]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Recall',fontsize=16)
    plt.ylabel('Precision',fontsize=16)
    plt.title('PR Curve',fontsize=16)
    plt.legend(loc="lower right",fontsize=14)
    # plt.savefig("clear5.svg", dpi=600, format="svg")
    plt.savefig("clear8.svg", dpi=600, format="svg")
    plt.show()