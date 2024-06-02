import matplotlib.pyplot as plt
import numpy as np
resnet18_fpr_dict, resnet18_tpr_dict, \
             resnet18_roc_auc_dict = np.load('./dict/resnet18_my_fpr_dict.npy'), \
                                     np.load('./dict/resnet18_my_tpr_dict.npy'),\
                                     np.load('./dict/resnet18_my_roc_auc_dict.npy')

plt.plot(resnet18_fpr_dict, resnet18_tpr_dict, '-', lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', resnet18_roc_auc_dict))


resnet50_fpr_dict, resnet50_tpr_dict, \
resnet50_roc_auc_dict = np.load('./dict/resnet50_my_fpr_dict.npy'), \
                        np.load('./dict/resnet50_my_tpr_dict.npy'), \
                        np.load('./dict/resnet50_my_roc_auc_dict.npy')

plt.plot(resnet50_fpr_dict, resnet50_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', resnet50_roc_auc_dict))


resnet101_fpr_dict, resnet101_tpr_dict, \
resnet101_roc_auc_dict = np.load('./dict/resnet101_my_fpr_dict.npy'), \
                        np.load('./dict/resnet101_my_tpr_dict.npy'), \
                        np.load('./dict/resnet101_my_roc_auc_dict.npy')

plt.plot(resnet101_fpr_dict, resnet101_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', resnet101_roc_auc_dict))


resnext_fpr_dict, resnext_tpr_dict, \
resnext_roc_auc_dict = np.load('./dict/resnext_my_fpr_dict.npy'), \
                        np.load('./dict/resnext_my_tpr_dict.npy'), \
                        np.load('./dict/resnext_my_roc_auc_dict.npy')

plt.plot(resnext_fpr_dict, resnext_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', resnext_roc_auc_dict))


mobilenetv3_fpr_dict, mobilenetv3_tpr_dict, \
mobilenetv3_roc_auc_dict = np.load('./dict/mobilenetv3_my_fpr_dict.npy'), \
                        np.load('./dict/mobilenetv3_my_tpr_dict.npy'), \
                        np.load('./dict/mobilenetv3_my_roc_auc_dict.npy')

plt.plot(mobilenetv3_fpr_dict, mobilenetv3_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', mobilenetv3_roc_auc_dict))


shufflenet_fpr_dict, shufflenet_tpr_dict, \
shufflenet_roc_auc_dict = np.load('./dict/shufflenet_my_fpr_dict.npy'), \
                        np.load('./dict/shufflenet_my_tpr_dict.npy'), \
                        np.load('./dict/shufflenet_my_roc_auc_dict.npy')

plt.plot(shufflenet_fpr_dict, shufflenet_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', shufflenet_roc_auc_dict))


inceptionv3_fpr_dict, inceptionv3_tpr_dict, \
inceptionv3_roc_auc_dict = np.load('./dict/inceptionv3_my_fpr_dict.npy'), \
                        np.load('./dict/inceptionv3_my_tpr_dict.npy'), \
                        np.load('./dict/inceptionv3_my_roc_auc_dict.npy')

plt.plot(inceptionv3_fpr_dict, inceptionv3_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', inceptionv3_roc_auc_dict))



convnext_fpr_dict, convnext_tpr_dict, \
convnext_roc_auc_dict = np.load('./dict/convnext_my_fpr_dict.npy'), \
                        np.load('./dict/convnext_my_tpr_dict.npy'), \
                        np.load('./dict/convnext_my_roc_auc_dict.npy')

plt.plot(convnext_fpr_dict, convnext_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', convnext_roc_auc_dict))



swintransformer_fpr_dict, swintransformer_tpr_dict, \
swintransformer_roc_auc_dict = np.load('./dict/swintransformer_my_fpr_dict.npy'), \
                        np.load('./dict/swintransformer_my_tpr_dict.npy'), \
                        np.load('./dict/swintransformer_my_roc_auc_dict.npy')

plt.plot(swintransformer_fpr_dict, swintransformer_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', swintransformer_roc_auc_dict))



twostreamResnet18_fpr_dict, twostreamResnet18_tpr_dict, \
twostreamResnet18_roc_auc_dict = np.load('./dict/twostreamResnet18_my_fpr_dict.npy'), \
                        np.load('./dict/twostreamResnet18_my_tpr_dict.npy'), \
                        np.load('./dict/twostreamResnet18_my_roc_auc_dict.npy')

plt.plot(twostreamResnet18_fpr_dict, twostreamResnet18_tpr_dict,lw=2,
         label='ROC curve of class {0} (area = {1:0.3f})'
               ''.format('0-1', twostreamResnet18_roc_auc_dict))



plt.show()