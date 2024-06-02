# from ignite.engine import *
# from ignite.handlers import *
# from ignite.metrics import *
# from ignite.utils import *
# from ignite.contrib.metrics.regression import *
# from ignite.contrib.metrics import *
import numpy as np
import torch
import torch.nn as nn


def kappa_loss(score, target, weights, classes=4, e=1e-15):
    score = score / (e + score.sum(1)).view(-1, 1)
    target_onehot = torch.zeros(target.size(0), classes).cuda().scatter_(1, target.data.view(-1, 1), 1)
    hist_rater_pred = score.sum(0)
    hist_rater_target = target_onehot.sum(0)
    conf_mat = torch.t(score).mm(target_onehot)
    nom = torch.mul(weights, conf_mat)
    denom = torch.mm(hist_rater_pred.view(-1, 1), hist_rater_target.view(1, -1)) / float(score.shape[0])
    denom = torch.mul(weights, denom)
    kappa = -(1 - torch.div(nom.sum(), denom.sum()))
    kappa = (kappa + 1.) / 2.
    return kappa


class KappaLoss(nn.Module):
    def __init__(self, classes=4, kappa_weight=0.5):
        super(KappaLoss, self).__init__()
        rating_mat = torch.FloatTensor(np.arange(classes)).view(-1, 1).repeat(1, classes)
        ratings_squared = (rating_mat - torch.t(rating_mat)) ** 2
        self.weights = torch.div(ratings_squared, (classes - 1) ** 2).cuda()
        self.classes = classes
        self.kappa_weight = kappa_weight

    def forward(self, pred, target):
        loss = kappa_loss(pred, target, self.weights, self.classes)
        return loss

    
loss_func1 = KappaLoss()
loss_func2 = nn.CrossEntropyLoss()
loss_func3 = nn.MSELoss()