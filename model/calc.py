import torch
import numpy as np
import pandas as pd
import sys
import torch.nn as nn


class EvaluateCat():
    def __init__(self, out: torch.Tensor, target: torch.Tensor):
        s = nn.Softmax(dim=1)
        self.out = out
        self.predict = s(out)
        self.predict_cat = torch.argmax(self.predict, dim=1)
        self.target = target

    def total_num(self):
        return len(self.target)

    def total_correct_num(self):
        return (self.predict_cat == self.target).sum().cpu().numpy()

    def acc_total(self):
        return self.total_correct_num() / self.total_num()

    def acc_by_cat(self, cat_num: int):
        total = (self.target == cat_num).sum().cpu().numpy()
        correct = ((self.predict_cat == cat_num)&(self.target == cat_num)).sum().cpu().numpy()

        return correct / total

    def acc_ave(self):
        acc0 = self.acc_by_cat(cat_num=0)
        acc1 = self.acc_by_cat(cat_num=1)
        acc2 = self.acc_by_cat(cat_num=2)
        acc3 = self.acc_by_cat(cat_num=3)

        return 0.25 * (acc0 + acc1 + acc2 + acc3)

    def diversity(self):
        return len(torch.unique(self.out[:, 0])) / self.total_num()

    @staticmethod
    def _calc_AR(predict: torch.Tensor, target: torch.Tensor) -> float:
        """
        :param predict: sftmax value shape=[samples, 2]
        :param target: value shape=[samples,1]
        :return: AR value
        """
        predict_data = predict.cpu().numpy()
        target_data = target.cpu().numpy()
        # try:
        gg = list(np.concatenate((predict_data, target_data), axis=1))
        h = np.array(sorted(gg, key=lambda x: x[1], reverse=True))
        # x[1]:predicted PD value
        var = len(h)
        hol = np.sum(h[:, 2])
        # h[:,2] : default flag
        y = 0
        i = 0
        b = 0
        while y < hol:
            if h[i][2] == 1:
                y = y + 1
                x = 0
            else:
                x = 1
            b = b + (hol - y) * x
            i = i + 1
        ab = hol * (var - hol) / 2.0
        a = ab - b
        ar = a / ab
        # except Exception:
        #     print("error in calcAR")
        #     sys.exit()
        return ar

def calc_AR(predict: torch.Tensor, target: torch.Tensor) -> float:
    """
    :param predict: sftmax value shape=[samples, 2]
    :param target: value shape=[samples,1]
    :return: AR value
    """
    predict_data = predict.cpu().numpy()
    target_data = target.cpu().numpy()
    # try:
    gg = list(np.concatenate((predict_data, target_data), axis=1))
    h = np.array(sorted(gg, key=lambda x: x[1], reverse=True))
    # x[1]:predicted PD value
    var = len(h)
    hol = np.sum(h[:, 2])
    # h[:,2] : default flag
    y = 0
    i = 0
    b = 0
    while y < hol:
        if h[i][2] == 1:
            y = y + 1
            x = 0
        else:
            x = 1
        b = b + (hol - y) * x
        i = i + 1
    ab = hol * (var - hol) / 2.0
    a = ab - b
    ar = a / ab
    # except Exception:
    #     print("error in calcAR")
    #     sys.exit()
    return ar


def calc_quadratic_weighted_kappa(predict_cat: torch.Tensor, target: torch.Tensor, cat_num: int) -> float:
    try:
        predict_cat = predict_cat.cpu().numpy()
        target = target.cpu().numpy()
    except:
        pass

    o = np.zeros([cat_num, cat_num])
    e = np.zeros([cat_num, cat_num])
    w = np.zeros([cat_num, cat_num])
    samples = len(target)
    for i in range(cat_num):
        for j in range(cat_num):
            w[i, j] = (i - j) ** 2
            o[i, j] = ((predict_cat == i) & (target == j)).sum()
            e[i, j] = (predict_cat == i).sum() * (target == j).sum() / samples

    numerator = 0.0
    denominator = 0.0
    for i in range(cat_num):
        for j in range(cat_num):
            numerator += w[i, j] * o[i, j]
            denominator += w[i, j] * e[i, j]

    kappa = 1.0 - numerator / denominator

    return kappa


