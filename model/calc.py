import torch
import numpy as np
import pandas as pd
import sys
import torch.nn as nn
from typing import List


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
        correct = ((self.predict_cat == cat_num) & (self.target == cat_num)).sum().cpu().numpy()

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

    def quadratic_weighted_kappa(self) -> float:
        cat_num = 4
        try:
            predict_cat = self.predict_cat.cpu().numpy()
            target = self.target.cpu().numpy()
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
        if denominator != 0.0:
            kappa = 1.0 - numerator / denominator
        else:
            kappa = 1.0

        return kappa



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


def quadratic_weighted_kappa(predict_cat: torch.Tensor, target: torch.Tensor, cat_num: int) -> float:
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


# -- nelder mead
# -- maximization quadratic_weighted_kappa
class NelderMead():
    def __init__(self, out: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0,
                 beta: float = 0.5, delta: float = 0.5):
        s = nn.Softmax(dim=1)
        self.out = out
        self.predict = s(out)
        self.target = target
        self.predict_val = self.expected_value()
        self.alpha = alpha  # -- reglection param
        self.beta = beta  # -- contraction param
        self.gamma = gamma  # -- expansion param
        self.delta = delta  # -- shrink_contraction param

    def optimize(self):
        thresholds = self.init_thresholds()
        for k in range(100):
            thresholds = self.order_thresholds(thresholds)
            kappas = self.order_kappa(thresholds)
            # print(kappas)
            worst_threshold = thresholds[0]
            center_threshold = (thresholds[1] + thresholds[2] + thresholds[3]) / 3.0
            reflect_threshold = self.reflection(center_threshold, worst_threshold)
            kappa = self.quadratic_weighted_kappa(self.predict_cat(reflect_threshold), self.target)
            if kappa > kappas[-1]:
                # -- expansion
                threshold = self.expansion(center_threshold, reflect_threshold)
                thresholds = thresholds[1:] + [threshold]
            elif kappa < kappas[1]:
                # contraction
                threshold = self.contraction(center_threshold, worst_threshold)
                kappa = self.quadratic_weighted_kappa(self.predict_cat(threshold), self.target)
                if kappa < self.quadratic_weighted_kappa(self.predict_cat(center_threshold), self.target):
                    # -- replace worst -> center
                    thresholds = thresholds[1:] + [threshold]
                else:
                    # -- shrink_contraction
                    thresholds[:-1] = self.shrink_contraction(thresholds[-1], thresholds[:-1])
            else:
                # -- replace worst -> reflect
                thresholds[0] = reflect_threshold

        thresholds = self.order_thresholds(thresholds)
        return thresholds[-1]

    def init_thresholds(self):
        vec0 = torch.Tensor([0.75, 1.5, 2.25])
        vec1 = torch.Tensor([0.5, 1.5, 2.5])
        vec2 = torch.Tensor([0.2, 1.0, 2.0])
        vec3 = torch.Tensor([0.6, 1.2, 1.8])
        vecs = [vec0, vec1, vec2, vec3]
        return vecs

    def expected_value(self) -> torch.Tensor:
        predict_val = torch.zeros(len(self.predict))
        for i in range(len(self.predict)):
            predict_val[i] = self.predict[i, 0] * 0.0 \
                             + self.predict[i, 1] * 1.0 \
                             + self.predict[i, 2] * 2.0 \
                             + self.predict[i, 3] * 3.0
        return predict_val

    def predict_cat(self, thresholds: torch.Tensor) -> torch.Tensor:
        predict_cat = self.predict_val
        for i in range(len(self.predict_val)):
            val = self.predict_val[i]
            if val <= thresholds[0]:
                predict_cat[i] = 0.0
            elif val <= thresholds[1]:
                predict_cat[i] = 1.0
            elif val <= thresholds[2]:
                predict_cat[i] = 2.0
            else:
                predict_cat[i] = 3.0
        return predict_cat

    def order_thresholds(self, thresholds: List[torch.Tensor]) -> List[torch.Tensor]:
        l = []
        for i in range(len(thresholds)):
            predict_cat = self.predict_cat(thresholds[i])
            kappa = self.quadratic_weighted_kappa(predict_cat, self.target)
            l += [[kappa, thresholds[i]]]
        l.sort(key=lambda x: x[0])
        order_thresholds = []
        for i in range(len(l)):
            order_thresholds += [l[i][1]]
        return order_thresholds

    def order_kappa(self, thresholds: List[torch.Tensor]) -> List[float]:
        l = []
        for i in range(len(thresholds)):
            predict_cat = self.predict_cat(thresholds[i])
            # print(predict_cat)
            # print(self.target)
            kappa = self.quadratic_weighted_kappa(predict_cat, self.target)
            l += [[kappa, thresholds[i]]]
        l.sort(key=lambda x: x[0])
        kappas = []
        for i in range(len(l)):
            kappas += [l[i][0]]
        return kappas

    def reflection(self, center_vec: torch.Tensor, worst_vec: torch.Tensor) -> torch.Tensor:
        return center_vec + self.alpha * (center_vec - worst_vec)

    def expansion(self, center_vec: torch.Tensor, reflect_vec: torch.Tensor) -> torch.Tensor:
        return center_vec + self.gamma * (reflect_vec - center_vec)

    def contraction(self, center_vec: torch.Tensor, worst_vec: torch.Tensor) -> torch.Tensor:
        return center_vec + self.beta * (worst_vec - center_vec)

    def shrink_contraction(self, best_vec: torch.Tensor, other_vecs: List[torch.Tensor]) -> torch.Tensor:
        result = other_vecs
        for i in range(len(other_vecs)):
            result[i] = best_vec + self.delta * (other_vecs[i] - best_vec)
        return result

    @staticmethod
    def quadratic_weighted_kappa(predict_cat: torch.Tensor, target: torch.Tensor) -> float:
        cat_num = 4
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
        # print(predict_cat)
        # print(target)
        # print(o)
        # print(e)
        # print(w)
        for i in range(cat_num):
            for j in range(cat_num):
                numerator += w[i, j] * o[i, j]
                denominator += w[i, j] * e[i, j]
        if denominator != 0.0:
            kappa = 1.0 - numerator / denominator
        else:
            kappa = 1.0

        return kappa


class EvaluateCatWithThreshold():
    def __init__(self, out: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor):
        s = nn.Softmax(dim=1)
        self.out = out
        self.threshold = threshold
        self.predict = s(out)
        self.predict_val = self.expected_value()
        self.predict_cat = self.predict_cat().long().cpu()
        self.target = target.cpu()

    def expected_value(self) -> torch.Tensor:
        predict_val = torch.zeros(len(self.predict))
        for i in range(len(self.predict)):
            predict_val[i] = self.predict[i, 0] * 0.0 \
                             + self.predict[i, 1] * 1.0 \
                             + self.predict[i, 2] * 2.0 \
                             + self.predict[i, 3] * 3.0
        return predict_val

    def predict_cat(self) -> torch.Tensor:
        predict_cat = self.predict_val
        for i in range(len(self.predict_val)):
            val = self.predict_val[i]
            if val <= self.threshold[0]:
                predict_cat[i] = 0.0
            elif val <= self.threshold[1]:
                predict_cat[i] = 1.0
            elif val <= self.threshold[2]:
                predict_cat[i] = 2.0
            else:
                predict_cat[i] = 3.0
        return predict_cat

    def total_num(self):
        return len(self.target)

    def total_correct_num(self):
        return (self.predict_cat.long().cpu() == self.target.cpu()).sum().cpu().numpy()

    def acc_total(self):
        return self.total_correct_num() / self.total_num()

    def acc_by_cat(self, cat_num: int):
        total = (self.target == cat_num).sum().cpu().numpy()
        correct = ((self.predict_cat == cat_num) & (self.target == cat_num)).sum().cpu().numpy()

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

    def quadratic_weighted_kappa(self) -> float:
        cat_num = 4
        try:
            predict_cat = self.predict_cat.cpu().numpy()
            target = self.target.cpu().numpy()
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
        if denominator != 0.0:
            kappa = 1.0 - numerator / denominator
        else:
            kappa = 1.0

        return kappa

