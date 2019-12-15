import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear0 = nn.Linear(11, 55)
        self.linear1 = nn.Linear(55, 55)
        self.linear2 = nn.Linear(55, 55)
        self.linear3 = nn.Linear(55, 55)
        self.linear4 = nn.Linear(55, 4)

    def forward(self, input):
        # m = nn.Sigmoid()
        m = nn.LeakyReLU(0.1)
        out0 = m(self.linear0(input))
        out0 = F.dropout(out0, 0.2, training=self.training)
        out1 = m(self.linear1(out0))
        out1 = F.dropout(out1, 0.2, training=self.training)
        out2 = m(self.linear2(out1))
        out2 = F.dropout(out2, 0.2, training=self.training)
        out3 = m(self.linear3(out2))
        out3 = F.dropout(out3, 0.2, training=self.training)
        out4 = self.linear4(out3)

        return out4



class ReverseKlLoss(nn.Module):
    def __init__(self):
        super(ReverseKlLoss, self).__init__()

    def forward(self,out: torch.Tensor, target: torch.Tensor):
        s = nn.Softmax(dim=1)
        predict_dist = s(out)
        target_dist = self.target_to_dist(target)
        loss = 0.0
        # print(predict_dist)
        # print(target_dist)
        for i in range(len(out)):
            for j in range(len(out[0])):
                loss += predict_dist[i,j] * torch.log((predict_dist[i,j]+1e-10)/(target_dist[i,j]+1e-10))
        return loss

    @staticmethod
    def target_to_dist(target: torch.Tensor) -> torch.Tensor:
        l = []
        for i in range(len(target)):
            t = target[i]
            if t == 0.0:
                l += [[0.9, 0.1, 0.0, 0.0]]
            elif t == 1.0:
                l += [[0.1, 0.9, 0.0, 0.0]]
            elif t == 2.0:
                l += [[0.0, 0.0, 0.9, 0.1]]
            else:
                l += [[0.0, 0.0, 0.1, 0.9]]
        return torch.Tensor(l)


