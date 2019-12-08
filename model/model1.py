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
