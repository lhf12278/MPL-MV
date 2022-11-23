import torch.nn as nn
import torch
import random

class SeDropLayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SeDropLayer,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x,is_test=False):
        list = [0,0,0,0]
        num = random.sample(list,1)[0]
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        # if is_test:
        #     y = self.fc(y).view(b, c, 1, 1)
        #     return x * y.expand_as(x)
        # else:
        #     if(num==0):
        #         y = self.fc(y).view(b,c,1,1)
        #         return x * y.expand_as(x)
        #     elif(num==1):
        #         ones = torch.ones(y.size()).cuda()
        #         y = ones - y
        #         y = self.fc(y).view(b,c,1,1)
        #         return x * y.expand_as(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)