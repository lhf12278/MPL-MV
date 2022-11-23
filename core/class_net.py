import torch
from torch import nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ID_Market_Net2(nn.Module):
    def __init__(self):
        super(ID_Market_Net2, self).__init__()
        self.layer1 = self._make_layer(2048,1848)
        self.layer2 = self._make_layer(1848, 1648)
        self.layer3 = self._make_layer(1648, 1548)
        self.fc = nn.Linear(1548,1502)
        self.fc.apply(weights_init_kaiming)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
    def _make_layer(self,in_nc,out_nc):
        block = [nn.Conv2d(in_nc,out_nc,kernel_size=3,stride=1,padding=1),
                 nn.BatchNorm2d(out_nc),
                 nn.LeakyReLU(0.2,True)]
        return nn.Sequential(*block)



    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.drop(output)
        output = self.gap(output)
        output = output.squeeze()
        output = self.fc(output)

        return output
class ID_Duke_Net(nn.Module):
    def __init__(self):
        super(ID_Duke_Net, self).__init__()
        self.classifier = self._make_layer(2048,1404)
        self.dropout = nn.Dropout(0.5)
        # self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, out_nc, bias=False),
                 # nn.BatchNorm1d(out_nc),
                 # nn.ReLU()
        ]
        return nn.Sequential(*block)
    def forward(self, feature):
        feature = self.dropout(feature)
        cls_score = self.classifier(feature)
        return cls_score


class ID_Msmt_Net(nn.Module):
    def __init__(self):
        super(ID_Msmt_Net, self).__init__()
        self.classifier = self._make_layer(2048,2802)
        self.dropout = nn.Dropout(0.5)
        # self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
    def _make_layer(self, in_nc, out_nc):
        block = [
            nn.Linear(in_nc, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, out_nc, bias=False),
            # nn.BatchNorm1d(out_nc),
            # nn.ReLU()
        ]
        return nn.Sequential(*block)
    def forward(self, feature):
        feature = self.dropout(feature)
        cls_score = self.classifier(feature)
        return cls_score



class ID_Market_Net(nn.Module):
    def __init__(self):
        super(ID_Market_Net, self).__init__()
        self.classifier = self._make_layer(2048,1502)

        # self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.dropout = nn.Dropout(0.5)
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.ReLU(),
                 nn.Linear(1000, out_nc, bias=False),
                 # nn.BatchNorm1d(out_nc),
                 # nn.ReLU()
        ]
        return nn.Sequential(*block)
    def forward(self, feature):
        feature = self.dropout(feature)
        cls_score = self.classifier(feature)
        return cls_score  # [batch,15]

class Carmera_Net(nn.Module):
    def __init__(self):
        super(Carmera_Net, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=2048, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(5):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)

            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, 24)
        self.dropout = nn.Dropout(0.5)
    def forward(self, latent):
        latent = latent.unsqueeze(2)
        # latent = self.dropout(latent)
        hidden = self.first_layer(latent)
        for i in range(5):
            hidden = self.layers[i](hidden)
        hidden = hidden.squeeze(2)
        domain_clss = self.Liner(hidden)
        return domain_clss  # [batch,15]

class Carmera_Net2222(nn.Module):
    def __init__(self):
        super(Carmera_Net2222, self).__init__()
        self.classifier = self._make_layer(2048,15)

        # self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 1024, bias=False),
                 nn.BatchNorm1d(1024),
                 nn.ReLU(),
                 nn.Linear(1024, 512, bias=False),
                 nn.BatchNorm1d(512),
                 nn.ReLU(),
                 nn.Linear(512, 256, bias=False),
                 nn.BatchNorm1d(256),
                 nn.ReLU(),
                 nn.Linear(256, 128, bias=False),
                 nn.BatchNorm1d(128),
                 nn.ReLU(),
                 nn.Linear(128, 64, bias=False),
                 nn.BatchNorm1d(64),
                 nn.ReLU(),
                 nn.Linear(64, 32, bias=False),
                 nn.BatchNorm1d(32),
                 nn.ReLU(),
                 nn.Linear(32, out_nc, bias=False),
                 # nn.BatchNorm1d(out_nc),
                 # nn.ReLU()
        ]
        return nn.Sequential(*block)
    def forward(self, feature):
        cls_score = self.classifier(feature)
        return cls_score  #


class ID_Market_Net23(nn.Module):
    def __init__(self):
        super(ID_Market_Net23, self).__init__()
        self.classifier = self._make_layer(2048,1502)

        # self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.LeakyReLU(0.2, True),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.LeakyReLU(0.2, True),
                 nn.Linear(1000, 1000, bias=False),
                 nn.BatchNorm1d(1000),
                 nn.LeakyReLU(0.2, True),
                 nn.Linear(in_nc, out_nc, bias=False),
                     nn.BatchNorm1d(out_nc),
                     nn.LeakyReLU(0.2,True)]
        return nn.Sequential(*block)
    def forward(self, feature):
        cls_score = self.classifier(feature)
        return cls_score  # [batch,15]


class ID_Market_Net22(nn.Module):
    def __init__(self):
        super(ID_Market_Net22, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=2048, out_channels=1000, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(1000),
                nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Linear(1000, 1502, bias=False),
            nn.BatchNorm1d(1502),
            nn.ReLU(inplace=True))

    def forward(self, latent):
        # latent = self.gap(latent)
        latent = latent.unsqueeze(2)
        x = self.conv1(latent)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = x.squeeze()
        cls_score = self.conv7(x)
        # fmeichu

        # cls_score = self.Liner(hidden)
        return cls_score  # [batch,15]


class DomainNet(nn.Module):
    def __init__(self):
        super(DomainNet, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=2048, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(5):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, 4)
        self.gap = nn.AdaptiveMaxPool2d((3,1))
    def forward(self, latent):
        # latent = self.gap(latent)
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(5):
            hidden = self.layers[i](hidden)
        hidden = hidden.squeeze(2)
        domain_clss = self.Liner(hidden)
        return domain_clss  # [batch,15]
