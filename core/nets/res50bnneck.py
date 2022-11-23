import torch
import torch.nn as nn
import torchvision
import random
from .bnneck import BNClassifier,ABNClassifier,weights_init_kaiming
from .AttentionNet import SeDropLayer
class Res50BNNeck(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Res50BNNeck, self).__init__()

        self.class_num = class_num
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        # cnn backbone
        self.resnet_conv_one_1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2)
        self.resnet_conv_one_2 = nn.Sequential(
            resnet.layer3, resnet.layer4)

        self.resnet_conv_two_1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1, resnet.layer2)
        self.resnet_conv_two_2 = nn.Sequential(
            resnet.layer3, resnet.layer4)
        self.mix_conv = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.se_drop = SeDropLayer(2048)
        # classifier
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

        self.classifier = BNClassifier(2048, self.class_num)
        self.classifier_baseline = ABNClassifier(2048, self.class_num)


    def forward(self, x,x_middle=0,baseline=False,mix=False):
        if baseline:
            features = self.gap(self.resnet_conv(x)).squeeze(dim=2).squeeze(dim=2)
            bned_features, cls_score = self.classifier_baseline(features)

            if self.training:
                return bned_features, cls_score
            else:
                return bned_features
        else:
            if mix:
                list = [0.25,0.5,0.75]
                a = random.sample(list,1)[0]
                mix_feature = self.mix_s_t(x_middle[0],x_middle[1],a)
                return mix_feature,a
            else:
                #DISFL
                x_max_middle,feature_max = self.foward_one(x)
                #MPL
                feature_avg = self.foward_two(x)

                merge_feature = feature_max + feature_avg
                features, cls_score = self.classifier(feature_max, feature_avg)
                bned_features, merge_cls_score = self.classifier_baseline(merge_feature)

                if self.training:
                    return bned_features, cls_score,merge_cls_score,features,x_max_middle
                else:
                    return bned_features

    def foward_one(self,x):
        '''
        :param x: the input image
        :return: x ,the output feature. x_middle the middle feature map
        '''
        x_middle = self.resnet_conv_one_1(x)
        x = self.resnet_conv_one_2(x_middle)
        x = self.max_pool(x).squeeze(dim=2).squeeze(dim=2)
        return x_middle,x

    def foward_two(self,x):
        '''
        :param x: the input image
        :return: x the output feature.
        '''
        x_middle = self.resnet_conv_two_1(x)
        x = self.resnet_conv_two_2(x_middle)
        x = self.gap(x).squeeze(dim=2).squeeze(dim=2)
        return x

    def mix_s_t(self,s,t,a):
        x = a*s + (1-a)*t
        y = self.resnet_conv_one_2(x)
        y = self.max_pool(y).squeeze(dim=2).squeeze(dim=2)
        y = self.bn(y)
        return y

