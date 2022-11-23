import torch.nn as nn

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

class BNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.bn2 = nn.BatchNorm1d(self.in_dim)
        self.bn2.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        self.classifier2 = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

    def forward(self, x,x2):
        feature = self.bn(x)
        feature2 = self.bn2(x2)
        if not self.training:
            feature = [feature, feature2]
            return feature, None
        cls_score = self.classifier(feature)
        cls_score2 = self.classifier(feature2)
        feature=[feature,feature2]
        cls_score=[cls_score,cls_score2]
        return feature, cls_score

class ABNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(ABNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)


        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)


        self.bn.apply(weights_init_kaiming)

        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        feature = self.bn(x)
        if not self.training:
            return feature, None
        cls_score = self.classifier(feature)
        return feature, cls_score