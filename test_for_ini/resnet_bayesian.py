import torch
import torch.nn as nn
import torch.nn.init as init
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

__all__ = ['Resnet_bayesian']

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            #print(m.bias)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class ResBlock_Bayesian(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock_Bayesian, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = BayesianConv2d(inplanes, planes, (3,3),stride=stride,padding=1, bias=True)
        self.norm2 = norm(planes)
        self.conv2 = BayesianConv2d(planes, planes,(3,3),stride=stride,padding=1, bias=True)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


@variational_estimator
class Resnet_bayesian(nn.Module):
    def __init__(self):
        super(Resnet_bayesian, self).__init__()
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature_layers_0 = ResBlock(64, 64)
        self.feature_layers_1 = ResBlock(64, 64)
        self.feature_layers_2 = ResBlock(64, 64)
        self.feature_layers_3 = ResBlock(64, 64)
        self.feature_layers_4 = ResBlock(64, 64)
        self.feature_layers_5 = ResBlock(64, 64)
        self.feature_layers_0_bayesian = ResBlock_Bayesian(64, 64)
        self.feature_layers_1_bayesian = ResBlock_Bayesian(64, 64)
        self.feature_layers_2_bayesian = ResBlock_Bayesian(64, 64)
        self.feature_layers_3_bayesian = ResBlock_Bayesian(64, 64)
        self.feature_layers_4_bayesian = ResBlock_Bayesian(64, 64)
        self.feature_layers_5_bayesian = ResBlock_Bayesian(64, 64)
        self.fc_layers = nn.Sequential(norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10))
        # self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)
    def forward(self, x):
        out = self.downsampling_layers(x)
        x_0 = self.feature_layers_0_bayesian(out)
        out = self.feature_layers_0(out)+x_0
        x_1 = self.feature_layers_1_bayesian(x_0)
        out = self.feature_layers_1(out)+x_1
        x_2 = self.feature_layers_1_bayesian(x_1)
        out = self.feature_layers_2(out)+x_2
        x_3 = self.feature_layers_1_bayesian(x_2)
        out = self.feature_layers_3(out)+x_3
        x_4 = self.feature_layers_1_bayesian(x_3)
        out = self.feature_layers_4(out)+x_4
        x_5 = self.feature_layers_1_bayesian(x_4)
        out = self.feature_layers_5(out)+x_5
        out = self.fc_layers(out)
        return out


def test():
    model = Resnet_bayesian()
    return model  
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = test()
    print(model)
    num_params = count_parameters(model.fc_layers)
    print(num_params)
    model2 = ResBlock(64, 64)
    num_params2 = count_parameters(model2.conv1)
    print(num_params2)
    pretrained_dict = torch.load("./22/final_model")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.load_state_dict(pretrained_dict,strict=False)