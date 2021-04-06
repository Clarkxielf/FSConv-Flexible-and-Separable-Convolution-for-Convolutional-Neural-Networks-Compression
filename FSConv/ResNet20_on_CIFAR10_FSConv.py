'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

import math
import random
import numpy as np

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, gate_fn=hard_sigmoid, **_):
        super(SqueezeExcite, self).__init__()

        self.gate_fn = gate_fn
        reduced_chs = int(in_chs * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class BasicModule(nn.Module):
    def __init__(self, input, output, DW_size=3, stride=1, Alpha=0.7):
        super(BasicModule, self).__init__()

        self.input = input
        self.output = output
        self.stride = stride

        if self.stride!=1 or self.input!=self.output:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=self.input, out_channels=self.output,
                                                   kernel_size=1, stride=self.stride, padding=0, bias=False),
                                         nn.BatchNorm2d(self.output))
        else:
            self.shortcut = nn.Sequential()

        if self.output>=self.input:
            self.DW_input = int(input * Alpha)
            self.Conv_input = input - self.DW_input
            times = math.floor(self.output/self.input)
            self.DW_output = self.DW_input * times
            self.Conv_output = self.output-self.DW_output
            self.group = self.DW_input
        else:
            self.DW_output = int(output * Alpha)
            self.Conv_output = output-self.DW_output
            times = math.floor(self.input/self.output)
            self.DW_input = self.DW_output * times
            self.Conv_input = self.input - self.DW_input
            self.group = self.DW_output

        self.DW = nn.Sequential(
            nn.Conv2d(self.DW_input, self.DW_output, kernel_size=DW_size, stride=self.stride, padding=1, groups=self.group, bias=False),
            nn.BatchNorm2d(self.DW_output))

        self.Conv = nn.Sequential(nn.Conv2d(in_channels=self.Conv_input, out_channels=self.Conv_output,
                                            kernel_size=3, stride=self.stride, padding=1, bias=False),
                                  nn.BatchNorm2d(self.Conv_output))

        self.se = SqueezeExcite(self.output)

    def forward(self, x):

        for factor in range(2, x.shape[1]+1):
            if x.shape[1]%factor == 0:
                x = x.reshape(-1, x.shape[1]//factor, factor, x.shape[2], x.shape[3])
                x = x.transpose(2, 1)
                x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

                break

        out1 = self.shortcut(x)

        x1 = self.DW(x[:, self.Conv_input:, :, :])
        x2 = self.Conv(x[:, :self.Conv_input, :, :])

        out2 = torch.cat([x1, x2], dim=1)

        out = self.se(F.relu(out1+out2, inplace=True))

        return out



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', Beta=2):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = BasicModule(in_planes, int(Beta * planes), stride=stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = BasicModule(int(Beta * planes), planes)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])



# from torchstat import stat
#
# model = resnet20()
#
#
# stat(model, (3, 32, 32))


