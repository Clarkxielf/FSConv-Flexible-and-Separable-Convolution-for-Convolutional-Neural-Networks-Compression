import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import utils
import torch.nn as nn
import os

import argparse
import os
import time


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import Visualization.ResNet20_on_CIFAR10_FSConv as ResNet20_on_CIFAR10

model_names = sorted(name for name in ResNet20_on_CIFAR10.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(ResNet20_on_CIFAR10.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('--resume', default='../Visualization/tabby_cat_s_000319/checkpointResNet20_CIFAR10_0.9_2/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ../Visualization/tabby_cat_s_000319/checkpointResNet20_CIFAR10_0.9_2/model_best.pth.tar)')


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = ResNet20_on_CIFAR10.__dict__[args.arch]()

    model = torch.nn.DataParallel(model)

    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    img = Image.open(r'../Visualization/将cifar10数据集转化为图片/tabby_cat_s_000319.png')

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        out_put = model(img.cuda())

    img = out_put.squeeze().detach().cpu().numpy()

    Colorful_Matrix = []
    for Img in range(img.shape[0]):
        FSConv = img[Img]
        FSConv = np.repeat(FSConv[None, :, :], img.shape[0], axis=0)
        MSE_FSConv = np.mean((FSConv - img) ** 2, axis=(1, 2))

        if len(str(Img)) == 1:
            print("MSE_FSConv_0{}:{}".format(Img+1, MSE_FSConv))

            Colorful_Matrix.append(MSE_FSConv)
        else:
            print("MSE_FSConv_{}:{}".format(Img+1, MSE_FSConv))

            Colorful_Matrix.append(MSE_FSConv)

    Colorful_Matrix = np.array(Colorful_Matrix)
    plt.imshow(Colorful_Matrix, cmap=plt.cm.Reds)
    plt.xticks(range(Colorful_Matrix.shape[0]), range(Colorful_Matrix.shape[0]))
    plt.yticks(range(Colorful_Matrix.shape[1]), range(Colorful_Matrix.shape[1]))
    plt.colorbar()
    plt.xlabel('Index of feature maps')
    plt.ylabel('Index of feature maps')
    plt.title('Mean squared error(MSE)')
    plt.savefig(r'./tabby_cat_s_000319_FSConv.png')
    plt.show()

if __name__ == '__main__':
        main()