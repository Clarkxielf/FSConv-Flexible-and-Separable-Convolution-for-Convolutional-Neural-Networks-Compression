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
parser.add_argument('--resume', default='./checkpointResNet20_CIFAR10_0.9_2/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./checkpointResNet20_CIFAR10_0.9_2/model_best.pth.tar)')


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


    img = Image.open(r'../将cifar10数据集转化为图片/tabby_cat_s_000319.png')

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        out_put = model(img.cuda())

    img = out_put.squeeze().detach().cpu().numpy()

    for Img in range(img.shape[0]):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        fig, ax = plt.subplots()
        ax.imshow(img[Img], aspect="equal", cmap='gray')

        plt.axis(False)
        height, width,  = img[Img].shape

        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        if len(str(Img)) == 1:
            plt.savefig(r'./' + 'FSConv' + '0' + str(Img))
        else:
            plt.savefig(r'./' + 'FSConv' + str(Img))

        plt.show()


if __name__ == '__main__':
        main()