import cv2
import numpy as np
import os

file = os.getcwd().split('\\Visualization')[0] + '\data\CIFAR10\cifar-10-batches-py\data_batch_1'



def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='iso-8859-1')
    return dict


dict1 = unpickle(file)

for i in range(1000):  # 保存1000张
    img = dict1["data"][i]
    img = np.reshape(img, (3, 32, 32))
    img = img.transpose((1, 2, 0))
    img_name = dict1["filenames"][i]
    img_label = str(dict1["labels"][i])
    cv2.imwrite(os.path.join(img_name), img)