from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


dataset_dir = './data/'                 # 数据集路径
model_file = './model/model.pth'        # 模型保存路径

def test():

    model = Net()                                       # 实例化一个网络
    model.cuda()                                        # 送入GPU，利用GPU计算
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    datafile = DVCD('test', dataset_dir)                # 实例化一个数据集
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    index = np.random.randint(0, datafile.data_size, 1)[0]      # 获取一个随机数，即随机从数据集中获取一个测试图片
    img = datafile.__getitem__(index)                           # 获取一个图像
    img = img.unsqueeze(0)                                      # 因为网络的输入是一个4维Tensor，3维数据，1维样本大小，所以直接获取的图像数据需要增加1个维度
    img = Variable(img).cuda()                                  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
    out = model(img)                                            # 网路前向计算，输出图片属于猫或狗的概率，第一列维猫的概率，第二列为狗的概率
    print(out)                      # 输出该图像属于猫或狗的概率
    if out[0, 0] > out[0, 1]:                   # 猫的概率大于狗
        print('the image is a cat')
    else:                                       # 猫的概率小于狗
        print('the image is a dog')

    img = Image.open(datafile.list_img[index])      # 打开测试的图片
    plt.figure('image')                             # 利用matplotlib库显示图片
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test()


