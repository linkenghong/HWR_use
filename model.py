import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import sys
from alfred.utils.log import logger as logging

class Inception(nn.Module):
    def __init__(self, inD, x1D, x3D_1, x3D_2, x5D_1,  x5D_2, poolD):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Conv2d(inD, x1D, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(inD, x3D_1, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(x3D_1, x3D_2, kernel_size=3, padding=1)
        self.branch5x5_1 = nn.Conv2d(inD, x5D_1, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(x5D_1, x5D_2, kernel_size=5, padding=2)
        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = nn.Conv2d(inD, poolD, kernel_size=1)


    def forward(self, x):
        branch1x1 = F.relu(self.branch1x1(x))

        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))

        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.branch5x5_2(branch5x5))

        branch_pool = self.branch_pool_1(x)
        branch_pool = F.relu(self.branch_pool_2(branch_pool))

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class HWDB_GoogLeNet(nn.Module):
    def __init__(self, num_class):
        super(HWDB_GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.reduction1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inc2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc3 = Inception(480, 160, 112, 224, 24, 64, 64)
        self.inc4 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=3, padding=1)  ####
        self.reduction2 = nn.Conv2d(832, 128, kernel_size=1)
        self.fc1 = nn.Linear(2*2*128, 1024)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, num_class)
        # self.sm = nn.Softmax(dim=1)
        self.weight_init()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.reduction1(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool3(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.pool4(x)
        x = F.relu(self.reduction2(x))
        x = x.view(-1, 2 * 2 * 128)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # x = self.sm(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class HWDB_SegNet(nn.Module):
    def __init__(self, num_class):
        super(HWDB_SegNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.reduction1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inc2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc3 = Inception(480, 160, 112, 224, 24, 64, 64)
        self.inc4 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=3, padding=1)  ####
        self.reduction2 = nn.Conv2d(832, 128, kernel_size=1)
        self.fc1 = nn.Linear(2*2*128, 256)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, num_class)
        # self.sm = nn.Softmax(dim=1)
        self.weight_init()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.reduction1(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool3(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.pool4(x)
        x = F.relu(self.reduction2(x))
        x = x.view(-1, 2 * 2 * 128)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # x = self.sm(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        logging.error('send a pattern like this: {}'.format('G'))
    else:
        p = sys.argv[1]
        logging.info('show img from: {}'.format(p))
        if p == 'G':
            model = HWDB_GoogLeNet(6765).cuda()
            summary(model, input_size=(3, 120, 120), device='cuda')
        elif p == 'S':
            model = HWDB_SegNet(6765).cuda()
            summary(model, input_size=(3, 120, 120), device='cuda')