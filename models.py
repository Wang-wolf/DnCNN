import math
import torch.nn as nn


def weights_init_kaiming(m):
    # 根據不同的層進行權重初始化
    # 名稱與classname不符時會等於-1，反之會等於0
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


class DnCNN(nn.Module):
    # Denoisor
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = (3, 3)
        padding = 1
        stride = (1, 1)
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # 依卷積層->批次正規化->激活函數(relu)的順序
        # 預設為重複15次
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # 前向傳遞
        output = self.dncnn(x)
        return output
