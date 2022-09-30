from audioop import bias
from math import fabs
from operator import mod
from turtle import forward
import torch
from torch import nn

class ResNetBlock(nn.Module):
    '''
    a resnet block
    ''' 
    def __init__(self, channel_in, channel_out, channel_final_out, stride = 1) -> None:
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size = 3, stride = stride, padding = 0)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_final_out, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channel_final_out)

        self.extra = nn.Sequential()
        if channel_in != channel_out:
            #[b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(channel_in, channel_final_out, kernel_size = 1, stride = stride, padding = 1, bias = False),
                nn.BatchNorm2d(channel_final_out)
            )

    def forward(self, x):
        """
        :param x: [batch size, channel, height, weight]
        :return 
        """
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # shortcut connections
        # [b, ch_in, h, w]  [b, ch_out, h, w] 
        out += self.extra(x) # element-wise add
        return out



class ResNet34(torch.nn.Module):
    def __init__(self) -> None:
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1, bias = False)
        self.conv2_1 = ResNetBlock(64, 64, 64)
        self.conv2_2 = ResNetBlock(64, 64, 128)
        self.conv3_1 = ResNetBlock(128, 128, 128)
        self.conv3_2 = ResNetBlock(128, 128, 256)
        self.conv4_1 = ResNetBlock(256, 256, 256)
        self.conv4_2 = ResNetBlock(256, 256, 512)
        self.conv5_1 = ResNetBlock(512, 512, 512)
        self.conv5_2 = ResNetBlock(512, 512, 1000)
        self.outlayer = nn.Linear(1000, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_1(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_1(x)
        x = self.conv4_1(x)
        x = self.conv4_1(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.outlayer(x)
        
        return x
    

def main():
    blk = ResNetBlock(64, 128, 128, stride=1)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print(out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet34()
    out = model(x)
    print(f"resnet: {out.shape}")

if __name__ == "__main__":
    main()