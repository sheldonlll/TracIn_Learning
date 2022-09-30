import torch
from torch import nn


class LeNet5(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        #x: [b, 3, 32, 32]
        self.convUnit = nn.Sequential(  nn.Conv2d(3, 6, kernel_size = 5, stride = 1, padding = 0),
                                        nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
                                        nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0),
                                        nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        )

        self.fcUnit = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                     nn.ReLU(),
                                     nn.Linear(120, 84),
                                     nn.ReLU(),
                                     nn.Linear(84, 10)
        )


        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.convUnit(tmp)
        # print("conv out:", out.shape) # [2, 16, 5, 5]


    def forward(self, x):
        batch_size = x.shape[0]
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.convUnit(x)
        # [b, 16, 5, 5] => [b, 16 * 5 * 5]
        x = x.view(batch_size, 16 * 5 * 5)
        logits = self.fcUnit(x)
        return logits


def main():
    LeNet5 = LeNet5()


if __name__ == "__main__":
    main()