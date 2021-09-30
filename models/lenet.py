import torch
import torch.nn as nn


__all__ = ['LeNet5', 'lenet5']


class LeNet5(nn.Module):
    # 对原始的LeNet-5进行了部分修改
    def __init__(self, num_classes: int = 1000) -> None:
        super(LeNet5, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1) # or: x = x.view(1, 120)
        x = self.classifier(x)
        return x

def lenet5(pretrained: bool = False, progress: bool = True, num_classes: int = 1000) -> LeNet5:
    model = LeNet5(num_classes)
    if pretrained:
        print("Currently does not support pre-training models, haha!")
    return model

if __name__ == "__main__":
    model = LeNet5(n_classes=10)
    x = torch.rand(size=[1,3,32,32])
    y = model(x)
    print(y, y.shape)

        



