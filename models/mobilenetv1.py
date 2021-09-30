import torch
import torch.nn as nn

__all__ = ['MobileNetV1', 'mobilenetv1']

def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,  bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )


def conv_dw(in_channels, out_channels, kernel_size=3, stride=3, padding=1, bias=False):
    return nn.Sequential(
        # dw
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )

def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
    return nn.Sequential(
        # pw
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def dw_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        conv_dw(in_channels=in_channels, out_channels=in_channels, stride=stride),
        conv_pw(in_channels=in_channels, out_channels=out_channels, stride=stride),
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(MobileNetV1, self).__init__()

        self.features = nn.Sequential(
            conv_bn(3, 32, stride=2),
            dw_block(32, 64),
            dw_block(64, 128, 2),
            dw_block(128, 128),
            dw_block(128, 256, 2),
            dw_block(256, 256),
            dw_block(256, 512, 2),
            dw_block(512, 512),
            dw_block(512, 512),
            dw_block(512, 512),
            dw_block(512, 512),
            dw_block(512, 512),
            dw_block(512, 1024, 2),
            dw_block(1024, 1024),
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(-1, 1024)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenetv1(pretrained: bool = False, progress: bool = True, num_classes: int = 1000) -> MobileNetV1:
    if pretrained:
        model = MobileNetV1()
        print("Currently does not support pre-training models, haha!")
        fc_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features=fc_features, out_features=num_classes)
    else:
        model = MobileNetV1(num_classes)
    return model
