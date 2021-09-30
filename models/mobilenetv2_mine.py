import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenetv2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def dwise_conv(in_channels, stride):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(in_channels, out_channels):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(in_channels, out_channels, stride):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    )

def inverted_block(in_channels, mid_channels, out_channels, stride, first_layer=False):
    if first_layer:
        return nn.Sequential(
            dwise_conv(in_channels, stride),
            conv1x1(mid_channels,out_channels),
        )
    return nn.Sequential(
        conv1x1(in_channels,mid_channels),
        dwise_conv(mid_channels, stride),
        conv1x1(mid_channels,out_channels), 
    )


# def inverted_block0():
#     return nn.Sequential(
#         dwise_conv(32, 1),
#         conv1x1(32,16),
#     )

# def inverted_block1():
#     return nn.Sequential(
#         conv1x1(16,96),
#         dwise_conv(96, 2),
#         conv1x1(96,24), 
#     )

# def inverted_block2():
#     return nn.Sequential(
#         conv1x1(24,144),
#         dwise_conv(144, 1),
#         conv1x1(144,24), 
#     )

# def inverted_block3():
#     return nn.Sequential(
#         conv1x1(24,144),
#         dwise_conv(144, 2),
#         conv1x1(144,32), 
#     )

# def inverted_block4():
#     return nn.Sequential(
#         conv1x1(32,192),
#         dwise_conv(192, 1),
#         conv1x1(192,32), 
#     )

# def inverted_block5():
#     return inverted_block4()

# def inverted_block6():
#     return nn.Sequential(
#         conv1x1(32,192),
#         dwise_conv(192, 2),
#         conv1x1(192,64), 
#     )

# def inverted_block7():
#     return nn.Sequential(
#         conv1x1(64,384),
#         dwise_conv(384, 1),
#         conv1x1(384,64), 
#     )

# def inverted_block8():
#     return inverted_block7()

# def inverted_block9():
#     return inverted_block7()

# def inverted_block10():
#     return nn.Sequential(
#         conv1x1(64,384),
#         dwise_conv(384, 1),
#         conv1x1(384,96), 
#     )

# def inverted_block11():
#     return nn.Sequential(
#         conv1x1(96,576),
#         dwise_conv(576, 1),
#         conv1x1(576,96), 
#     )

# def inverted_block12():
#     return inverted_block11()

# def inverted_block13():
#     return nn.Sequential(
#         conv1x1(96,576),
#         dwise_conv(576, 2),
#         conv1x1(576,160), 
#     )

# def inverted_block14():
#     return nn.Sequential(
#         conv1x1(160,960),
#         dwise_conv(960, 1),
#         conv1x1(960,160), 
#     )

# def inverted_block15():
#     return inverted_block14()


# def inverted_block16():
#     return nn.Sequential(
#         conv1x1(160,960),
#         dwise_conv(960, 1),
#         conv1x1(960,320), 
#     )




class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(MobileNetV2, self).__init__()
        
        self.features = nn.Sequential(
            conv3x3(3,32,2),
            # inverted_block0(),
            # inverted_block1(),
            # inverted_block2(),
            # inverted_block3(),
            # inverted_block4(),
            # inverted_block5(),
            # inverted_block6(),
            # inverted_block7(),
            # inverted_block8(),
            # inverted_block9(),
            # inverted_block10(),
            # inverted_block11(),
            # inverted_block12(),
            # inverted_block13(),
            # inverted_block14(),
            # inverted_block15(),
            # inverted_block16(),
            # in_channels, mid_channels, out_channels, stride(dwise_conv)
            inverted_block( 32,  32,  16,  1, first_layer=True),
            inverted_block( 16,  96,  24,  2),
            inverted_block( 24, 144,  24,  1),
            inverted_block( 24, 144,  32,  2),
            inverted_block( 32, 192,  32,  1),
            inverted_block( 32, 192,  32,  1),
            inverted_block( 32, 192,  64,  2),
            inverted_block( 64, 384,  64,  1),
            inverted_block( 64, 384,  64,  1),
            inverted_block( 64, 384,  64,  1),
            inverted_block( 64, 384,  96,  1),
            inverted_block( 96, 576,  96,  1),
            inverted_block( 96, 576,  96,  1),
            inverted_block( 96, 576, 160,  2),
            inverted_block(160, 960, 160,  1),
            inverted_block(160, 960, 160,  1),
            inverted_block(160, 960, 320,  1),
            conv1x1(320, 1280),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


def mobilenetv2(pretrained: bool = False, progress: bool = True, num_classes: int = 1000) -> MobileNetV2:
    model = MobileNetV2(num_classes)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict) # /home/cxinglong/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth
    # fc_features = model.classifier.in_features
    # model.classifier = nn.Linear(in_features=fc_features, out_features=num_classes)
    return model


