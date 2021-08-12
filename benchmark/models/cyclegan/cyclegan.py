from typing import List

from torch import nn


class Resnet_Generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, block=6):
        assert block > 0, "block must be bigger than zero."
        super(Resnet_Generator, self).__init__()
        model = []

        # Let c7s1-k denote a 7x7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
        # dk denotes a 3x3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
        # Reflection padding was used to reduce artifacts.
        # Rk denotes a residual block that contains two 3x3 convolutional layers with the same number of filters on layer.
        # uk denotes a 3x3 factional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2.

        # c7s1-64
        channel = 64
        model += [
            nn.Conv2d(
                in_channel,
                channel,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]

        # d128, d256
        downsampling = 2
        for i in range(downsampling):
            model += [
                nn.Conv2d(channel, channel * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channel * 2),
                nn.ReLU(True),
            ]
            channel *= 2

        # R256 * block
        for i in range(block):
            model += [Resnet(channel), nn.ReLU(True)]

        # u128, u64
        upsamping = 2
        for i in range(upsamping):
            model += [
                nn.ConvTranspose2d(
                    channel,
                    int(channel / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(int(channel / 2)),
                nn.ReLU(True),
            ]
            channel = int(channel / 2)

        # c7s1-3
        model += [
            nn.Conv2d(
                channel,
                out_channel,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Resnet(nn.Module):
    def __init__(self, channel):
        super(Resnet, self).__init__()
        block = []
        block += [
            nn.Conv2d(
                channel,
                channel,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(channel),
            nn.ReLU(True),
        ]
        block += [
            nn.Conv2d(
                channel,
                channel,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(channel),
        ]
        self.Res_block = nn.Sequential(*block)

    def forward(self, input):
        output = input + self.Res_block(input)
        return output


class PatchGAN_Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(PatchGAN_Discriminator, self).__init__()
        model = []

        # Let Ck denote a 4x4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
        # After the last layer, we apply a convolution to produce a 1-dimensional output.
        # We do not use InstanceNorm for the first C64 layer.
        # We use leaky ReLUs with a slope of 0.2.

        # C64
        channel = 64
        model += [
            nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        # C128
        model += [
            nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel * 2),
            nn.LeakyReLU(0.2, True),
        ]
        channel *= 2

        # C256
        model += [
            nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel * 2),
            nn.LeakyReLU(0.2, True),
        ]
        channel *= 2

        # C512
        model += [
            nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel * 2),
            nn.LeakyReLU(0.2, True),
        ]
        channel *= 2

        # 1-dimensional Conv
        model += [nn.Conv2d(channel, out_channel, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """return 7 by 7 ouptut"""
        return self.model(input)


def initialize(model: nn.Module):
    for param in model.parameters():
        nn.init.normal_(param, 0, 0.002)


def initialize_model_list(models: List[nn.Module]):
    for model in models:
        initialize(model)
