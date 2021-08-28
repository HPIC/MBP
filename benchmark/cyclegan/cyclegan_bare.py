import argparse
import itertools
import json
import time

import torch
import torch.nn as nn
from dataloader import cyclegan_dataset
from torch.nn import init


class Resnet_Generator(nn.Module):
    def __init__(self, in_chhannel=3, out_channel=3, block=6):
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
                in_chhannel,
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


def initialize(model: torch.nn.Module):
    for para in model.parameters():
        init.normal_(para, 0, 0.002)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test our framework with CycleGAN")
    parser.add_argument("-b", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("-e", type=int, default=100)
    args = parser.parse_args()

    dataloader = cyclegan_dataset(
        path="./dataset/horse2zebra/train", image_size=args.image_size, batch_size=args.b
    )
    dev = torch.device("cuda:0")
    epochs = args.e

    # Define Models.
    block = 9 if args.image_size == 256 else 6
    size_ = 15 if args.image_size == 256 else 7
    G_A = Resnet_Generator(block=block).to(dev)
    G_B = Resnet_Generator(block=block).to(dev)
    D_A = PatchGAN_Discriminator().to(dev)
    D_B = PatchGAN_Discriminator().to(dev)

    # initialization models
    initialize(model=G_A)
    initialize(model=G_B)
    initialize(model=D_A)
    initialize(model=D_B)

    # Define loss function.
    cyc_l1loss = nn.L1Loss().to(dev)
    idt_l1loss = nn.L1Loss().to(dev)
    adv_loss = nn.MSELoss().to(dev)

    # Define optimizers
    opt_G = torch.optim.Adam(
        itertools.chain(G_A.parameters(), G_B.parameters()), lr=0.0002
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()), lr=0.0002
    )

    # Define vars
    lambda_A = 10
    lambda_B = 10
    lambda_idt = 0.5

    loss_values = {}

    # Define check performance vars
    epoch_time = 0
    epoch_iter = 0
    train_time = 0
    train_iter = 0

    with open("./loss/cyclegan_origin_loss_value.json", "w") as file:
        json_file = {}
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            loss_values[epoch] = {"g_loss": 0.0, "A_loss": 0.0, "B_loss": 0.0}
            for idx, input in enumerate(dataloader):
                train_start = time.perf_counter()

                # forward and backward
                opt_G.zero_grad()
                opt_D.zero_grad()

                real_A = input["A"].to(dev)
                real_B = input["B"].to(dev)

                fake_B = G_A(real_A)
                recover_A = G_B(fake_B)

                fake_A = G_B(real_B)
                recover_B = G_A(fake_A)

                idt_B = G_A(real_B)
                idt_A = G_B(real_A)

                """ create real and fake label for adversarial loss """
                batch_size = real_A.size(0)
                real_label = torch.ones(batch_size, 1, size_, size_).to(dev)
                fake_label = torch.zeros(batch_size, 1, size_, size_).to(dev)

                fake_output_A = D_A(fake_A)
                adv_loss_A = adv_loss(fake_output_A, real_label)
                fake_output_B = D_B(fake_B)
                adv_loss_B = adv_loss(fake_output_B, real_label)

                """ Calculate cycle loss """
                cyc_loss_A = cyc_l1loss(recover_A, real_A) * lambda_A
                cyc_loss_B = cyc_l1loss(recover_B, real_B) * lambda_B

                """ Calculate idt loss """
                idt_loss_A = idt_l1loss(idt_A, real_A) * lambda_A * lambda_idt
                idt_loss_B = idt_l1loss(idt_B, real_B) * lambda_B * lambda_idt
                g_loss = (
                    adv_loss_A
                    + adv_loss_B
                    + cyc_loss_A
                    + cyc_loss_B
                    + idt_loss_A
                    + idt_loss_B
                )

                real_output_A = D_A(real_A)

                real_A_loss = adv_loss(real_output_A, real_label)

                fake_output_A = D_A(fake_A.detach())
                fake_A_loss = adv_loss(fake_output_A, fake_label)

                real_output_B = D_B(real_B)
                real_B_loss = adv_loss(real_output_B, real_label)

                fake_output_B = D_B(fake_B.detach())
                fake_B_loss = adv_loss(fake_output_B, fake_label)

                A_loss = real_A_loss + fake_A_loss
                B_loss = real_B_loss + fake_B_loss

                g_loss.backward()
                A_loss.backward()
                B_loss.backward()

                opt_G.step()
                opt_D.step()

                loss_values[epoch]["g_loss"] += g_loss.detach().item()
                loss_values[epoch]["A_loss"] += A_loss.detach().item()
                loss_values[epoch]["B_loss"] += B_loss.detach().item()

                train_end = time.perf_counter()
                train_time += train_end - train_start
                train_iter += 1
            epoch_end = time.perf_counter()
            epoch_time += epoch_end - epoch_start
            epoch_iter += 1

            loss_values[epoch]["g_loss"] /= len(dataloader)
            loss_values[epoch]["A_loss"] /= len(dataloader)
            loss_values[epoch]["B_loss"] /= len(dataloader)

            torch.save(G_A.state_dict(), "./parameters/cyclegan_origin/G_A.pth")
            torch.save(G_B.state_dict(), "./parameters/cyclegan_origin/G_B.pth")
            torch.save(D_A.state_dict(), "./parameters/cyclegan_origin/D_A.pth")
            torch.save(D_B.state_dict(), "./parameters/cyclegan_origin/D_B.pth")

            print(
                f"[{epoch+1}/{epochs}]",
                "train time :",
                format(train_time / train_iter, ".3f") + "s",
                "epoch time :",
                format(epoch_time / epoch_iter, ".3f") + "s",
                end=" ",
            )
            json_file[epoch + 1] = {}
            for _, name in enumerate(loss_values[epoch]):
                print(
                    f"{name} :",
                    format(loss_values[epoch][name], ".2f"),
                    end=" ",
                )
                json_file[epoch + 1][name] = loss_values[epoch][name]
            print()
        json.dump(json_file, file, indent=4)
