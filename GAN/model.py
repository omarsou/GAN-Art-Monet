# File where we build the cycle GAN
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    # Initialize the Generator
    def __init__(self):
        super(Generator, self).__init__()

        ## Architecture of the Generator ##
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=1, padding=3)
        self.conv1_in = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1)
        self.conv2_in = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=2, padding=1)
        self.conv3_in = nn.InstanceNorm2d(256)

        self.residual_block_1 = ResidualBlock()
        self.residual_block_2 = ResidualBlock()
        self.residual_block_3 = ResidualBlock()
        self.residual_block_4 = ResidualBlock()
        self.residual_block_5 = ResidualBlock()
        self.residual_block_6 = ResidualBlock()
        self.residual_block_7 = ResidualBlock()
        self.residual_block_8 = ResidualBlock()
        self.residual_block_9 = ResidualBlock()

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv1_in = nn.InstanceNorm2d(128)

        self.tconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv2_in = nn.InstanceNorm2d(64)

        self.conv_final = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    # Forward Pass

    def forward(self, input_sample):

        x = F.relu(self.conv1_in(self.conv1(input_sample)))
        x = F.relu(self.conv2_in(self.conv2(x)))
        x = F.relu(self.conv3_in(self.conv3(x)))

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)
        x = self.residual_block_5(x)
        x = self.residual_block_6(x)
        x = self.residual_block_7(x)
        x = self.residual_block_8(x)
        x = self.residual_block_9(x)

        x = F.relu(self.tconv1_in(self.tconv1(x)))
        x = F.relu(self.tconv2_in(self.tconv2(x)))

        x = F.tanh(self.conv_final(x))

        return x


class Discriminator(nn.Module):

    # Initialize the Discriminator
    def __init__(self):
        super(Discriminator, self).__init__()

        ##Discriminator Architecture##
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv2_in = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv3_in = nn.InstanceNorm2d(256)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv4_in = nn.InstanceNorm2d(512)

        self.final_conv = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

    # Forward pass
    def forward(self, input_sample):

        x = F.leaky_relu_(self.conv1(input_sample), negative_slope=0.2)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), negative_slope=0.2)
        x = self.final_conv(x)
        # Average pooling
        return F.avg_pool2d(x, kernel_size=30)


class ResidualBlock(nn.Module):

    # initialize the ResidualBlock
    def __init__(self):
        super(ResidualBlock, self).__init__()

    conv1 = nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1)

    conv1_bn = nn.BatchNorm2d(256)

    conv2 = nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1)
    conv2_bn = nn.BatchNorm2d(256)

    # Forward pass
    def forward(self, input_sample):
        x = F.relu(self.conv1_bn(self.conv1(input_sample)))
        x = self.conv2_bn(self.conv2(x))
        return x+input_sample
