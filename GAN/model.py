# File where we build the cycle GAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


class Generator(nn.Module):

    # Initialize the Generator
    def __init__(self):
        super(Generator, self).__init__()

        ## Architecture of the Generator ##

        self.rp1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=1, bias=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2_in = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3_in = nn.InstanceNorm2d(256, affine=True)

        self.residual_block_1 = ResidualBlock()
        self.residual_block_2 = ResidualBlock()
        self.residual_block_3 = ResidualBlock()
        self.residual_block_4 = ResidualBlock()
        self.residual_block_5 = ResidualBlock()
        self.residual_block_6 = ResidualBlock()

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=256*2, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.tconv1_in = nn.InstanceNorm2d(128, affine=True)

        self.tconv2 = nn.ConvTranspose2d(
            in_channels=128*2, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.tconv2_in = nn.InstanceNorm2d(64, affine=True)

        self.conv_final = nn.Conv2d(
            in_channels=64*2, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True)

        self.activation = nn.Tanh()

    # Forward Pass

    def forward(self, input_sample):

        x1 = F.relu(self.conv1(self.rp1(input_sample)))
        x2 = F.relu(self.conv2_in(self.conv2(x1)))
        x3 = F.relu(self.conv3_in(self.conv3(x2)))

        x = self.residual_block_1(x3)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)
        x = self.residual_block_5(x)
        x = self.residual_block_6(x)

        x_cat1 = torch.cat((x, x3), 1)
        x = F.relu(self.tconv1_in(self.tconv1(x_cat1)))
        x_cat2 = torch.cat((x, x2), 1)
        x = F.relu(self.tconv2_in(self.tconv2(x_cat2)))
        x_cat3 = torch.cat((x, x1), 1)
        x = self.activation(self.conv_final(x_cat3))
        return x


class Discriminator(nn.Module):

    # Initialize the Discriminator
    def __init__(self):
        super(Discriminator, self).__init__()

        ##Discriminator Architecture##
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2_in = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3_in = nn.InstanceNorm2d(256, affine=True)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True)
        self.conv4_in = nn.InstanceNorm2d(512, affine=True)

        self.final_conv = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True)

    # Forward pass
    def forward(self, input_sample):

        x = F.leaky_relu_(self.conv1(input_sample), negative_slope=0.2)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), negative_slope=0.2)
        x = self.final_conv(x)
        # Average pooling
        return torch.reshape(F.avg_pool2d(x, kernel_size=30), (1,))


class ResidualBlock(nn.Module):

    # initialize the ResidualBlock
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1_bn = nn.InstanceNorm2d(256, affine=True)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_bn = nn.InstanceNorm2d(256, affine=True)

        self.conv_block = nn.Sequential(
            *[self.conv1, nn.ReLU(inplace=True), self.conv2])

    # Forward pass

    def forward(self, input_sample):
        return self.conv_block(input_sample) + input_sample


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
