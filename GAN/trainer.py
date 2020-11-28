# File where we build the trainer
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import random
from .model import ReplayBuffer
from .logs import Log

# Class that handle the training of the cycle GAN


class Trainer():

    def __init__(
        self,
        generator_ab,
        generator_ba,
        discriminator_a,
        discriminator_b,
        generator_optimizer,
        discriminator_optimizer,
        n_epochs,
        dataloader,
        device,
    ):

        self.gen_ab = generator_ab
        self.gen_ba = generator_ba
        self.dis_a = discriminator_a
        self.dis_b = discriminator_b
        self.gen_optim = generator_optimizer
        self.dis_optim = discriminator_optimizer
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.device = device

        self.identity_loss = torch.nn.L1Loss()
        self.discriminator_loss = torch.nn.MSELoss()
        self.double_pass_loss = torch.nn.L1Loss()

        # Log class to keep track of losses
        self.log = Log()

    def train(self):

        # Avoid cyclic optimization
        fake_BA_buffer = ReplayBuffer()
        fake_AB_buffer = ReplayBuffer()

        for c_epoch in range(self.n_epochs):

            one = Variable(torch.ones(1), requires_grad=False).to(self.device)
            zero = Variable(torch.zeros(
                1), requires_grad=False).to(self.device)
            print('Start epoch {}'.format(c_epoch))

            for i, data in enumerate(self.dataloader):

                print(i)

                # Sample from the image folder
                data_A = Variable(data[0]).to(self.device)  # Painting
                data_B = Variable(data[1]).to(self.device)  # Image

                # -------------------------------------Train Generator

                self.gen_optim.zero_grad()

                # 1st loss: identity
                fake_AA = self.gen_ba(data_A)
                fake_BB = self.gen_ab(data_B)
                id_loss_A = self.identity_loss(data_A, fake_AA)
                id_loss_B = self.identity_loss(data_B, fake_BB)

                # 2nd loss: Discriminator loss
                fake_AB = self.gen_ab(data_A)
                fake_BA = self.gen_ba(data_B)
                fake_prediction_AB = self.dis_b(fake_AB)
                fake_prediction_BA = self.dis_a(fake_BA)

                disc_loss_AB = self.discriminator_loss(fake_prediction_AB, one)
                disc_loss_BA = self.discriminator_loss(fake_prediction_BA, one)

                # 3rd loss : double pass
                fake_ABA = self.gen_ba(fake_AB)
                fake_BAB = self.gen_ab(fake_BA)

                dp_loss_ABA = self.double_pass_loss(fake_ABA, data_A)
                dp_loss_BAB = self.double_pass_loss(fake_BAB, data_B)

                # Final loss
                L_g = 5*id_loss_A + 5*id_loss_B + disc_loss_AB + \
                    disc_loss_BA + 10*dp_loss_ABA + 10*dp_loss_BAB
                L_g.backward()

                self.gen_optim.step()

                # --------------------------------Print option
                if i % 100 == 0:

                    self.log.plot_img(
                        c_epoch,
                        data_A,
                        data_B,
                        fake_AB,
                        fake_AA,
                        fake_BA,
                        fake_BB
                    )

                # ------------------------------------ Train Discriminator

                self.dis_optim.zero_grad()

                # Sample prediction

                fake_AB = fake_AB_buffer.push_and_pop(fake_AB)
                fake_BA = fake_BA_buffer.push_and_pop(fake_BA)

                real_prediction_A = self.dis_a(data_A)
                real_loss_A = self.discriminator_loss(real_prediction_A, one)

                real_prediction_B = self.dis_b(data_B)
                real_loss_B = self.discriminator_loss(real_prediction_B, one)

                # Fake prediction

                fake_prediction_A = self.dis_a(fake_BA.detach())
                fake_prediction_B = self.dis_b(fake_AB.detach())

                fake_loss_A = self.discriminator_loss(fake_prediction_A, zero)
                fake_loss_B = self.discriminator_loss(fake_prediction_B, zero)

                L_D = (real_loss_A + real_loss_B +
                       fake_loss_A + fake_loss_B)*0.5
                L_D.backward()

                self.dis_optim.step()

                # Logs : add the loss output
                self.log.update([
                    id_loss_A,
                    disc_loss_AB,
                    dp_loss_ABA,
                    id_loss_B,
                    disc_loss_BA,
                    dp_loss_BAB,
                    real_loss_A,
                    real_loss_B,
                    fake_loss_A,
                    fake_loss_B
                ])
