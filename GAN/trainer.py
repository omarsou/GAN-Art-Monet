# File where we build the trainer
import torch


# Class that handle the training of the cycle GAN

class Trainer():

    def __init__(
        self,
        generator_ab,
        generator_ba,
        discriminator_ab,
        discriminator_ba,
        generator_ab_optimizer,
        generator_ba_optimizer,
        discriminator_ab_optimizer,
        discriminator_ba_optimizer,
        n_epochs,
        dataloader,
        device,
    ):

        self.gen_ab = generator_ab
        self.gen_ba = generator_ba
        self.dis_ab = discriminator_ab
        self.dis_ba = discriminator_ba
        self.gen_ab_optim = generator_ab_optimizer,
        self.gen_ab_optim = generator_ba_optimizer,
        self.dis_ab_optim = discriminator_ab_optimizer,
        self.dis_ab_optim = discriminator_ba_optimizer,
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.device = device

        pass

    def train(self):

        for _ in range(self.n_epochs):
            for i, data in enumerate(self.dataloader):
                pass
