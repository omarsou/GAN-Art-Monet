# Initialize the GAN and launch the training

import torch
import numpy as np
from GAN.model import Generator, Discriminator
from GAN.trainer import Trainer
from GAN.data_read import get_data_loader


def main():

    # Load the data (DataLoader object)
    path = 'C:/Users/remys/GAN-Art-Monet/img/dfdfd'
    batch_size = 8
    n_epochs = 100
    device = 'cpu'
    dataset = get_data_loader(path, batch_size)

    # Create Generators and Discriminators and put them on GPU/TPU
    generator_AB = Generator().to(device)
    generator_BA = Generator().to(device)

    discriminator_AB = Discriminator().to(device)
    discriminator_BA = Discriminator().to(device)

    # Set optimizers
    G_AB_optimizer = torch.optim.Adam(generator_AB.parameters(), lr=1e-4)
    G_BA_optimizer = torch.optim.Adam(generator_BA.parameters(), lr=1e-4)

    D_AB_optimizer = torch.optim.Adam(discriminator_AB.parameters(), lr=1e-4)
    D_BA_optimizer = torch.optim.Adam(discriminator_BA.parameters(), lr=1e-4)

    # Set trainer
    trainer = Trainer(
        generator_ab=generator_AB,
        generator_ba=generator_BA,
        discriminator_ab=discriminator_AB,
        discriminator_ba=discriminator_BA,
        generator_ab_optimizer=G_AB_optimizer,
        generator_ba_optimizer=G_BA_optimizer,
        discriminator_ab_optimizer=D_AB_optimizer,
        discriminator_ba_optimizer=D_BA_optimizer,
        n_epochs=n_epochs,
        dataloader=dataset,
        device=device,
    )

    # Launch Training
    trainer.train()


if __name__ == '__main__':
    main()
