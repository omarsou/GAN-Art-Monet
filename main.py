# Initialize the GAN and launch the training
import os
import torch
import numpy as np
from GAN.model import Generator, Discriminator, weights_init_normal
from GAN.trainer import Trainer
from GAN.data_read import get_data_loader, get_data_loader2
import itertools
import argparse


def main(args):

    # Load the data (DataLoader object)
    path_monnet = args.Monet_Path
    path_pictures = args.Pictures_Path
    save_path = args.Save_Path
    batch_size = args.batch_size
    n_epochs = args.epochs
    device = args.device
    dataset = get_data_loader(path_monnet, path_pictures, batch_size)

    # Create Generators and Discriminators and put them on GPU/TPU
    generator_AB = Generator().to(device)
    generator_BA = Generator().to(device)

    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    generator_AB.apply(weights_init_normal)
    generator_BA.apply(weights_init_normal)
    discriminator_A.apply(weights_init_normal)
    discriminator_B.apply(weights_init_normal)

    # Set optimizers

    G_optimizer = torch.optim.Adam(itertools.chain(
        generator_AB.parameters(), generator_BA.parameters()), lr=2e-4)

    D_optimizer = torch.optim.Adam(itertools.chain(
        discriminator_A.parameters(), discriminator_B.parameters()), lr=2e-4)

    # Set trainer
    trainer = Trainer(
        generator_ab=generator_AB,
        generator_ba=generator_BA,
        discriminator_a=discriminator_A,
        discriminator_b=discriminator_B,
        generator_optimizer=G_optimizer,
        discriminator_optimizer=D_optimizer,
        n_epochs=n_epochs,
        dataloader=dataset,
        device=device,
    )

    # Launch Training
    trainer.train()

    # Save the model and the loss during training
    # Save logs
    trainer.log.save(os.path.join(save_path, 'save_loss.txt'))
    # Save the model
    torch.save(generator_AB.state_dict(),
               os.path.join(save_path, 'generator_AB.pt'))
    torch.save(generator_BA.state_dict(),
               os.path.join(save_path, 'generator_BA.pt'))
    torch.save(discriminator_A.state_dict(),
               os.path.join(save_path, 'discriminator_A.pt'))
    torch.save(discriminator_B.state_dict(),
               os.path.join(save_path, 'discriminator_B.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Monet_Path")
    parser.add_argument("--Pictures_Path")
    parser.add_argument("--Save_Path", default='')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()
    main(args)
