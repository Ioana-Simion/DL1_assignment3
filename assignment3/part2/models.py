################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-20
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self,  z_dim: int = 20, num_input_channels: int = 1, num_filters: int = 32,):
        """
        Convolutional Encoder network with Convolution and Linear layers, ReLU activations. The output layer
        uses a Fully connected layer to embed the representation to a latent code with z_dim dimension.
        Inputs:
            z_dim - Dimensionality of the latent code space.
        """
        super(ConvEncoder, self).__init__()
       
        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        # Tutorial 9 architecture > num_input_channels
        #print(num_filters, num_input_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2), # 28x28 => 15x15?? or 14x14
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 15x15 => 8x8
            nn.GELU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*num_filters, z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch of Images. Shape: [B,C,H,W]
        Outputs:
            z - Output of latent codes [B, z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #print('shape of x >>>>>>>>>>>', x.shape)
        z = self.encoder(x)
        #print('shape of z ===========ENCODERR', z.shape)
        #######################
        # END OF YOUR CODE    #
        #######################
        return z

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class ConvDecoder(nn.Module):
    def __init__(self, z_dim: int = 20, num_input_channels: int = 1, num_filters: int = 32):
        """
        Convolutional Decoder network with linear and deconvolution layers and ReLU activations. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(ConvDecoder, self).__init__()
        # For an intial architecture, you can use the decoder of Tutorial 9. You can set the
        # output padding in the first transposed convolution to 0 to get 28x28 outputs.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*num_filters),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * num_filters, 2 * num_filters, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 => 15x15
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 15x15 => 28x28
            nn.Tanh()  # Scales output to [-1, 1]
        )
    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        recon_x = self.linear(z)
        
        #print('shape of z >>>>>>>>>>>', z.shape)

        #print('shape of recon_x >>>>>>>>>>>', recon_x.shape)
        recon_x = recon_x.reshape(recon_x.shape[0], -1, 4, 4)

        #print('shape of recon_x after >>>>>>>>>>>', recon_x.shape)
        recon_x = self.decoder(recon_x)

        #print('shape of recon_x decoded >>>>>>>>>>>', recon_x.shape)
        return recon_x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        """
        Discriminator network with linear layers and LeakyReLU activations.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(Discriminator, self).__init__()
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # As a default setup, we recommend 3 linear layers (512 for hidden units) with LeakyReLU activation functions (negative slope 0.2).
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        # for m in self.layers:
        #     if isinstance(m, nn.Linear):
        #         #nn.init.constant_(m.weight, 0.0)
        #         nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            preds - Predictions whether a specific latent code is fake (<0) or real (>0). 
                    No sigmoid should be applied on the output. Shape: [B,1]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #print('DISCRIMINTATOR --------- z', z.shape)
        preds = self.layers(z)

        #print('DISCRIMINTATOR --------- preds', preds.shape)
        return preds

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class AdversarialAE(nn.Module):
    def __init__(self, z_dim=8):
        """
        Adversarial Autoencoder network with a Encoder, Decoder and Discriminator.
        Inputs:
              z_dim - Dimensionality of the latent code space. This is the number of neurons of the code layer
        """
        super(AdversarialAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #print('shape of x ===========', x.shape)
        z = self.encoder(x)
        #print('shape of z ===========', z.shape)
        recon_x = self.decoder(z)
        #print('shape of reon_x ===========', recon_x.shape)
        return recon_x, z

    def get_loss_autoencoder(self, x, recon_x, z_fake, lambda_=1):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
            recon_x - Reconstructed image of shape [B,C,H,W]
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
            lambda_ - The reconstruction coefficient (between 0 and 1).

        Outputs:
            recon_loss - The MSE reconstruction loss between actual input and its reconstructed version.
            gen_loss - The Generator loss for fake latent codes extracted from input.
            ae_loss - The combined adversarial and reconstruction loss for AAE
            lambda_ * reconstruction loss + (1 - lambda_) * adversarial loss
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        recon_loss = F.mse_loss(recon_x, x)
        logits = self.discriminator(z_fake)
        targets = torch.ones_like(logits)
        gen_loss = F.binary_cross_entropy_with_logits(logits, targets)
        ae_loss = lambda_ * recon_loss + (1 - lambda_) * gen_loss
        logging_dict = {"gen_loss": gen_loss,
                        "recon_loss": recon_loss,
                        "ae_loss": ae_loss}
        return ae_loss, logging_dict


    def get_loss_discriminator(self,  z_fake):
        """
        Inputs:
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]

        Outputs:
            disc_loss - The discriminator loss for real and fake latent codes.
            logging_dict - A dictionary for logging the model performance by following keys:
                disc_loss - The discriminator loss for real and fake latent codes.
                loss_real - The discriminator loss for latent codes sampled from the standard Gaussian prior.
                loss_fake - The discriminator loss for latent codes extracted by encoder from input
                accuracy - The accuracy of the discriminator for both real and fake samples.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # z_real = torch.randn_like(z_fake, device=self.device)
        # real_preds = self.discriminator(z_real)
        # fake_preds = self.discriminator(z_fake)

        # loss_real = F.relu(1.0 - real_preds).mean()
        # loss_fake = F.relu(1.0 + fake_preds).mean()
        # disc_loss = loss_real + loss_fake

        # accuracy = ((real_preds > 0).float().mean() + (fake_preds < 0).float().mean()) / 2

        # logging_dict = {"disc_loss": disc_loss.item(),
        #                 "loss_real": loss_real.item(),
        #                 "loss_fake": loss_fake.item(),
        #                 "accuracy": accuracy.item()}
        # return disc_loss, logging_dict

        z_real = torch.randn_like(z_fake, device=self.device)
        real_preds = self.discriminator(z_real)
        fake_preds = self.discriminator(z_fake)

        real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
        disc_loss = (real_loss + fake_loss) / 2

        accuracy = ((real_preds > 0).float().mean() + (fake_preds < 0).float().mean()) / 2

        logging_dict = {"disc_loss": disc_loss,
                        "loss_real": real_loss,
                        "loss_fake": fake_loss,
                        "accuracy": accuracy}
        return disc_loss, logging_dict

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random or conditioned images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        x = self.decoder(z)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return self.encoder.device


