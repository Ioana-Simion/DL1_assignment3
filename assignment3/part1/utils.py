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
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"

    eps = torch.randn_like(std)
    z = mean + std * eps
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    KLD = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - (2 * log_std).exp(), dim=-1) #TODO : recheck formulaa
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    bpd = elbo * (np.log2(np.e)/ np.prod(img_shape[1:])) 
    # section 1.6 formula double check
    # Divide by log(2) to convert nats to bits and normalize by the number of pixels
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    percentiles = 1 / grid_size * torch.arange(0.5, grid_size + 0.5)
    z_values = torch.distributions.normal.Normal(0, 1).icdf(percentiles)

    grid_x, grid_y = torch.meshgrid(z_values, z_values)
    grid_x = grid_x.reshape(-1,1)
    grid_y = grid_y.reshape(-1,1)
    grid = torch.column_stack([grid_x, grid_y])

    decoded_logits = decoder(grid)  
    #decoded_logits = torch.argmax(decoded_logits, dim=1, keepdim=True)/15
    #img_grid = make_grid(decoded_logits, nrow=grid_size)
    #return img_grid

    decoded_images = torch.softmax(decoded_logits, dim=1)  # Convert logits to probabilities or pixel values

    B, C, H, W = decoded_images.shape
    flat_probabilities = decoded_images.permute(0, 2, 3, 1).reshape(-1, C)# C, H, W dim

    
    sampled_flat = torch.multinomial(flat_probabilities, 1).squeeze(-1) # [B. H*W]
    x_samples = sampled_flat.view(B, H, W, 1).permute(0, 3, 1, 2)
    img_grid = make_grid(x_samples.float(), nrow=grid_size)

    return img_grid

