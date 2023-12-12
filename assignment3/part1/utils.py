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
import torchvision.utils
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
    assert not (std < 0).any().item(), (
        "The reparameterization trick got a negative std as input. "
        + "Are you sure your input is std and not log_std?"
    )
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    epsilon = torch.randn(mean.shape, device=mean.device)
    z = mean + std * epsilon
    #######################
    # END OF YOUR CODE    #
    #######################
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

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # \sum_{d=1}^{D} \frac{1}{2} \left( \exp(2 \log \sigma_{nd}) + \mu_{nd}^2 - 1 - 2\log \sigma_{nd} \right)
    KLD = torch.sum(torch.exp(2 * log_std) + mean**2 - 1 - 2 * log_std, -1) / 2
    #######################
    # END OF YOUR CODE    #
    #######################
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
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    b, c, h, w = img_shape
    # bpd = nll \cdot \log_2(e) \cdot \left( \prod_{i=1}^{K} d_i \right)^{-1}
    bpd = elbo * torch.log2(torch.Tensor([torch.e]).to(elbo.device)) / (c * h * w)
    #######################
    # END OF YOUR CODE    #
    #######################
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

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # percentiles
    start = 0.5 / grid_size
    end = (grid_size - 0.5) / grid_size
    percentiles = torch.range(start, end, (end - start) / (grid_size - 1))

    # normal
    distribution = torch.distributions.Normal(torch.zeros((1, 1)), torch.eye(1))
    norm_values = distribution.icdf(percentiles)

    # combined over 2 dimensions
    z_product = torch.cartesian_prod(norm_values[0], norm_values[0])

    img_logits = decoder(z_product)
    img_probs = torch.softmax(img_logits, dim=1)
    # If probs is N-dimensional, the first N-1 dimensions are treated as a batch of relative probability vectors.
    img_probs_reordered = torch.movedim(img_probs, 1, -1)

    img_distributions = torch.distributions.Categorical(probs=img_probs_reordered)
    imgs = img_distributions.sample()
    imgs = torch.unsqueeze(imgs, 1)
    imgs = imgs.float() / 15  # expects a float between 0-1
    print(imgs.shape)
    print(imgs)
    img_grid = torchvision.utils.make_grid(imgs, nrow=grid_size)
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid
