################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


def one_hot(y, num_classes=None):
    assert np.issubdtype(y.dtype, np.integer)

    if not num_classes:
        num_classes = np.max(y) + 1
    Y = np.zeros((y.shape[0], num_classes))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False, name=""):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {"weight": None, "bias": None}  # Model parameters
        self.grads = {"weight": None, "bias": None}  # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.name = name
        self.in_features = in_features
        self.out_features = out_features

        # FIXME this is actually the initialization rule derived for ReLU activations,
        # though it works fine in practice for ELU as well
        if input_layer:
            # Different initialization for the first layer, as there is a no ReLU/ELU activation before it
            # that changes its values
            self.params["weight"] = np.random.normal(
                0, 1 / (in_features * out_features), (out_features, in_features)
            )
        else:
            self.params["weight"] = np.random.normal(
                0, 2 / (in_features * out_features), (out_features, in_features)
            )
        self.params["bias"] = np.zeros(out_features)
        self.grads["weight"] = np.zeros((out_features, in_features))
        self.grads["bias"] = np.zeros(out_features)

        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        assert x.shape[1] == self.in_features
        self.x = x
        out = x @ self.params["weight"].T + self.params["bias"]
        assert out.shape[1] == self.out_features
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        bias_grad = dout.sum(axis=0)
        assert self.grads["bias"].shape == bias_grad.shape
        self.grads["bias"] = bias_grad

        self.grads["weight"] = dout.T @ self.x
        dx = dout @ self.params["weight"]
        assert dx.shape == self.x.shape
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def update_weights(self, lr):
        """Update the weights and biases based on the last backward pass with a given learning rate"""
        self.params["weight"] -= self.grads["weight"] * lr
        self.params["bias"] -= self.grads["bias"] * lr

    def print_debug(self):
        print(
            f"Name: {self.name}, "
            f"weight shape: {self.params['weight'].shape}, weight grad shape: {self.grads['weight'].shape}, "
            f"bias shape: {self.params['bias'].shape}, bias grad shape: {self.grads['bias'].shape}, "
            f"input shape: {'none' if self.x is None else self.x.shape}"
        )


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        positive_mask = x > 0
        out = np.zeros_like(x)
        out[positive_mask] = x[positive_mask]
        out[~positive_mask] = np.exp(x[~positive_mask]) - 1
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        positive_mask = self.x > 0
        dx = np.ones_like(self.x)
        # dx[positive_mask] = 1
        dx[~positive_mask] = np.exp(self.x[~positive_mask])
        dx = dout * dx

        assert dx.shape == self.x.shape
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def update_weights(self, lr):
        pass

    def print_debug(self):
        print(
            f"Name: ELU, " f"input shape: {'none' if self.x is None else self.x.shape}"
        )


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        b = x.max()  # normalization factor to avoid overflow
        y = np.exp(x - b)
        out = y / y.sum(axis=1, keepdims=True)
        self.y = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        # z, da shapes - (m, n)
        m, n = self.x.shape
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum("ij,ik->ijk", self.y, self.y)  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum("ij,jk->ijk", self.y, np.eye(n, n))  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        dx = np.einsum("ijk,ik->ij", dSoftmax, dout)  # (m, n)

        assert dx.shape == self.x.shape
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def update_weights(self, lr):
        pass

    def print_debug(self):
        print(
            f"Name: SoftMax, "
            f"input shape: {'none' if self.x is None else self.x.shape}"
        )


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = (
            -(one_hot(y, num_classes=self.num_classes) * np.log(x))
            .sum(axis=1)
            .mean(axis=0)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = -one_hot(y, num_classes=self.num_classes) / x / x.shape[0]
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
