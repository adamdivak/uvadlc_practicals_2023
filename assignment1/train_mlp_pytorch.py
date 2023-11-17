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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

from train_mlp_numpy import visualize


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      targets: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # copied from the NumPy solution
    unique_labels = np.unique(targets)
    if len(predictions.shape) == 2:
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        predicted_labels = predictions
    assert len(predicted_labels) == len(targets)
    conf_mat = np.zeros((len(unique_labels), len(unique_labels)))
    for true_label_idx, true_label in enumerate(unique_labels):
        for predicted_label_idx, predicted_label in enumerate(unique_labels):
            conf_mat[true_label_idx, predicted_label_idx] = (
                (targets == true_label) & (predicted_labels == predicted_label)
            ).sum()
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.0):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
        beta: beta parameter for f_beta score calculation
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # copied from the NumPy solution
    tp = np.diagonal(confusion_matrix).sum()
    total = confusion_matrix.sum()

    metrics = {}
    metrics["accuracy"] = tp / total
    # precision: tp / (tp + fp), so we need all predictions for a given class in the denominator -> column-wise sum
    metrics["precision"] = np.diagonal(confusion_matrix) / confusion_matrix.sum(axis=0)
    # precision: tp / (tp + fn), so we need all true occurrences for a given class in the denominator -> row-wise sum
    metrics["recall"] = np.diagonal(confusion_matrix) / confusion_matrix.sum(axis=1)
    # f1_beta
    metrics["f1_beta"] = (
        (1 + beta**2)
        * metrics["precision"]
        * metrics["recall"]
        / (beta**2 * metrics["precision"] + metrics["recall"])
    )
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    epoch_preds = []
    epoch_labels = []
    batch_losses = []

    loss_module = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        for batch_idx, (data_t, target_t) in (
            batch_pbar := tqdm(
                enumerate(data_loader), total=len(data_loader), leave=False
            )
        ):
            # data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            outputs_t = model(data_t)
            loss_t = loss_module(outputs_t, target_t)
            batch_losses.append(loss_t.item())
            _, pred_t = torch.max(outputs_t, dim=1)
            epoch_preds.append(pred_t.numpy())
            epoch_labels.append(target_t.numpy())
            batch_pbar.set_description(f"Eval batch: {batch_idx:5}")

    epoch_preds = np.concatenate(epoch_preds)
    epoch_labels = np.concatenate(epoch_labels)
    cm = confusion_matrix(epoch_preds, epoch_labels)
    metrics = confusion_matrix_to_metrics(cm)
    metrics["loss"] = np.mean(batch_losses)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    assert use_batch_norm is False

    # TODO: Initialize model and loss module
    batch_size, s1, s2, s3 = list(next(iter(cifar10_loader["train"]))[0].shape)
    input_size = s1 * s2 * s3
    model = MLP(n_inputs=input_size, n_hidden=hidden_dims, n_classes=10)
    print(model)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    logging_info = {
        "training_losses": [],
        "training_accuracies": [],
        "validation_accuracies": [],
        "validation_losses": [],
    }
    best_model = None
    best_model_accuracy = -np.inf
    best_model_in_epoch = -1

    train_loader = cifar10_loader["train"]

    total_step = len(train_loader)
    for epoch in (epoch_pbar := tqdm(range(1, epochs + 1))):
        epoch_pbar.set_description(f"Epoch: {epoch}")

        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total = 0
        for batch_idx, (data_, target_) in (
            batch_pbar := tqdm(
                enumerate(cifar10_loader["train"]),
                total=len(cifar10_loader["train"]),
                leave=False,
            )
        ):
            # data_, target_ = data_.to(device), target_.to(device)# on GPU
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            loss = loss_module(outputs, target_)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)

            batch_pbar.set_description(f"Train batch: {batch_idx:3}")
            batch_pbar.set_postfix({"Batch loss": f"{loss:.2f}"})

        training_accuracy = correct / total
        logging_info["training_accuracies"].append(training_accuracy)
        training_loss = running_loss / total_step
        logging_info["training_losses"].append(training_loss)

        val_metrics = evaluate_model(model, cifar10_loader["validation"])
        logging_info["validation_accuracies"].append(val_metrics["accuracy"])
        logging_info["validation_losses"].append(val_metrics["loss"])

        if best_model_accuracy < val_metrics["accuracy"]:
            # model.clear_cache()
            best_model = deepcopy(model)
            torch.save(model.state_dict(), "best_model_classification.pt")
            best_model_accuracy = val_metrics["accuracy"]
            best_model_in_epoch = epoch
        epoch_pbar.set_postfix(
            {
                "Tr loss": f"{training_loss:.2f}",
                "Tr acc": f"{training_accuracy:.2f}",
                "val loss": f"{val_metrics['loss']:.2f}",
                "val acc": f"{val_metrics['accuracy']:.2f}",
            }
        )

        model.train()

    print(
        f"Best model trained in epoch {best_model_in_epoch} with accuracy: {best_model_accuracy}"
    )
    # TODO: Test best model
    test_metrics = evaluate_model(best_model, cifar10_loader["test"])
    test_accuracy = test_metrics["accuracy"]
    # TODO: Add any information you might want to save for plotting
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, logging_info["validation_accuracies"], test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use this option to add Batch Normalization layers to the MLP.",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, validation_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    visualize(logging_info, model_name="MLP PyTorch")
