################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision
from tqdm import tqdm
import json

from cifar100_utils import (
    get_train_validation_set,
    get_test_set,
    set_dataset,
    dataset_name,
)


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Returns the device for PyTorch to use."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # mac MPS support: https://pytorch.org/docs/stable/notes/mps.html
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = "mps"
    return device


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    # Don't forget pretrained=True as I did first..
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.requires_grad_(False)  # freeze all layers
    # print(model) # used for manually checking the model structure and picking the last layer to unfreeze

    # Replace the last layer with one that has the correct output shape
    # It also set requires_grad=True for this specific layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight.data.normal_(0, 0.01)  # initialize as requested in the exercise

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(
    model,
    lr,
    batch_size,
    epochs,
    data_dir,
    checkpoint_name,
    device,
    augmentation_name=None,
    print_tqdm_interval=0.1,
    max_batches=0,
):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(
        data_dir, augmentation_name=augmentation_name
    )
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    writer = SummaryWriter("runs/")

    # Initialize the optimizer (Adam) to train the last layer of the model.
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    best_model_accuracy = -np.inf
    best_model_in_epoch = -1

    total_step = len(train_loader)
    for epoch in (
        epoch_pbar := tqdm(
            range(1, epochs + 1),
            mininterval=print_tqdm_interval,
            maxinterval=print_tqdm_interval,
        )
    ):
        epoch_pbar.set_description(f"Epoch: {epoch}")

        running_loss = 0.0

        correct = 0
        processed_samples = 0
        model.train()
        for batch_idx, (data_, target_) in (
            batch_pbar := tqdm(
                enumerate(train_loader),
                total=total_step,
                leave=False,
                mininterval=args.print_tqdm_interval,
                maxinterval=args.print_tqdm_interval,
            )
        ):
            # Adam: break loop requested, to speed up testing locally
            if 0 < max_batches < batch_idx:
                break

            data_, target_ = data_.to(device), target_.to(device)

            # log some debug info in the very first batch
            if epoch == 1 and batch_idx == 0:
                # save model
                writer.add_graph(model, data_)

                # create grid of images
                img_grid = torchvision.utils.make_grid(data_)

                # write to tensorboard
                writer.add_image("first_training_batch", img_grid)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(data_)
            loss = loss_module(outputs, target_)
            loss.backward()
            optimizer.step()

            # calculate statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            processed_samples += target_.size(0)

            running_training_accuracy = correct / processed_samples
            writer.add_scalar(
                "training accuracy",
                running_training_accuracy,
                epoch * len(train_loader) + batch_idx,
            )
            running_training_loss = running_loss / processed_samples
            writer.add_scalar(
                "training loss",
                running_training_loss,
                epoch * len(train_loader) + batch_idx,
            )

            batch_pbar.set_description(f"Train batch: {batch_idx:3}", refresh=False)
            batch_pbar.set_postfix(
                {
                    "Batch loss": f"{loss:.2f}",
                    "Running training loss": running_training_loss,
                    "Running training accuracy": running_training_accuracy,
                },
                refresh=False,
            )

        val_accuracy = evaluate_model(
            model, val_loader, device, max_batches=max_batches
        )
        # writer.add_scalar(
        #     "validation loss",
        #     val_metrics["accuracy"],
        #     epoch * len(train_loader) + batch_idx,
        # )
        writer.add_scalar(
            "validation accuracy",
            val_accuracy,
            epoch * len(train_loader) + batch_idx,
        )

        if best_model_accuracy < val_accuracy:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_training_loss,
                },
                checkpoint_name,
            )
            best_model_accuracy = val_accuracy
            best_model_in_epoch = epoch
        epoch_pbar.set_postfix(
            {
                "Tr loss": f"{running_training_loss:.2f}",
                "Tr acc": f"{running_training_accuracy:.2f}",
                # "val loss": f"{val_metrics['loss']:.2f}",
                "val acc": f"{val_accuracy:.2f}",
            },
            refresh=False,
        )

        model.train()

    print(
        f"Best model trained in epoch {best_model_in_epoch} with accuracy: {best_model_accuracy}"
    )

    # Load the best model on val accuracy and return it.
    model, best_epoch = load_model(checkpoint_name, model)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def load_model(checkpoint_name, model):
    """Load model from a given checkpoint"""
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    # FIXME we should also load the optimizer if we plan to continue training
    # Now I only use model loading for evaluation, and the optimizer wasn't immediately available
    # at all places where I call this function, so I skipped this
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    # loss = checkpoint['loss']
    return model, epoch


def evaluate_model(model, data_loader, device, max_batches):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    epoch_preds = []
    epoch_labels = []
    batch_losses = []

    loss_module = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data_t, target_t) in (
            batch_pbar := tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                leave=False,
                mininterval=args.print_tqdm_interval,
                maxinterval=args.print_tqdm_interval,
            )
        ):
            # Adam: break loop requested, to speed up testing locally
            if 0 < max_batches < batch_idx:
                break

            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t)
            loss_t = loss_module(outputs_t, target_t)
            batch_losses.append(loss_t.item())
            _, pred_t = torch.max(outputs_t, dim=1)
            epoch_preds.append(pred_t.cpu().numpy())
            epoch_labels.append(target_t.cpu().numpy())
            batch_pbar.set_description(f"Eval batch: {batch_idx:5}")

    epoch_preds = np.concatenate(epoch_preds)
    epoch_labels = np.concatenate(epoch_labels)
    correct_preds = epoch_preds == epoch_labels
    accuracy = correct_preds.sum() / len(epoch_labels)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(
    lr,
    batch_size,
    epochs,
    data_dir,
    seed,
    augmentation_name,
    test_noise,
    print_tqdm_interval,
    evaluate=False,
    resume_best=False,
    max_batches=0,
):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = get_device()

    # Set up checkpoint to save to or load from
    model_dir = "save/models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_name = f"{model_dir}/restnet18_best_model_{dataset_name}_{augmentation_name}_{lr}_{batch_size}.pt"

    # Load the model
    model = get_model().to(device)

    # Load best previously trained model if required
    if resume_best:
        # Load the best model on val accuracy and return it.
        model, best_epoch = load_model(checkpoint_name, model)

    # Train the model, unless we are in evaluation-only mode
    if not evaluate:
        # Get the augmentation to use
        # ..we just pass the name here, nothing to do

        # Train the model
        model = train_model(
            model=model,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            data_dir=data_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            augmentation_name=augmentation_name,
            print_tqdm_interval=print_tqdm_interval,
            max_batches=max_batches,
        )

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir, test_noise)
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_accuracy = evaluate_model(model, test_loader, device, max_batches=max_batches)
    print(f"Test accuracy of best model: {test_accuracy}")

    results_dir = "results_resnet18"
    os.makedirs(results_dir, exist_ok=True)
    fn = f"resnet_{dataset_name}_{augmentation_name}_{test_noise}.json"
    result = {
        "dataset": dataset_name,
        "augmentation_name": augmentation_name,
        "test_noise": test_noise,
        "test_accuracy": test_accuracy,
    }
    with open(f"{results_dir}/{fn}", "w") as f:
        json.dump(result, f)

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")
    parser.add_argument("--epochs", default=30, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=123, type=int, help="Seed to use for reproducing results"
    )
    # Adam: limit tqdm logging frequency
    parser.add_argument(
        "--print_tqdm_interval",
        type=float,
        default=1.0,
        help="min and max interval to print tqdm progress bars to avoid polluting the Snellius log files too much",
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR100 dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        type=str,
        choices=["cifar100", "cifar10"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--augmentation_name", default=None, type=str, help="Augmentation to use."
    )
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="Whether to test the model on noisy images or not.",
    )
    # Adam: add some helpful flags like in part2 to only evaluate when checking test_noise
    parser.add_argument(
        "--resume_best",
        default=False,
        action="store_true",
        help="resume best model from default checkpoint",
    )
    parser.add_argument(
        "--evaluate", default=False, action="store_true", help="evaluate model test set"
    )
    # Adam: add option to quickly go through the training and evaluation to catch errors in the code
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="limit number of batches in each training and evaluation loop to aid testing",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop("dataset"))
    main(**kwargs)
