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

"""Main driver script to run the code."""
import os
import argparse
import torch
from learner import Learner
import json
import warnings
from utils import get_device


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    # Adam: limit tqdm logging frequency
    parser.add_argument(
        "--print_tqdm_interval",
        type=float,
        default=1.0,
        help="min and max interval to print tqdm progress bars to avoid polluting the Snellius log files too much",
    )
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )
    # Adam: add option to quickly go through the training and evaluation to catch errors in the code
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="limit number of batches in each training and evaluation loop to aid testing",
    )
    parser.add_argument(
        "--square_size",
        type=int,
        default=8,
        help="size of each square in checkboard prompt",
    )
    # optimization
    parser.add_argument("--optim", type=str, default="sgd", help="optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=40, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="number of steps to warmup for"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--patience", type=int, default=1000)

    # model
    parser.add_argument("--model", type=str, default="clip")
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["visual_prompt", "deep_prompt"],
        default="visual_prompt",
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=4,
        help="number of learnable deep prompts to use",
    )
    parser.add_argument(
        "--injection_layer",
        type=int,
        default=0,
        help="id of transformer layer to inject prompt into",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="padding",
        choices=[
            "padding",
            "random_patch",
            "fixed_patch",
        ],
        help="choose visual prompting method",
    )
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="size for visual prompts"
    )
    # Adam
    parser.add_argument(
        "--prompt_init_method",
        type=str,
        default="random",
        choices=[
            "random",
            "empty",
        ],
        help="choose visual prompting method",
    )
    parser.add_argument(
        "--text_prompt_template",
        type=str,
        default="This is a photo of a {}",
    )
    parser.add_argument(
        "--visualize_prompt",
        action="store_true",
        help="visualize the (randomly initialized) prompt and save it to a file for debugging",
    )

    # dataset
    parser.add_argument("--root", type=str, default="./data", help="dataset")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="whether to add noise to the test images",
    )

    # other
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for initializing training"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./save/models", help="path to save models"
    )
    parser.add_argument(
        "--image_dir", type=str, default="./save/images", help="path to save images"
    )
    parser.add_argument("--filename", type=str, default=None, help="filename to save")
    parser.add_argument("--trial", type=int, default=1, help="number of trials")
    parser.add_argument(
        "--resume", type=str, default=None, help="path to resume from checkpoint"
    )
    parser.add_argument(
        "--resume_best",
        default=False,
        action="store_true",
        help="resume best model from default checkpoint",
    )
    parser.add_argument(
        "--evaluate", default=False, action="store_true", help="evaluate model test set"
    )
    parser.add_argument("--gpu", type=int, default=None, help="gpu to use")
    parser.add_argument(
        "--use_wandb", default=False, action="store_true", help="whether to use wandb"
    )

    args = parser.parse_args()

    args.num_workers = min(args.num_workers, os.cpu_count())

    args.filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}".format(
        args.prompt_type,  # Adam: add prompt type to avoid visual and deep prompting models overwriting each other
        args.method,
        args.prompt_size,
        args.injection_layer,
        args.prompt_num,
        args.prompt_init_method,
        args.dataset,
        args.model,
        args.arch,
        args.optim,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.warmup,
        args.trial,
    )

    args.device = get_device()
    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # Adam: option to easily resume from the best saved model for the given parameters to do additional evaluation
    # without manually specifying the default file name in the job file
    if args.resume_best:
        args.resume = os.path.join(args.model_folder, "model_best.pth.tar")

    return args


def main():
    args = parse_option()
    print(args)

    if args.visualize_prompt:
        os.makedirs("images", exist_ok=True)

    learn = Learner(args)

    # Adam: collect and save results
    results_dir = "results_vp"
    os.makedirs(results_dir, exist_ok=True)
    top1_val_acc, top1_test_acc = None, None

    if args.evaluate:
        if not args.resume:
            warnings.warn(
                f"Evaluation is requested but no previously trained checkpoint is loaded. "
                f"An evaluation of the default model weights is performed, please make sure this is what you intended."
            )
        top1_test_acc = learn.evaluate("test")
    else:
        learn.run()
        learn.resume_best_checkpoint()  # Adam: force reloading the best checkpoint before model evaluation

        top1_val_acc = learn.evaluate("valid")
        top1_test_acc = learn.evaluate("test")

    # Adam: save results into a single directory to make it easier to plot in the end
    result = vars(args)
    result["top1_val_acc"] = top1_val_acc
    result["top1_test_acc"] = top1_test_acc
    result["best_epoch"] = learn.best_epoch
    fn = f"{args.dataset}_{args.prompt_type}_{args.method}_{args.prompt_num}_{args.injection_layer}_{args.prompt_size}_{args.prompt_init_method}_{args.test_noise}.json"
    with open(f"{results_dir}/{fn}", "w") as f:
        json.dump(result, f)

    # Adam: if visualize prompt is requested then also visualize after training/loading the best model,
    # to see what we've actually learnt
    if args.visualize_prompt:
        learn.clip.visualize_prompt(
            filename=f"images/prompt_{args.method}_{args.prompt_size}_{args.prompt_init_method}_after_training",
            device=args.device,
        )


if __name__ == "__main__":
    main()
