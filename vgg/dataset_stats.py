# %%

import argparse

import torch
from datasets import get_imagenet_dataloaders

# %%


params = argparse.ArgumentParser("Calculate [Imagenet-like] dataset stats")
params.add_argument("data")

params.add_argument("--data_dir", type=str, metavar="DIR", required=True,
    help="Path to the ImageNet dataset")

# What to calculate:
# - RGB mean, RGV average
# - class distribution

args = params.parse_args()

train, val = get_imagenet_dataloaders(data_dir=args.data_dir, resize=None)
