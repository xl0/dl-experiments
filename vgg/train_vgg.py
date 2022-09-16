#!/usr/bin/env python3

# %%

import argparse
import os
import json
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms.functional as tF


# from tqdm.autonotebook import tqdm

from torchinfo import summary
from matplotlib import pyplot as plt

from types import SimpleNamespace

import wandb
import visdom

import datasets
import vgg
import utils
import train


# %%

# Can be overridden by command line arguments
config = SimpleNamespace(
    # Dataset
    # data_dir="/home/xl0/work/ml/datasets/MNIST",
    # data_dir="/home/xl0/work/ml/datasets/ImageNet",
    data_dir="/home/xl0/work/ml/datasets/imagenette2",
    cls_json=None,
    # dataset="MNIST",
    # dataset="ImageNet",
    dataset="imagenette2",
    resize=128,

    # init="paper_normal",
    # init="pytorch",
    init="paper_glorot",

    dropout=0.,

    # Optimizer
    optim="SGD",
    lr=0.01,
    # weight_decay=0, #4e-5,
    weight_decay=5e-4, #4e-5,
    #weight_decay=4e-5,
    momentum=0.9,

    bs=256,
    # bs=32,
    epochs=100,

    overfit_batches=0,
    overfit_len=50000,

    visdom=True,
    wandb=False,
    wandb_project="vgg"

)

# %%
parser = argparse.ArgumentParser(description='Train VGG on the ImageNet Dataset',
                                formatter_class=argparse.RawTextHelpFormatter)

parser_ds = parser.add_argument_group("Dataset parameters")
parser_ds.add_argument("--dataset", type=str, metavar="DATASET", required=True,
    help="Dataset to use: [MNIST, ImageNet, imagenette2]")
parser_ds.add_argument("--data_dir", type=str, metavar="DIR", required=True,
    help="Path to the ImageNet dataset")
parser_ds.add_argument("--cls_json", type=str, metavar="file.json",
    help="""Path the the class labels json file.
    See https://github.com/anishathalye/imagenet-simple-labels""")


parser_aug = parser.add_argument_group("Data Augmentation")
parser_aug.add_argument("--resize", type=int, default=config.resize,
    help="Resize the images to this size")

parser_model = parser.add_argument_group("Model paramters")
parser_model.add_argument("--dropout", type=float, default=config.dropout,
    help="Dropout probability for FC layers")
parser_model.add_argument("--init", type=str, default=config.init,
    help="Inintilize weights: [paper_normal, paper_glorot, pytorch]")


parser_optim = parser.add_argument_group("Optimization paramters")
parser_optim.add_argument("--lr", type=float, metavar="LR", default=config.lr,
    help="Base learning rate")
parser_optim.add_argument("--optim", type=str, default=config.optim,
    help="Optimizer")
parser_optim.add_argument('--weight-decay', type=float, default=config.weight_decay, metavar="WD",
    help='Weight decay')
parser_optim.add_argument("--momentum", type=float, default=config.momentum, metavar="MOM",
    help="Momentum")

parser_optim.add_argument("--epochs", type=int, default=config.epochs, metavar="N",
    help="Number of epochs to train")
parser_optim.add_argument("--bs", type=int, default=config.bs, metavar="N",
    help="Training Batch size")
parser_optim.add_argument("--overfit-batches", type=int, metavar="N",
    default=config.overfit_batches,
    help="Overfit on N batches (from the dataset) instead of trainig. 0 Means train normally")
parser_optim.add_argument("--overfit-len", type=int, default=config.overfit_len, metavar="N",
    help="If in overfit mode, set the virtual Epoch to N \"samples\"")

parser_log = parser.add_argument_group("Logging")
parser_log.add_argument("--visdom", type=bool, default=config.visdom,
    help="Use visdom (https://github.com/fossasia/visdom) for visualization")
parser_log.add_argument("--wandb", type=bool, default=config.wandb,
    help="Log metrics to W&B")
parser_log.add_argument("--wandb-project", type=str, metavar="NAME",
    default=config.wandb_project,
    help="W&B project name")


def init_weights_paper_normal(module):
    """Init weights with normal(0, 0.01) like originally in the paper
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            module.bias.zero_()

def init_weights_paper_glorot(module):
    """Init weights using glorot like they later found out works better
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=1.414)
        if module.bias is not None:
            module.bias.zero_()


# %%

if utils.is_notebook():
    args = config
    print("Not parsing args when running in an ipython notebook")
else:
    args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "offline"

run = wandb.init(config=args, project=args.wandb_project)
config = wandb.config

print(args)
print(config)


vis = None
if config.visdom:
    vis = visdom.Visdom()

if config.dataset == "MNIST":
    train_dl, val_dl = datasets.get_mnist_dataloaders(config.data_dir, config.bs,
                                            resize=config.resize,
                                            overfit_batches=config.overfit_batches,
                                            overfit_len=config.overfit_len)
elif config.dataset in  ["ImageNet", "imagenette2"]:
    train_dl, val_dl = datasets.get_imagenet_dataloaders(config.data_dir, config.cls_json,
                                            config.bs, resize=config.resize,
                                            overfit_batches=config.overfit_batches,
                                            overfit_len=config.overfit_len)
else:
    raise ValueError(f"Unknown dataset {config.dataset}")

data_shape=train_dl.dataset[0][0].shape

n_channels=data_shape[0]
n_classes=len(train_dl.dataset.classes)


device = torch.device("cuda")

model = vgg.VGG11(n_classes=n_classes, in_chans=n_channels)

model.to(device)

# We need to figure out the lazy layer sizes before init.
model(torch.randn((1, *data_shape)).to(device))


with torch.no_grad():
    if config.init == "paper_normal":
        model.apply(init_weights_paper_normal)
    elif config.init == "paper_glorot":
        model.apply(init_weights_paper_glorot)
    elif config.init == "pytorch":
        pass # Nothing to do
    else:
        raise ValueError(f"Unknown init method {config.init}")

if config.dropout != 0.5:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            print(f"Setting dropout to {config.dropout}")
            module.p = config.dropout


if config.optim == "SGD":
    optim = torch.optim.SGD(model.parameters(),
                            lr=config.lr,
                            weight_decay=config.weight_decay,
                            momentum=config.momentum)
elif config.optim == "Adam":
    optim = torch.optim.Adam(model.parameters(),
                            lr=config.lr,
                            weight_decay=config.weight_decay)
else:
    raise ValueError(f"Unsupported optimizer '{config.optim}'")

criterion = nn.CrossEntropyLoss()



epoch_metrics, iter_metrics = train.train(model=model,
                                            epochs=config.epochs,
                                            optimizer=optim,
                                            loss_fn=criterion,
                                            train_dl=train_dl,
                                            val_dl=val_dl)

wandb.finish()

# %%



# X, y = next(iter(train_dl))
# preds = train.predict(model, X)
# annotated = utils.annotate_batch(X, y, preds, idx2class=train_dl.dataset.classes)

# vis.images(annotated)

metrics = {
    "epoch_metrics" : epoch_metrics,
    "iter_metrics" : iter_metrics
}

with open("results.json", "w+") as f: #pylint: disable=unspecified-encoding
    json.dump(metrics, f)


# %%
# epoch_metrics, iter_metrics = train.train(model=model,
#                 epochs=100,
#                 optimizer=optim,
#                 loss_fn=criterion,
#                 train_dl=train_dl,
#                 val_dl=val_dl)


# %%


# for name, p in model.named_parameters():
#     print(f"Param: {name}: min={p.min():.4f} mean={p.mean():.4f} var={p.var():.4f} max={p.max():.4f}")

# for name, p in model.named_parameters():
#     grad = p.grad
#     print(f"Grad: {name}: min={grad.min():.4f} mean={grad.mean():.4f} var={grad.var():.4f} max={grad.max():.4f}")

# def main():
#     if utils.is_notebook():
#         print("Not parsing args when running in an ipython notebook")
#         args = noteboook_config
#     else:
#         args = parser.parse_args()
#     print(args)

#     train_dl, val_dl = get_dataloaders(args.data_dir, args.cls_json, args.bs)

#     device = torch.device("cuda")

#     model = vgg.VGG11(n_classes=5)
#     model.to(device)

#     if args.optimizer == "SGD":
#         optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         raise ValueError(f"Unsupported optimizer '{args.optimizer}'")
#     criterion = nn.CrossEntropyLoss()

#     history = train.train(model=model,
#                     epochs=args.epochs,
#                     optimizer=optim,
#                     criterion=criterion,
#                     train_dl=train_dl,
#                     val_dl=val_dl)






# %%
