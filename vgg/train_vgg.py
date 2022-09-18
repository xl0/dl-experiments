#!/usr/bin/env python3

# %%

import argparse

from collections import defaultdict

import json
import torch
import torch.nn as nn

# import visdom
import wandb

import datasets
import vgg
import utils
import train


# %%

class Config(defaultdict):
    """A dict-like structure with attribute keys"""
    def __init__(self, *args, **kwargs):
        super().__init__(lambda : None, *args, **kwargs)
        self.__dict__ = self
    def __getattr__(self, k):
        return self[k] if k in self else None


# Can be overridden by command line arguments or wandb
config = Config(
    # Dataset
    # data_dir="/home/xl0/work/ml/datasets/MNIST",
    # data_dir="/home/xl0/work/ml/datasets/ImageNet",
    data_dir="/home/xl0/work/ml/datasets/imagenette2",
    cls_json=None,
    # dataset="MNIST",
    # dataset="ImageNet",
    dataset="imagenette2",
    resize=128,
    norm=True,

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
    grad_accum=1,
    fp16=True,
    grad_scaling=True,
    # bs=32,
    epochs=100,

    overfit_batches=0,
    overfit_len=50000,

    visdom=False,
    wandb=True,
    wandb_mode="online",
    # wandb_offline=False,
    wandb_project="vgg"
)

# %%

# Argparse: Allow multi-line help entries and show default values
class Formatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Argparse formatter, inherits from:
        - argparse.RawTextHelpFormatter - allows \\n in help.
        - pargarse.ArgumentDefaultsHelpFormatter - show defaults.
    """

parser = argparse.ArgumentParser(description='Train VGG on the ImageNet Dataset',
                                formatter_class=Formatter)

parser_ds = parser.add_argument_group("Dataset parameters")
parser_ds.add_argument("--dataset", type=str, metavar="DATASET",
    default=config.dataset,
    choices=["MNIST", "ImageNet", "imagenette2"],
    help="Dataset to use")
parser_ds.add_argument("--data_dir", type=str, metavar="DIR", required=True,
    help="Path to the ImageNet dataset")
parser_ds.add_argument("--cls_json", type=str, metavar="file.json",
    help="""Path the the class labels json file.
    See https://github.com/anishathalye/imagenet-simple-labels""")


parser_aug = parser.add_argument_group("Data Augmentation")
parser_aug.add_argument("--resize", type=int, default=config.resize,
    help="Resize the images to this size")
parser_aug.add_argument("--norm", type=bool, default=config.norm,
    action=argparse.BooleanOptionalAction,
    help="Apply Imagenet mean/var normalization")

parser_model = parser.add_argument_group("Model paramters")
parser_model.add_argument("--dropout", type=float, default=config.dropout,
    help="Dropout probability for FC layers")
parser_model.add_argument("--init", type=str, default=config.init,
    choices=["paper_normal", "paper_glorot", "pytorch"],
    help="Inintilize weights")

parser_optim = parser.add_argument_group("Optimization paramters")
parser_optim.add_argument("--lr", type=float, metavar="LR", default=config.lr,
    help="Base learning rate")
parser_optim.add_argument("--optim", type=str, default=config.optim,
    choices=["SGD", "Agam"],
    help="Optimizer")
parser_optim.add_argument('--weight_decay', type=float, default=config.weight_decay, metavar="WD",
    help='Weight decay')
parser_optim.add_argument("--momentum", type=float, default=config.momentum, metavar="MOM",
    help="Momentum")

parser_optim.add_argument("--epochs", type=int, default=config.epochs, metavar="N",
    help="Number of epochs to train")
parser_optim.add_argument("--bs", type=int, default=config.bs, metavar="N",
    help="Training Batch size")
parser_optim.add_argument("--val_bs", type=int, metavar="N",
    help="Valudation Batch size (default: bs*2)")
parser_optim.add_argument("--grad_accum", type=int, default=config.grad_accum, metavar="N",
    help="Split batches into N chunks, forward/back-prob on at a time, optimizer step once")
parser_optim.add_argument("--fp16", type=bool, default=config.fp16,
    action=argparse.BooleanOptionalAction,
    help="Use Automatic Mixed Precision")
parser_optim.add_argument("--grad_scaling", type=bool, default=config.grad_scaling,
    action=argparse.BooleanOptionalAction,
    help="Use Gradient Scaling")
parser_optim.add_argument("--overfit_batches", type=int, metavar="N",
    default=config.overfit_batches,
    help="Overfit on N batches (from the dataset) instead of trainig. 0 Means train normally")
parser_optim.add_argument("--overfit_len", type=int, default=config.overfit_len, metavar="N",
    help="If in overfit mode, set the virtual Epoch to N \"samples\"")

parser_log = parser.add_argument_group("Logging")
parser_log.add_argument("--visdom", type=bool, default=config.visdom,
    action=argparse.BooleanOptionalAction,
    help="Use visdom (https://github.com/fossasia/visdom) for visualization")
parser_log.add_argument("--wandb", type=bool, default=config.wandb,
    action=argparse.BooleanOptionalAction,
    help="Log metrics to W&B")
parser_log.add_argument("--wandb_mode", type=str, default=config.wandb_mode,
    choices=["online", "offline", "disabled"],
    help="W&B mode")
parser_log.add_argument("--wandb_project", type=str, metavar="NAME",
    default=config.wandb_project,
    help="W&B project name")
parser_log.add_argument("--wandb_run", type=str, metavar="RUN",
    help="W&B run name (autogenerated if not specified)")
parser_log.add_argument("--fold", type=int,
    help="Pass a number if you want to run it multiple times")


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
    args["wandb_mode"] = "disabled"

run = wandb.init(config=args, project=args.wandb_project, name=args.wandb_run, mode=args.wandb_mode)
config = wandb.config

print(config)

if config.grad_accum != 1:
    assert config.bs % config.grad_accum == 0
    print(f"Accumulating gradient over {config.grad_accum} chunks")
    print(f"Chunk size is {config.bs // config.grad_accum}")

if "val_bs" not in config:
    config.val_bs = (config.bs // config.grad_accum)*2


vis = None
if config.visdom:
    raise NotImplementedError
    # vis = visdom.Visdom()

if config.dataset == "MNIST":
    train_dl, val_dl = datasets.get_mnist_dataloaders(
                                    data_dir=config.data_dir,
                                    bs=config.bs // config.grad_accum,
                                    val_bs=config.val_bs,
                                    resize=config.resize,
                                    overfit_batches=config.overfit_batches,
                                    overfit_len=config.overfit_len)

elif config.dataset in  ["ImageNet", "imagenette2"]:
    train_dl, val_dl = datasets.get_imagenet_dataloaders(
                                    data_dir=config.data_dir, cls_json=config.cls_json,
                                    bs=config.bs // config.grad_accum,
                                    val_bs=config.val_bs,
                                    resize=config.resize,
                                    norm=config.norm,
                                    grad_accum=config.grad_accum,
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

if "grad_accum" not in config:
    config.grad_accum = 1


epoch_metrics, iter_metrics = train.train(model=model,
                                            epochs=config.epochs,
                                            optimizer=optim,
                                            loss_fn=criterion,
                                            train_dl=train_dl,
                                            val_dl=val_dl,
                                            grad_accum=config.grad_accum,
                                            fp16=config.fp16,
                                            grad_scaling=config.grad_scaling)

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
