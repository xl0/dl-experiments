# %%

import argparse
from collections import defaultdict
import time

from typing import Callable

# from logging.config import valid_ident
# from types import SimpleNamespace

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from tqdm.autonotebook import tqdm

# from torchinfo import summary
import visdom
import wandb

import utils

def accuracy(y_hat, y):
    """Accuracy for a single-label classification
    """
    return (y_hat.argmax(dim=-1) == y).float().mean()

def n_correct(y_hat, y):
    """Number of correctly predicted samples in a single-label classificaiton
    """
    return (y_hat.argmax(dim=-1) == y).int().sum()

def train(model: nn.Module,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            loss_fn: Callable,
            train_dl: DataLoader,
            val_dl: DataLoader | None,
            grad_accum=1,
            start_epoch=1):
    """Training loop

    Args:
        model: nn.Module, already moved to the correct device.
        epochs: int
        optimizer: ["SGD", "AdamW"],
        loss_fn: loss, callable
        train_dl: Training DataLoader
        val_dl: Validaiton DataLoader

    Returns:
        History dictionary

    """
    epoch_metrics = defaultdict(list)
    iter_metrics = defaultdict(list)

    epoch_metrics["epoch"]
    epoch_metrics["train_loss"]
    epoch_metrics["train_acc"]
    epoch_metrics["val_loss"]
    epoch_metrics["val_acc"]
    epoch_metrics["time"]

    iter_metrics["batch"]
    iter_metrics["epoch"]
    iter_metrics["train_loss"]
    iter_metrics["train_acc"]


    epoch_pbar = tqdm(desc="Training", total=epochs, unit="Epoch")
    batch_pbar = tqdm(unit="batch")

    device = next(model.parameters()).device #pylint: disable=redefined-outer-name
    epoch_pbar.write(f"Running on {device}")


    # Just in case
    # optimizer.zero_grad()

    for epoch in range(start_epoch, start_epoch + epochs):
        ### Train

        batch_pbar.reset(total=len(train_dl))
        batch_pbar.set_description(f"Epoch {epoch - start_epoch +1}/{epochs}")

        epoch_train_loss = 0.
        epoch_val_loss = 0.

        n_train_correct = 0
        n_train_samples = 0
        n_train_batches = 0

        n_val_correct = 0
        n_val_samples = 0
        n_val_batches = 0

        start_time = time.time()

        model.train()
        for batch, (X, y) in enumerate(train_dl):
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            loss.backward()

            optimizer.step()



            with torch.no_grad():
                # print(y)
                # print(y_hat)

                # for name, p in model.named_parameters():
                #     print(f"Param: {name}: min={p.min():.4f} mean={p.mean():.4f} var={p.var():.4f} max={p.max():.4f}")

                # for name, p in model.named_parameters():
                #     grad = p.grad
                #     print(f"Grad: {name}: min={grad.min():.4f} mean={grad.mean():.4f} var={grad.var():.4f} max={grad.max():.4f}")

                acc = accuracy(y_hat, y).item()

                iter_metrics["batch"].append(batch)
                iter_metrics["epoch"].append(epoch)
                iter_metrics["train_loss"].append(loss.item())
                iter_metrics["train_acc"].append(acc)

                wandb.log({
                    "epoch" : epoch,
                    "batch" : batch,
                    "train_loss" : loss,
                    "train_acc" : acc
                })


                n_train_batches = batch
                n_train_correct += n_correct(y_hat, y).item()
                n_train_samples += y.shape[0]

                epoch_train_loss += loss.item()

                batch_pbar.set_postfix_str(f"train_acc: {acc:.4f} train_loss: {loss.item():.4f}")

                batch_pbar.update()

            optimizer.zero_grad()


        epoch_metrics["epoch"].append(epoch)
        epoch_metrics["train_loss"].append(epoch_train_loss / (n_train_batches + 1))
        epoch_metrics["train_acc"].append(n_train_correct / n_train_samples)

        wandb.log({
            "epoch" : epoch,
            "train_loss" : epoch_train_loss / (n_train_batches + 1),
            "train_acc" : n_train_correct / n_train_samples
        })

        train_end_time = time.time()

        if val_dl:
            model.eval()
            batch_pbar.reset(total=len(val_dl))
            with torch.no_grad():
                for batch, (X, y) in enumerate(val_dl):
                    X = X.to(device)
                    y = y.to(device)

                    y_hat = model(X)
                    loss = loss_fn(y_hat, y)


                    acc = accuracy(y_hat.detach(), y).item()
                    n_val_batches = batch
                    n_val_correct += n_correct(y_hat.detach(), y).item()
                    n_val_samples += y.shape[0]
                    epoch_val_loss += loss.item()

                    batch_pbar.set_postfix_str(f"val_acc: {acc:.4f} val_loss: {loss.item():.4f}")
                    batch_pbar.update()

            epoch_metrics["val_loss"].append(epoch_val_loss / (n_val_batches + 1))
            epoch_metrics["val_acc"].append(n_val_correct / n_val_samples)
            wandb.log({
                "epoch" : epoch,
                "val_loss" : epoch_val_loss / (n_val_batches + 1),
                "val_acc" : n_val_correct / n_val_samples
            })


        # # batch_pbar.close()

        val_end_time = time.time()

        epoch_metrics["time"].append(tqdm.format_interval(val_end_time - start_time))
        epoch_metrics["train_time"].append(tqdm.format_interval(train_end_time - start_time))
        epoch_metrics["val_time"].append(tqdm.format_interval(val_end_time - train_end_time))

        wandb.log({
            "epoch" : epoch,
            "time" : epoch_metrics["time"][-1],
            "train_time" : epoch_metrics["train_time"][-1],
            "val_time": epoch_metrics["val_time"][-1],
        })

        epoch_pbar.update()

        if epoch == 1:
            tqdm.write(utils.metrics_names_pretty(epoch_metrics))
        tqdm.write(utils.metrics_last_pretty(epoch_metrics))

    batch_pbar.close()
    epoch_pbar.close()


    return epoch_metrics, iter_metrics


def predict(model: nn.Module, X: torch.Tensor):
    """Run inference on X

    """
    orig_device = X.device
    device = next(model.parameters()).device #pylint: disable=redefined-outer-name
    with torch.no_grad():
        X = X.to(device)
        y_hat = model(X)
        preds = y_hat.argmax(dim=-1).to(orig_device)

    return preds



# %%
