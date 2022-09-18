"""Neural Network Trainer"""
# %%

from collections import defaultdict
import time

from typing import Callable

# from logging.config import valid_ident
# from types import SimpleNamespace

# import contextlib
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
# import torchvis
# ion

from tqdm.autonotebook import tqdm

# from torchinfo import summary
# import visdom
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

# %%

def train(model: nn.Module,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            loss_fn: Callable,
            train_dl: DataLoader,
            val_dl: DataLoader | None,
            grad_accum=1,
            start_epoch=1,
            fp16=False,
            grad_scaling=False):
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

    # epoch_metrics["epoch"]
    # epoch_metrics["train_loss"]
    # epoch_metrics["train_acc"]
    # epoch_metrics["val_loss"]
    # epoch_metrics["val_acc"]
    # epoch_metrics["time"]

    # iter_metrics["batch"]
    # iter_metrics["epoch"]
    # iter_metrics["train_loss"]
    # iter_metrics["train_acc"]

    wandb.define_metric("step_*", step_metric="_step")
    wandb.define_metric("*", step_metric="epoch")

    epoch_pbar = tqdm(desc="Training", total=epochs, unit="Epoch")
    batch_pbar = tqdm(unit="batch")

    device = next(model.parameters()).device #pylint: disable=redefined-outer-name

    # Just in case
    optimizer.zero_grad()
    step = 0

    # if not fp16:
    #     atc = contextlib.nullcontext
    # else:
    #     atc = autocast

    scaler = GradScaler(enabled=grad_scaling)

    # We only run a small piece with torch.enable_grad()

    with torch.no_grad():
        for epoch in range(start_epoch, start_epoch + epochs):
            batch_pbar.reset(total=len(train_dl) // grad_accum)
            batch_pbar.set_description(f"Epoch {epoch}/{epochs}")

            epoch_train_loss = 0.
            epoch_val_loss = 0.

            n_train_correct = 0
            n_train_samples = 0

            n_val_correct = 0
            n_val_samples = 0

            start_time = time.time()

            virt_batch_loss = 0.
            virt_batch_correct = 0
            virt_batch_samples = 0

            model.train()
            for batch, (X, y) in enumerate(train_dl):
                X = X.to(device)
                y = y.to(device)

                with torch.enable_grad():
                    with autocast(enabled=fp16):
                        y_hat = model(X)
                        loss = loss_fn(y_hat, y)
                        loss = loss / grad_accum
                    scaler.scale(loss).backward()  # type: ignore

                # Accumulate batch metrics
                virt_batch_loss += loss.item()
                virt_batch_correct += n_correct(y_hat, y).item()
                virt_batch_samples += y.shape[0]

                if (batch+1) % grad_accum == 0 or (batch+1) == len(train_dl):
                    step += 1
                    # batch_pbar.write(f"batch {batch}, step {step}")
                    scaler.step(optimizer)

                    grad_scale = scaler.get_scale()
                    scaler.update()
                    # optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    virt_batch_acc = virt_batch_correct / virt_batch_samples

                    # For per-epoch stats
                    epoch_train_loss += virt_batch_loss
                    n_train_correct += virt_batch_correct
                    n_train_samples += virt_batch_samples

                    iter_stats = {
                        "epoch" : epoch,
                        "batch" : (batch + 1) // grad_accum,
                        "step" : step,
                        "step_train_loss" : virt_batch_loss,
                        "step_train_acc" : virt_batch_acc,
                        "step_grad_scale" : grad_scale
                    }

                    wandb.log(iter_stats, step=step)

                    for key, val in iter_stats.items():
                        iter_metrics[key].append(val)

                    batch_pbar.set_postfix_str(f"train_acc: {virt_batch_acc:.4f} train_loss: {virt_batch_loss:.4f}")
                    batch_pbar.update()

                    virt_batch_loss = 0
                    virt_batch_correct = 0
                    virt_batch_samples = 0

            train_end_time = time.time()

            # -1 to log the epoch stats at the last training step of the previous epoch
            # step = epoch * len(train_dl) - 1
            epoch_stats = {
                "epoch" : epoch,
                "train_loss": epoch_train_loss / len(train_dl),
                "train_acc": n_train_correct / n_train_samples,
                "train_time": tqdm.format_interval(train_end_time - start_time)
            }

            # Log metrics here in case we crash during validation
            wandb.log(epoch_stats, step=step)

            # epoch_metrics["epoch"].append(epoch)
            # epoch_metrics["train_loss"].append(epoch_train_loss / (n_train_batches + 1))
            # epoch_metrics["train_acc"].append(n_train_correct / n_train_samples)

            if val_dl:
                model.eval()
                batch_pbar.reset(total=len(val_dl))
                with torch.no_grad(), autocast():
                    for batch, (X, y) in enumerate(val_dl):
                        X = X.to(device)
                        y = y.to(device)

                        y_hat = model(X)
                        loss = loss_fn(y_hat, y).item()
                        acc = accuracy(y_hat.detach(), y).item()

                        n_val_correct += n_correct(y_hat.detach(), y).item()
                        n_val_samples += y.shape[0] # In case we get a partial batch
                        epoch_val_loss += loss

                        batch_pbar.set_postfix_str(f"val_acc: {acc:.4f} val_loss: {loss:.4f}")
                        batch_pbar.update()

                    val_end_time = time.time()

                    epoch_stats |= {
                        "val_loss": epoch_val_loss / len(val_dl),
                        "val_acc": n_val_correct / n_val_samples,
                        "val_time": tqdm.format_interval(val_end_time - train_end_time)
                    }

                # epoch_metrics["epoch_val_loss"].append(epoch_val_loss / (n_val_batches + 1))
                # epoch_metrics["epoch_val_acc"].append(n_val_correct / ())




                # wandb.log({
                #     "epoch" : epoch,
                #     "val_loss" : epoch_val_loss / (n_val_batches + 1),
                #     "val_acc" : n_val_correct / n_val_samples,
                #     "epoch_train_loss" : epoch_train_loss / (n_train_batches + 1),
                #     "epoch_train_acc" : n_train_correct / n_train_samples
                # })


            # # batch_pbar.close()

            epoch_stats |= {
                "time": tqdm.format_interval(time.time() - start_time)
            }

            wandb.log(epoch_stats, step=step)

            for key, val in epoch_stats.items():
                epoch_metrics[key].append(val)

            # epoch_metrics["time"].append(tqdm.format_interval(val_end_time - start_time))
            # epoch_metrics["train_time"].append(tqdm.format_interval(train_end_time - start_time))
            # epoch_metrics["val_time"].append(tqdm.format_interval(val_end_time - train_end_time))

            # wandb.log({
            #     "epoch" : epoch,
            #     "time" : epoch_metrics["time"][-1],
            #     "train_time" : epoch_metrics["train_time"][-1],
            #     "val_time": epoch_metrics["val_time"][-1],
            # })

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
