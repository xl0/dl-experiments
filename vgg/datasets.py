"""Generate train/test dataloaders
"""
# %%
# from typing import Tuple, Any, Optional, Sequence, Iterator
import os
import json
import torch
from torch.utils.data import DataLoader, Sampler, RandomSampler
import torchvision


#%%

__all__ = ["ImageNet", "get_imagenet_dataloaders", "get_mnist_dataloaders"]

class ImageNet(torchvision.datasets.ImageFolder):
    """ImageNet dataset that works on ImageNet directory structure

        Args:
            root: Path to an unpacked ImageNet dataset.
                Should incluide the train/val part.

            class_json: Path to a json file mapping class indices to human-readeable names
                See https://github.com/anishathalye/imagenet-simple-labels

            transform: transform function to apply to the image
            target_transform: transform funciton to apply to the label

        Returns:
            troch.utils.data.Dataset that generates tuples (PILImage, label_int)
    """
    def __init__(self, root, class_json=None, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.wnids = self.classes
        self.wnids_to_idx = self.class_to_idx

        if class_json:
            with open(class_json) as f: #pylint: disable=unspecified-encoding
                self.classes = json.load(f)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    # def __getitem__(self, index: int):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (sample, target) where target is class_index of the target class.
    #     """
    #     sample = torch.zeros(3, 224, 224)
    #     target = 0

    #     if self.transform is not None:
    #         sample = self.transform(sample)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return sample, target


class IndexSampler(Sampler):
    """Sample from a dataset based on indices
    """
    def __init__(self, indices):
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.indices)

def make_transforms(s=384, crop=224, norm=True, hflip=True,
                    color=(0.10, 0.10, 0.10, 0.10)):
    """Create a list of transforms

    Args:
        s: (int) isotropically rescale the image to have smaller size = s.
        crop: (int or tuple) Random crop size. If int -> (int, int)
        norm: (bool or dict)
            Normalize the image. If True, use ImageNet defaults.
            If dict, it should contain "mean" and "std" keys with values
            being lists that match the number of channels.
        hflip: (bool or float) Flip horizontally with probability p. True -> p=0.5.
        color: (Falsey or 4-tuple)
            Color jitter with brightness, contrast, saturation, hue from the tuple.
    """
    transforms = []

    if norm is True:
        norm = { "mean": (0.485, 0.456, 0.406),
                "std": (1.,1.,1.) }

    if hflip is True:
        hflip = 0.5

    if crop and not isinstance(crop, list | tuple):
        crop = (crop, crop)

    transforms = [
        # With size=int, Resize will limit the smaller size to s.
        torchvision.transforms.Resize(s) if s else None,
        torchvision.transforms.ToTensor(),

        torchvision.transforms.RandomCrop(crop) if crop else None,

        torchvision.transforms.RandomHorizontalFlip(hflip) if hflip else None,
        torchvision.transforms.ColorJitter(*color) if color else None,
        torchvision.transforms.Normalize(**norm) if norm else None,
    ]

    # Remove 'None' from the list.
    return list(filter(lambda x: x, transforms))


def get_imagenet_dataloaders(data_dir, cls_json=None, bs=8, val_bs=None,
                            grad_accum=1, transforms=None,
                            overfit_batches=None, overfit_len=10000):
    """Get some images
    """

    if transforms is None:
        transforms = [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[1.0, 1.0, 1.0])
        ]

    transforms = torchvision.transforms.Compose(transforms)

    train_ds = ImageNet(root=data_dir+"/train", class_json=cls_json, transform=transforms)
    val_ds = ImageNet(root=data_dir+"/val", class_json=cls_json, transform=transforms)

    # Not neccessary here, but make sure the class lists are in sync
    val_ds.classes = train_ds.classes
    val_ds.class_to_idx = train_ds.class_to_idx

    if not val_bs:
        val_bs = bs*2

    g_perm = torch.Generator()
    g_perm.manual_seed(69)

    val_dl = None
    if overfit_batches:
        print(f"Overfit on {overfit_batches} batches ({overfit_batches*grad_accum*bs} samples) instead of training")
        overfit_batches *= grad_accum # If we do grad accum,bs  is a fraction of what was asked

        indices = torch.randperm(len(train_ds), generator=g_perm)[:bs*overfit_batches]
        indices = indices.repeat(overfit_len//(bs*overfit_batches))
        sampler = IndexSampler(indices)
        train_dl = DataLoader(dataset=train_ds, batch_size=bs,
                        sampler=sampler, num_workers=6, drop_last=True)
    else:
        train_dl = DataLoader( dataset=train_ds,
                            batch_size=bs,
                            shuffle=True,
                            num_workers=min(os.cpu_count() or 1, 32),
                            drop_last=True,
                            pin_memory=True)

        val_dl = DataLoader(dataset=val_ds,
                            batch_size=val_bs,
                            shuffle=False,
                            num_workers=min(os.cpu_count() or 1, 32),
                            pin_memory=True)


    return train_dl, val_dl


def get_mnist_dataloaders(data_dir, bs=8, resize=224,
                        val_bs=None, overfit_batches=None, overfit_len=10000):
    """Get some numbers
    """

    tfm_list = []
    if resize is not None:
        tfm_list.append(torchvision.transforms.Resize((resize, resize)))
    tfm_list.append(torchvision.transforms.ToTensor())


    transforms = torchvision.transforms.Compose(
        tfm_list
        # torchvision.transforms.Normalize(mean=[0.4703, 0.4471, 0.4075], std=[1., 1., 1.]),
    )

    train_ds = torchvision.datasets.MNIST(root=data_dir,
                                            download=True,
                                            train=True,
                                            transform=transforms)
    val_ds = torchvision.datasets.MNIST(root=data_dir,
                                        train=False,
                                        transform=transforms)

    g_perm = torch.Generator()
    g_perm.manual_seed(69)

    if overfit_batches:
        raise NotImplementedError
        indices = torch.randperm(len(train_ds), generator=g_perm)[:bs*overfit_batches]
        indices = indices.repeat(overfit_len//(bs*overfit_batches))
        sampler = IndexSampler(indices)
    else:
        indices = torch.randperm(len(train_ds), generator=g_perm)
        sampler = RandomSampler(indices)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, sampler=sampler)
    val_dl = None
    if not overfit_batches:
        val_dl = DataLoader(dataset=val_ds, batch_size=val_bs, shuffle=False, num_workers=6)
    return train_dl, val_dl

# %%



# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Resize((224, 224)),
#     # torchvision.transforms.Normalize(mean=[0.4703, 0.4471, 0.4075], std=[1., 1., 1.]),
# ])

# train_ds = torchvision.datasets.MNIST(root="/home/xl0/work/ml/datasets/MNIST",
#                                         download=True,
#                                         train=True,
#                                         transform=transforms,
#                                         )


# g_perm = torch.Generator()
# g_perm.manual_seed(69)

# indices = torch.randperm(len(train_ds), generator=g_perm)[:10].repeat(5)

\

# sampler = IndexSampler(indices)

# print(indices)

# train_dl = DataLoader(dataset=train_ds, batch_size=5, sampler=sampler)

# for data in train_dl:
#     print(data[1])


# %%
