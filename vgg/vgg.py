# %%

import functools
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%

class VGG(nn.Module):
    """A generic version of VGG

    Args:
        chans: Array of number channels per conv group.
            In the original paper it's [64, 128, 256, 512, 512]

        convs: Array of number of convolutions per group.
            For example, VGG-11: [1, 1, 2, 2, 2]

        in_chans=3: Number of input channels.
        n_classes=1000: Number of classes to predict

    Returns:
        A newly constructed net with default random initialisations.
    """
    def __init__(self, chans, convs, in_chans=3, n_classes=1000):
        if not chans or not convs or len(chans) != len(convs):
            raise ValueError(f"Invalid network configuration: chans={chans}, convs={convs}")
        super().__init__()

        layers = []
        prev_channels = in_chans

        for chan, conv in zip(chans, convs):
            for i in range(conv):
                layers.append(nn.Conv2d(prev_channels, chan, 3, padding="same"))
                layers.append(nn.ReLU())
                prev_channels = chan
            layers.append(nn.MaxPool2d(2))

        self.extractor = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=n_classes),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.head(x)


class VGG3(VGG):
    def __init__(self, in_chans=3, n_classes=1000):
        chans = [64, 128,]
        convs = [1, 2]
        super().__init__(chans=chans, convs=convs,
                            in_chans=in_chans, n_classes=n_classes)

class VGG11(VGG):
    def __init__(self, in_chans=3, n_classes=1000):
        chans = [64, 128, 256, 512, 512]
        convs = [1, 1, 2, 2, 2]
        super().__init__(chans=chans, convs=convs,
                            in_chans=in_chans, n_classes=n_classes)

class VGG13(VGG):
    def __init__(self, in_chans=3, n_classes=1000):
        chans = [64, 128, 256, 512, 512]
        convs = [2, 2, 2, 2, 2]
        super().__init__(chans=chans, convs=convs,
                            in_chans=in_chans, n_classes=n_classes)

class VGG16(VGG):
    def __init__(self, in_chans=3, n_classes=1000):
        chans = [64, 128, 256, 512, 512]
        convs = [2, 2, 3, 3, 3]
        super().__init__(chans=chans, convs=convs,
                            in_chans=in_chans, n_classes=n_classes)

class VGG19(VGG):
    def __init__(self, in_chans=3, n_classes=1000):
        chans = [64, 128, 256, 512, 512]
        convs = [2, 2, 4, 4, 4]
        super().__init__(chans=chans, convs=convs,
                            in_chans=in_chans, n_classes=n_classes)


# %%
