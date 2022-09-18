# %%
from collections import defaultdict
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageColor
from matplotlib import pyplot as plt
import torch


# %%
def is_notebook():
    """Check if we are running in a jupyter-notebook-like envoronemnt
    """
    try:
        from IPython import get_ipython # pylint: disable=import-outside-toplevel
        if get_ipython():
            return True
        return False
    except: # pylint: disable=bare-except
        return False

    return False

def metrics_names_pretty(metrics):
    """
    Return an str with the names of the metrics, extended to at least 8 characters
    """

    names = filter(lambda x: x[0] != "_", metrics.keys())

    pretty = [name + " " * (8 - len(name)) for name in names]
    return " ".join(pretty)

def metrics_last_pretty(metrics):
    """
    Return an str with the last values of the metrics,
    extended to match the length of the metric names, min 8 characters
    """
    out = ""
    for name, value in metrics.items():
        if name[0] != "_":
            if value:
                value = value[-1]
                if isinstance(value, int):
                    value = str(value)[:9]
                elif isinstance(value, float):
                    value = (f"{value:.6f}")[:9]
                else:
                    value = str(value)[:9]

                out += value + " " * (max(len(name), 8) - len(value) + 1)
            else:
                out += " " * (max(len(name), 8) + 1)

    return out


# def show_one_batch(X, y, idx2class, vis=None, vis_win="Batch"):
#     if not vis:
#         batch_pil = TF.to_pil_image(make_grid(X))

# %%

def annotate_image(x, y, pred=None, idx2class=None, font=None, font_size=11):
    y = int(y)
    if pred != None:
        pred = int(pred)

    if font:
        font = ImageFont.truetype()

    font = ImageFont.load_default()
    font_h = font.getbbox("A")[-1]

    red = ImageColor.getrgb("#c71f12")
    green = ImageColor.getrgb("#0e8c27")
    black = ImageColor.getrgb("black")
    white = ImageColor.getrgb("white")

    pad_h = (font_h + 2) * (2 if pred != None else 1)

    x = TF.to_pil_image(x)
    x_padded = Image.new(x.mode, (x.width, x.height + pad_h), color=white)
    x_padded.paste(x, (0, pad_h))

    draw = ImageDraw.Draw(x_padded)

    pred_text = ""
    if pred != None:
        pred_text = str(pred)
    text = str(y)

    if idx2class:
        if pred_text: pred_text += ":" + idx2class[pred]
        text += ":" + idx2class[y]

    draw.text(xy=(2, 2), text=text, fill=black)
    if pred != None:
        if pred == y:
            fill = green
        else:
            fill = red

        draw.text(xy=(2, pad_h // 2), text=pred_text, fill=fill)

    return TF.to_tensor(x_padded)


def annotate_batch(X, Y, preds=None, idx2class=None, font=None, font_size=11):

    # def _lambda((x, y, pred)):
    #     return annotate_image(x, y, pred, idx2class=idx2class)

    # tfm = torchvision.transforms.Lambda(_lambda)
    # return tfm()

    def _lambda(args):
        x, y, pred = args
        return annotate_image(x, y, pred,
                    idx2class=idx2class,
                    font=font,
                    font_size=font_size)

    if preds != None:
        return torch.stack([ _lambda((x, y, pred)) for x, y, pred in zip(X, Y, preds) ])
    else:
        return torch.stack([ _lambda((x, y, None)) for x, y in zip(X, Y) ])


if __name__ == "__main__":
    import datasets

    train, _ = datasets.get_imagenet_dataloaders("/home/xl0/work/ml/datasets/ImageNet/",
                    cls_json="../../imagenet-simple-labels/imagenet-simple-labels.json",
                    bs=32, overfit_batches=1, overfit_len=64)

    X, y = next(iter(train))
    # x = X[0]
    # x = TF.resize(x, [64, 64])
    # plt.imshow(annotate_image(x, y[0], 0, idx2class=train.dataset.classes))



    # tfm = torchvision.transforms.Lambda(_lambda)
    # res = tfm((X, y, y))


# %%
