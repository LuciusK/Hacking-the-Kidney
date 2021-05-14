import sys

sys.path.append("../input/segmentation-models-pytorch-install")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import gc
from tqdm.notebook import tqdm
import rasterio
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from torch.nn import functional as F
from rasterio.windows import Window
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

sz = 256  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.3  # threshold for positive predictions
DATA = '../input/hubmap-kidney-segmentation/test/'
MODELS = [f'../input/b4-512/efficientnet-b4-512-FOLD-{i}-model.pth' for i in range(5)]
df_sample = pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv')
bs = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelname = "efficientnet-b4"
minoverlap = 300


# functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start: start + length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


# https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
# with transposed mask
def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


# https://www.kaggle.com/iafoss/256x256-images
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape  # 25784*34937
        self.reduce = reduce  # 4
        self.sz = reduce * sz  # 4*256 = 1024
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        if self.data.count == 3:
            img = np.moveaxis(self.data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2))), 0, -1)
        else:
            img = np.zeros((self.sz, self.sz, 3), np.uint8)
            for i, layer in enumerate(self.layers):
                img[:, :, i] = layer.read(1, window=Window.from_slices((x1, x2), (y1, y2)))

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz // reduce, self.sz // reduce),
                             interpolation=cv2.INTER_AREA)
        # check for empty images
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        vertices = [x1, x2, y1, y2]
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        count = 0
        # x: images, z: vertices, y: idx
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty images
                    x = x[y >= 0].to(device)
                    z = z[y >= 0]
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        # x, y, xy flips as TTA
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= (1 + len(flips))
                    py /= len(self.models)

                    py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                    # 1 class
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    py = py.squeeze(-1).numpy()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], z[i], y[i]
                        count += 1

    def __len__(self):
        return len(self.dl.dataset)


models = []
for path in MODELS:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = smp.Unet(modelname, encoder_weights=None, classes=1)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models.append(model)

del state_dict

names, preds = [], []

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row['id']
    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)
    # generate masks
    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, i in iter(mp):
        x1, x2, y1, y2 = vert
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)

    rle = rle_encode_less_memory(mask)
    names.append(idx)
    preds.append(rle)
    del mask, ds, dl
    gc.collect()

df = pd.DataFrame({'id': names, 'predicted': preds})
df.to_csv('submission.csv', index=False)
display(df)
