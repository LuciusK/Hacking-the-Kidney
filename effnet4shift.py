
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
import torchvision
import cv2
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from scipy.ndimage.interpolation import zoom
import albumentations as A
from torch.nn import functional as F
from albumentations.pytorch import ToTensorV2
from lovasz import lovasz_hinge
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import gc
import torch_optimizer as optim

import os


import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
import sys
import time
import random
import argparse

parser = argparse.ArgumentParser()

# python effnet4.py --data 256 --numworker 1 --batchsize 32 --arch 'efficientnet-b4' --gpu '3' --fold 0 1 2 3 4 --fix

# critical
parser.add_argument('--data', type=int, default=256, help='Input size of the images')
parser.add_argument('--numworker', type=int, default=1, help='The worker of the model')
parser.add_argument('--batchsize', type=int, default=8, help='The batchsize of the model')
parser.add_argument('--arch', type=str, default='efficientnet-b4', help='The backbone of the model')
parser.add_argument('--gpu', type=str, default='0', help='GPU of the server')
parser.add_argument('--fold', nargs='+', type=int) # python effnet4.py --fold 0 1 2 3 4
parser.add_argument('--fix', action="store_false", help='fix the model pseudo number')
parser.add_argument('--shift', action="store_true", help='add the shift data')
parser.add_argument('--op', type=str, default='adam', help='optimizer of the model')

# most default
parser.add_argument('--epoch', type=int, default=20, help='The epoch of the train')
parser.add_argument('--seed', type=int, default=2021, help='The seed of the random function')
parser.add_argument('--lr', type=int, default=1e-4, help='The start Learning rate of the model')
parser.add_argument('--min_lr', type=int, default=1e-6, help='The min Learning rate of the model')
parser.add_argument('--verbose', type=int, default=4, help='The output of the model')
parser.add_argument('--freeze_epoch', type=int, default=1, help='The freeze epoch of the model')
parser.add_argument('--criterion', type=str, default='BCELoss', help='The loss of the model')
parser.add_argument('--pretrain_lr', type=int, default=1e-3, help='The pretrain learning rate of the freeze model')
parser.add_argument('--weight_decay', type=int, default=1e-6, help='The weight decay rate of the model')
parser.add_argument('--base_model', type=str, default='unet', help='The backbone type of the model')
parser.add_argument('--allfold', type=int, default=5, help='The fold of the model')

args = parser.parse_args()

class CFG:
    data = args.data # 512, 1024, 256
    num_workers = args.numworker
    scheduler = 'CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epoch = args.epoch # Change epochs
    freeze_epoch = args.freeze_epoch
    criterion = args.criterion # 'DiceBCELoss' # ['DiceLoss', 'Hausdorff', 'Lovasz']
    base_model = args.base_model # ['Unet']
    encoder = args.arch # "efficientnet-b4", "se_resnext50_32x4d", "efficientnet-b5", "efficientnet-b6"
    lr = args.lr
    pretrain_lr = args.pretrain_lr
    min_lr = args.min_lr
    batch_size = args.batchsize # 32, 8, 2
    weight_decay = args.weight_decay
    seed = args.seed
    n_fold = args.allfold
    train = True
    optimizer = 'Adam'
    T_0 = epoch - freeze_epoch
    T_mult = 1
    smoothing = 0.001

    verbose_step = args.verbose
    loss_update = 32 // batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.fix
    
seed_torch(seed=CFG.seed)



base_transform = A.Compose([
    A.OneOf([
        # 加了hsv
        A.HueSaturationValue(10,15,10),
        A.RandomBrightness(limit=.2, p=1), 
        A.RandomContrast(limit=.2, p=1), 
        A.RandomGamma(p=1)
    ], p=.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1)
    ], p=.25),
    A.OneOf([
        A.GaussNoise(0.002, p=.5),
        A.IAAAffine(p=.5),
    ], p=.25),
    # A.OneOf([
    #     A.ElasticTransform(alpha=120, sigma=120 * .05, alpha_affine=120 * .03, p=.5),
    #     A.GridDistortion(p=.5),
    #     A.OpticalDistortion(distort_limit=2, shift_limit=.5, p=1)                  
    # ], p=.25),
    A.RandomRotate90(p=.5),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5),
    A.Cutout(num_holes=10, 
                max_h_size=int(.1 * CFG.data), max_w_size=int(.1 * CFG.data), 
                p=.25),
    A.ShiftScaleRotate(p=.5)
])
# else:
#     aug = A.Compose([
#         A.OneOf([
#             A.RandomBrightness(limit=.2, p=1), 
#             A.RandomContrast(limit=.2, p=1), 
#             A.RandomGamma(p=1)
#         ], p=.5),
#         A.RandomRotate90(p=.25),
#         A.HorizontalFlip(p=.25),
#         A.VerticalFlip(p=.25)
#     ])

# base_transform = A.Compose([
#         A.HorizontalFlip(),
#         A.VerticalFlip(),
#         A.RandomRotate90(),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.9, 
#                          border_mode=cv2.BORDER_REFLECT),
#         A.OneOf([
#             A.OpticalDistortion(p=0.4),
#             A.GridDistortion(p=.1),
#             A.IAAPiecewiseAffine(p=0.4),
#         ], p=0.3),
#         A.OneOf([
#             A.HueSaturationValue(10,15,10),
#             A.CLAHE(clip_limit=3),
#             A.RandomBrightnessContrast(),            
#         ], p=0.5),
#     ], p=1.0)


val_transform = A.Compose([
    ], p=1.0)




mean = np.array([0.65459856,0.48386562,0.69428385])
std = np.array([0.15167958,0.23584107,0.13146145])

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim == 2: 
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self,df, train='train', augment='base', transform=True):
        ids = df.id.values
        self.train = train
        
        if self.train == 'train':
            if args.shift:
                self.fnames = [fname for fname in os.listdir(f'/home/hubmap-{CFG.data}x{CFG.data}/train') 
                                if fname in self.ids]
            else:
                self.fnames = [fname for fname in os.listdir(f'/home/hubmap-{CFG.data}x{CFG.data}/train') 
                                if fname in self.ids and fname.split('_')[-1] != 'train2.png']
        
        elif self.train == 'val':
            self.fnames = [fname for fname in os.listdir(f'/home/hubmap-{CFG.data}x{CFG.data}/train') 
                            if fname in self.ids and fname.split('_')[-1] != 'train2.png']
        
        self.augment = augment
        self.transform = transform
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(f'/home/hubmap-{CFG.data}x{CFG.data}/train', fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(f'/home/hubmap-{CFG.data}x{CFG.data}/masks', fname), cv2.IMREAD_GRAYSCALE)
        
        if self.train == 'train':
            if self.transform == True:
                augmented = base_transform(image=img,mask=mask)
                img,mask = augmented['image'], augmented['mask']
                    
        elif self.train == 'val':
            transformed = val_transform(image=img,mask=mask)
            img,mask = transformed['image'], transformed['mask']
            
        return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)








directory_list = os.listdir(f'/home/hubmap-{CFG.data}x{CFG.data}/train')

train_list = [fnames.split('_')[0] for fnames in directory_list if fnames.split('_')[-1] != 'test.png' and fnames.split('_')[-1] != 'ext.png']
dir_df = pd.DataFrame(train_list, columns=['id'])

test_list = [fnames.split('_')[0] for fnames in directory_list if fnames.split('_')[-1] == 'test.png' and fnames.split('_')[-1] != 'ext.png']
test_df  = pd.DataFrame(test_list, columns=['id'])

ext_list = [fnames.split('_')[0] for fnames in directory_list if fnames.split('_')[-1] != 'test.png' and fnames.split('_')[-1] == 'ext.png']
ext_df  = pd.DataFrame(ext_list, columns=['id'])



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return dice
    

class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, ratio = 0.8, smooth=CFG.smoothing):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.mean() + targets.mean() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE.mean()


class symmetric_lovasz(nn.Module):
    def __init__(self):
        super(symmetric_lovasz, self).__init__()

    def forward(self, outputs, targets):
        return 0.5 * (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs.float(), 1.0 - targets))







if CFG.criterion == 'db':
    criterion = DiceBCELoss()
elif CFG.criterion == 'DiceLoss':
    criterion = DiceLoss()
elif CFG.criterion == 'Lovasz':
    criterion = symmetric_lovasz()
elif CFG.criterion == 'BCELoss':
    criterion = nn.BCEWithLogitsLoss()




def Dice_soft(probability, mask):

    inter, union = 0, 0
    pred = torch.flatten(torch.sigmoid(probability))
    targ = torch.flatten(mask)

    inter += (pred * targ).float().sum().item()
    union += (pred + targ).float().sum().item()

    return 2.0 * inter / (union + 0.001)

def Dice_th(probability, mask, epoch):
    if epoch < 10:
        ths = torch.arange(0.1, 0.9, 0.05)
    else:
        ths = torch.arange(0.2, 0.6, 0.01)

    inter = torch.zeros(len(ths))
    union = torch.zeros(len(ths))

    pred = torch.flatten(torch.sigmoid(probability))
    targ = torch.flatten(mask)

    for i, th in enumerate(ths):
        p = (pred > th).float()
        inter[i] += (p * targ).float().sum().item()
        union[i] += (p + targ).float().sum().item()

    dices = torch.where(union > 0.0, 2.0 * inter / (union + 0.001), torch.zeros_like(union))
    return dices.max(), ths[torch.argmax(dices)]




def HuBMAPLoss(images, targets, model, device, loss_func=criterion):
    model.to(device)
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    loss_func = loss_func
    loss = loss_func(outputs, targets)
    return loss, outputs




def train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader, schd_batch_update=False):
    model.train()
    
    running_loss = None

    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for step, (images, targets) in pbar:

        with autocast():
            loss, outputs = HuBMAPLoss(images, targets, model, device)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            
            if ((step + 1) % CFG.loss_update == 0 or (step + 1) == len(trainloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if scheduler is not None and schd_batch_update:
                scheduler.step()
                
            # loss = loss.detach().item()
            # total_loss += loss
            
            if ((step + 1) % CFG.verbose_step == 0) or ((step + 1) == len(trainloader)):
                description = f'epoch {epoch} loss: {running_loss:.6f}'

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()
            
        
def valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader):
    model.eval()
    
    loss_sum = 0
    sample_num = 0
    val_probability, val_mask = [], []
    
    pbar = tqdm(enumerate(validloader), total=len(validloader))
    for step, (images, targets) in pbar:
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss = loss.detach().item()
        
        loss_sum += loss * targets.shape[0]
        sample_num += targets.shape[0]

        output_ny = outputs.data.cpu() # .numpy()
        target_np = targets.data.cpu() # .numpy()
            
        val_probability.append(output_ny)
        val_mask.append(target_np)

        if ((step + 1) % CFG.verbose_step == 0) or ((step + 1) == len(validloader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.6f}'
            pbar.set_description(description)
            
    val_probability = torch.cat(val_probability)
    val_mask = torch.cat(val_mask)
    best_dice, best_th = Dice_th(val_probability, val_mask, epoch)
    dice_soft = Dice_soft(val_probability, val_mask)

    print("Dice_soft: {:.4f}".format(dice_soft), end='    ')
    print("Best_dice: {:.4f}".format(best_dice.item()), end='    ')
    print("best_th: {:.2f}".format(best_th.item()))

    return best_dice, best_th




gkf = GroupKFold(CFG.n_fold)
dir_df['Folds'] = 0
for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
    dir_df.loc[val_idx, 'Folds'] = fold

test_df['Folds'] = -1
ext_df['Folds'] = -2

dir_df = pd.concat([dir_df, test_df, ext_df])




def prepare_train_valid_dataloader(df, fold):
    train_ids = df[~df.Folds.isin(fold)]
    val_ids = df[df.Folds.isin(fold)]
    
    train_ds = HuBMAPDataset(train_ids, train='train', augment='base', transform=True)
    val_ds = HuBMAPDataset(val_ids, train='val', augment='base', transform=True)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, pin_memory=True, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, pin_memory=True, shuffle=False, num_workers=CFG.num_workers)
    return train_loader, val_loader

fold_train = args.fold 


for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
    if fold in fold_train:
        print("Let's start training for {}-{}".format(CFG.encoder, CFG.base_model))
        
        print("fold:  {}    ----------------------------------------".format(fold))
        best = 0
        trainloader, validloader = prepare_train_valid_dataloader(dir_df, [fold])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path = f'/mnt/result1/{CFG.data}/{CFG.encoder}-{CFG.base_model}-{CFG.criterion}-{CFG.data}-FOLD-{fold}-model.pth'
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if CFG.base_model == 'unet':
            model = smp.Unet(CFG.encoder, encoder_weights=None, classes=1)
        elif CFG.base_model == 'linknet':
            model = smp.Linknet(CFG.encoder, encoder_weights='imagenet', classes=1)

        model.load_state_dict(state_dict)
        del state_dict

        scaler = GradScaler()


        for epoch in range(CFG.epoch):
            if epoch < CFG.freeze_epoch:
                for p in model.encoder.parameters():
                    p.requires_grad = False

                if args.op == 'adam':
                    optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.pretrain_lr, weight_decay=CFG.weight_decay)

                train_one_epoch(epoch, model, device, optimizer1, None, trainloader, schd_batch_update=False)
            else:
                if epoch == CFG.freeze_epoch:
                    del optimizer1
                for p in model.encoder.parameters():
                    p.requires_grad = True

                if args.op == 'adam': 
                    optimizer2 = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

                scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=CFG.T_0, T_mult=CFG.T_mult, 
                                                eta_min=CFG.min_lr, last_epoch=-1)

                train_one_epoch(epoch, model, device, optimizer2, scheduler2, trainloader, schd_batch_update=False)

            with torch.no_grad():
                best_dice, best_th = valid_one_epoch(epoch, model, device, None, None, validloader)
            if best_dice > best:
                best = best_dice
                torch.save(model.state_dict(), f'/mnt/result1/{CFG.data}/{CFG.encoder}-{CFG.base_model}-{CFG.criterion}-{CFG.data}-FOLD-{fold}-model.pth')
                print("updated to the best model in Epoch {}".format(epoch))
                print("-----------------------------------------------------------------------------------------")
        del model, optimizer2, trainloader, validloader, scaler, scheduler2
        torch.cuda.empty_cache()
        gc.collect()

