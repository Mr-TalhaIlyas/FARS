#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/talha/CMED/scripts/")

import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# torch.set_float32_matmul_precision('high')
from data.utils import (images_transform, masks_transform, torch_imgresizer,
                        torch_resizer, std_norm)
from fda.fda_torch import DomainAdapter, get_full_fft_amp, get_full_fft_pha

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
from itertools import cycle
# mpl.rcParams['figure.dpi'] = 300


from data.dataloader import GEN_DATA_LISTS, CWD26
from data.utils import collate
from core.model import UHD_OCR, MaxViT_OCR
from core.discriminator import Discriminator
from core.losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import AdvTrain, Evaluator, ModelUtils, eval_wrapper
import torch.nn.functional as F

from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='pannuke', custom_pallet=None)

print('Source Dataset')
src_data_lists = GEN_DATA_LISTS(config['src_data_dir'], config['sub_directories'])
src_train_paths, src_val_paths, _ = src_data_lists.get_splits()
classes = src_data_lists.get_classes()
src_data_lists.get_filecounts()
print('Target Dataset')
trg_data_lists = GEN_DATA_LISTS(config['trg_data_dir'], config['sub_directories'])
trg_train_paths, _, trg_test_paths = trg_data_lists.get_splits()
trg_data_lists.get_filecounts()

train_data = CWD26(src_train_paths[0], src_train_paths[1], config['img_height'], config['img_width'],
                       True, config['Normalize_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'], drop_last=True, # important for adaptive augmentation to work properly.
                          collate_fn=collate, pin_memory=config['pin_memory'],
                          prefetch_factor=3, persistent_workers=True)

val_data = CWD26(src_val_paths[0], src_val_paths[1], config['img_height'], config['img_width'],
                     False, config['Normalize_data'])

val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)

trg_train_data = CWD26(trg_train_paths[0], trg_train_paths[1], config['img_height'], config['img_width'],
                       True, config['Normalize_data'])

trg_train_loader = DataLoader(trg_train_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)

trg_test_data = CWD26(trg_test_paths[0], trg_test_paths[1], config['img_height'], config['img_width'],
                     False, config['Normalize_data'])

trg_test_loader = DataLoader(trg_test_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)
#%%
'''
Change the fda torch script to output the phase and amplitude dict
'''


pbar = tqdm(zip(train_loader, cycle(trg_train_loader)), total=len(train_loader))

ta, tl, dl = [], [], []
for step, data_batch in enumerate(pbar):
    src_batch, trg_batch = data_batch
    break
#%
simg_batch = images_transform(src_batch['img'])
timg_batch = images_transform(trg_batch['img'])
lbl_batch = torch_resizer(masks_transform(src_batch['lbl']))

# we will transfer HCM stains texture to LCM stains.
da = DomainAdapter(L=0.01, space='ycrcb')
op_batch, ddd = da.apply_fda(simg_batch.clone(), timg_batch.clone())

x = op_batch.permute(0, 2, 3, 1).cpu().numpy()
plt.figure()
plt.imshow(x[-1,...])

da = DomainAdapter(L=0.01, space='none')
op_batch, ddd = da.apply_fda(simg_batch.clone(), timg_batch.clone())

x = op_batch.permute(0, 2, 3, 1).cpu().numpy()
plt.figure()
plt.imshow(x[-1,...])


plt.figure()
plt.imshow(simg_batch.permute(0, 2, 3, 1).cpu().numpy()[-1,...])

plt.figure()
plt.imshow(timg_batch.permute(0, 2, 3, 1).cpu().numpy()[-1,...])
#%%
amp_src = ddd['amp_src'].permute(0, 2, 3, 1).cpu().numpy().squeeze()[...,0]
pha_src = ddd['pha_src'].permute(0, 2, 3, 1).cpu().numpy().squeeze()[...,0]
amp_trg = ddd['amp_trg'].permute(0, 2, 3, 1).cpu().numpy().squeeze()[...,0]
pha_trg = ddd['pha_trg'].permute(0, 2, 3, 1).cpu().numpy().squeeze()[...,0]

full_amp_src = get_full_fft_amp(amp_src)
full_pha_src = get_full_fft_pha(pha_src)

full_amp_trg = get_full_fft_amp(amp_trg)
full_pha_trg = get_full_fft_pha(pha_trg)

plt.figure()
# plt.imshow(imgviz.tile([full_amp_src,full_pha_src,full_amp_trg,full_pha_trg], shape=(2,2), border=(255,0,0)))
plt.imshow(full_amp_src, cmap='gray_r')
plt.figure()
plt.imshow(full_pha_src, cmap='gray_r')
plt.figure()
plt.imshow(full_amp_trg, cmap='gray_r')
plt.figure()
plt.imshow(full_pha_trg, cmap='gray_r')
# %%

