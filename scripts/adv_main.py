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

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
from itertools import cycle
mpl.rcParams['figure.dpi'] = 300


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
                       config['Augment_data'], config['Normalize_data'])

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
                     False, config['Normalize_data'])

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
if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    s=255
    if config['batch_size'] > 1:
        img_ls = []
        [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
        [img_ls.append(g2c(batch['lbl'][i])) for i in range(config['batch_size'])]
        plt.title('Sample Batch')
        plt.imshow(imgviz.tile(img_ls, shape=(4,config['batch_size']//2), border=(255,0,0)))
        plt.axis('off')
    else:
        plt.title('Sample Batch')
        plt.imshow(imgviz.tile([(batch['img'][0]*s).astype(np.uint8), g2c(batch['lbl'][0])]
                               , border=(255,0,0)))
#%%
model = MaxViT_OCR(num_classes=config['num_classes'], in_channnels=3, embed_dims=config['embed_dims'],
                ffn_ratios=config['ffn_ratios'], depths=config['depths'], num_stages=4,
                dec_outChannels=config['dec_channels'], ls_init_val=float(config['layer_scaling_val']), 
                drop_path=float(config['stochastic_drop_path']), drop_path_mode=config['SD_mode'],
                config=config)
disc = Discriminator(inChannel=config['num_classes']) 
aux_disc = Discriminator(inChannel=config['num_classes'])
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
disc = disc.to('cuda' if torch.cuda.is_available() else 'cpu')
aux_disc = aux_disc.to('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    model= nn.DataParallel(model)
    disc= nn.DataParallel(disc)
    aux_disc= nn.DataParallel(aux_disc)
    # # print(torch._dynamo.list_backends())
    model = torch.compile(model, mode="max-autotune")
    disc = torch.compile(disc, mode="max-autotune")
    aux_disc = torch.compile(aux_disc, mode="max-autotune")

#AdamW
optim = torch.optim.AdamW([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

disc_optim = torch.optim.Adam([{'params': disc.parameters(),
                               'lr':config['learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'],
                                betas=(0.9, 0.99))

aux_disc_optim = torch.optim.Adam([{'params': aux_disc.parameters(),
                               'lr':config['learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'],
                                betas=(0.9, 0.99))

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

d_scheduler = LR_Scheduler(config['lr_schedule_d'], config['learning_rate_d'], config['epochs'],
                           iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric = ConfusionMatrix(config['num_classes'])


mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.get_model_profile(model, False)
# mu.load_chkpt(model, optimizer=None)
# mu.load_pretrained_chkpt(model, "/home/user01/data/talha/CWD26/pretrained/cityscape.pth")

trainer = AdvTrain(model, disc, aux_disc, optim, disc_optim, aux_disc_optim, metric)
evaluator = Evaluator(model, metric)
trg_evaluator = Evaluator(model, metric)
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"val_mIOU": 0, "mIOU": 0, "trg_mIOU": 0, "disc_loss": 0,
               "loss": 10, "learning_rate_d": 0, "learning_rate_ad": 0,
               "learning_rate": 0}, step=0)

#%%
start_epoch = 0
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou, trg_total_avg_viou = [], []
for epoch in range(start_epoch, config['epochs']):
    epoch 
    pbar = tqdm(zip(train_loader, cycle(trg_train_loader)), total=len(train_loader))
    model.train() # <-set mode important
    ta, tl, dl = [], [], []
    for step, data_batch in enumerate(pbar):
        src_batch, trg_batch = data_batch
        scheduler(optim, step, epoch)
        d_scheduler(disc_optim, step, epoch)
        d_scheduler(aux_disc_optim, step, epoch)

        losses = trainer.training_step(src_batch, trg_batch)
        iou = trainer.get_scores()
        trainer.reset_metric()
        
        loss_value = losses['seg_loss']
        tl.append(loss_value)
        ta.append(iou['iou_mean'])
        dl.append(losses['disc_loss'])
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
    print(f'=> Average loss: {np.nanmean(tl):.4f}, Average IoU: {np.nanmean(ta):.4f}, Discriminator loss: {np.nanmean(dl):.4f}')
    g, n = src_batch['geo_augs'][0], src_batch['noise_augs'][0]

    if (epoch + 1) % 2 == 0: # eval every 2 epoch
        curr_viou, avg_viou, total_avg_viou, tiled = eval_wrapper(evaluator, model, val_loader, total_avg_viou)
        trg_curr_viou, trg_avg_viou, trg_total_avg_viou, trg_tiled = eval_wrapper(trg_evaluator, model, trg_test_loader, trg_total_avg_viou)

        cprint(f'=> Averaged srcValidation IoU: {avg_viou:.4f}', 'magenta')
        cprint(f'=> Averaged trgValidation IoU: {trg_avg_viou:.4f}', 'red')

        if config['LOG_WANDB']:
            wandb.log({"val_mIOU": avg_viou, "trg_mIOU": trg_avg_viou}, step=epoch+1)
            wandb.log({'src_predictions': wandb.Image(tiled), 
                       'trg_predictions': wandb.Image(trg_tiled)}, step=epoch+1)

    if config['LOG_WANDB']:
        wandb.log({"loss": np.nanmean(tl), "mIOU": np.nanmean(ta), "disc_loss": np.nanmean(dl),
                   "learning_rate": optim.param_groups[0]['lr'],
                   "learning_rate_d": disc_optim.param_groups[0]['lr'],
                   "learning_rate_ad": aux_disc_optim.param_groups[0]['lr'],
                   'geo_augs': g, 'noise_augs': n}, step=epoch+1)
    
    if curr_viou > best_iou:
        best_iou = curr_viou
        mu.save_chkpt(model, optim, epoch, loss_value, best_iou)

if config['LOG_WANDB']:
    wandb.run.finish()
#%%
