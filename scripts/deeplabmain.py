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

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300


from data.dataloader import GEN_DATA_LISTS, CWD26
from data.utils import collate

from core.deeplab_resnet import  DeepLabv3_plus as DeepLabv3R
from core.deeplab_xception import  DeepLabv3_plus as Deeplabv3X

from core.losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer_Deeplab, Evaluator_DeepLab, eval_wrapper, ModelUtils
import torch.nn.functional as F

from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='cityscape',
                            custom_pallet=np.asarray(config['pallet']).reshape(1,-1,3)/255)

print('Source Dataset')
src_data_lists = GEN_DATA_LISTS(config['src_data_dir'], config['sub_directories'])
src_train_paths, src_val_paths, _ = src_data_lists.get_splits()
classes = src_data_lists.get_classes()
src_data_lists.get_filecounts()
print('Target Dataset')
trg_data_lists = GEN_DATA_LISTS(config['trg_data_dir'], config['sub_directories'])
_, _, trg_test_paths = trg_data_lists.get_splits()
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

trg_test_data = CWD26(trg_test_paths[0], trg_test_paths[1], config['img_height'], config['img_width'],
                     False, config['Normalize_data'])

trg_test_loader = DataLoader(trg_test_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=3, persistent_workers=True)
#%
# DataLoader Sanity Checks
# batch = next(iter(train_loader))
# s=255
# img_ls = []
# [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
# [img_ls.append(g2c(batch['lbl'][i])) for i in range(config['batch_size'])]
# plt.title('Sample Batch')
# plt.imshow(imgviz.tile(img_ls, shape=(4,config['batch_size']//2), border=(255,0,0)))
# plt.axis('off')

#%%
model = DeepLabv3R(nInputChannels=3, n_classes=config['num_classes'], os=16, pretrained=True, _print=True)
# model = Deeplabv3X(nInputChannels=3, n_classes=config['num_classes'], os=16, pretrained=False, _print=True)
               
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # print(torch._dynamo.list_backends())
    model = torch.compile(model, mode="max-autotune")
loss = FocalLoss()
criterion = lambda x,y: loss(x, y)

optimizer = torch.optim.AdamW([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

# optimizer = torch.optim.Adam([{'params': model.parameters(),
#                                'lr':config['learning_rate']}],
#                                 weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric = ConfusionMatrix(config['num_classes'])


mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.get_model_profile(model, False)
# mu.load_chkpt(model, optimizer=None)
# mu.load_pretrained_chkpt(model, "/home/user01/data/talha/CWD26/pretrained/cityscape.pth")
trainer = Trainer_Deeplab(model, config['batch_size'], optimizer, criterion, metric)
# trainer = Trainer(model, config['batch_size'], optimizer, metric)
evaluator = Evaluator_DeepLab(model, metric)
trg_evaluator = Evaluator_DeepLab(model, metric)
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"val_mIOU": 0, "mIOU": 0,"trg_mIOU": 0, "loss": 10,
               "learning_rate": 0}, step=0)

#%%
start_epoch = 0
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou, trg_total_avg_viou = [], []
for epoch in range(start_epoch, config['epochs']):
    epoch 
    pbar = tqdm(train_loader)
    model.train() # <-set mode important
    ta, tl = [], []
    for step, data_batch in enumerate(pbar):

        scheduler(optimizer, step, epoch)
        loss_value = trainer.training_step(data_batch)
        iou = trainer.get_scores()
        trainer.reset_metric()
        
        tl.append(loss_value)
        ta.append(iou['iou_mean'])
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
    print(f'=> Average loss: {np.nanmean(tl):.4f}, Average IoU: {np.nanmean(ta):.4f}')
    g, n = data_batch['geo_augs'][0], data_batch['noise_augs'][0]

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
        wandb.log({"loss": loss_value, "mIOU": np.nanmean(ta),
                   "learning_rate": optimizer.param_groups[0]['lr'],
                   'geo_augs': g, 'noise_augs': n}, step=epoch+1)
    
    if curr_viou > best_iou:
        best_iou = curr_viou
        mu.save_chkpt(model, optimizer, epoch, loss_value, best_iou)

if config['LOG_WANDB']:
    wandb.run.finish()
#%%


