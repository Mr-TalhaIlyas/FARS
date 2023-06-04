#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/talha/CMED/scripts/")
from pickletools import optimize
import yaml
with open('infer_config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
               config_include_keys=config.keys(), config=config)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300

from data.dataloader import GEN_DATA_LISTS, CWD26, inference_loader, write_eval_txt_files
from data.utils import collate, images_transform, torch_resizer, masks_transform

from core.model import UHD_OCR, MaxViT_OCR
from core.deeplab_resnet import  DeepLabv3_plus as DeepLabv3R
from core.deeplab_xception import  DeepLabv3_plus as Deeplabv3X
from core.psp import PSPNet

from core.discriminator import Discriminator
from core.losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import AdvTrain, Evaluator, ModelUtils, eval_wrapper
import torch.nn.functional as F
from fmutils import fmutils as fmu
from empatches import EMPatches
from data.utils import std_norm
from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='cityscape',
                            custom_pallet=np.asarray(config['pallet']).reshape(1,-1,3)/255)
emp = EMPatches()

def get_data_loaders(data_dir):
    data_lists = GEN_DATA_LISTS(data_dir, config['sub_directories'])
    _, test_paths, _ = data_lists.get_splits()
    classes = data_lists.get_classes()
    data_lists.get_filecounts()

    test_data = CWD26(test_paths[0], test_paths[1], config['img_height'], config['img_width'],
                        False, config['Normalize_data'])

    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate)
    return test_loader

def get_data_loaders2(data_dir):

    data_lists = GEN_DATA_LISTS(data_dir, config['sub_directories'])
    _, _, test_paths = data_lists.get_splits()
    img_paths, lbl_paths = test_paths
    return img_paths, lbl_paths

test_loader = get_data_loaders(config['trg_data_dir'])
test_paths = get_data_loaders2(config['trg_data_dir'])

if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(test_loader))
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

if config['use_model'] == 'proposed':
    model = MaxViT_OCR(num_classes=config['num_classes'], in_channnels=3, embed_dims=config['embed_dims'],
                    ffn_ratios=config['ffn_ratios'], depths=config['depths'], num_stages=4,
                    dec_outChannels=config['dec_channels'], ls_init_val=float(config['layer_scaling_val']), 
                    drop_path=float(config['stochastic_drop_path']), drop_path_mode=config['SD_mode'],
                    config=config)
elif config['use_model'] == 'PSP':
    model = PSPNet(n_classes=config['num_classes'], block_config=[3, 4, 23, 3], img_size=config['img_height'], img_size_8=64)
elif config['use_model'] == 'DeepLab':
    model = Deeplabv3X(nInputChannels=3, n_classes=config['num_classes'], os=16, pretrained=False, _print=True)
    # model = DeepLabv3R(nInputChannels=3, n_classes=config['num_classes'], os=16, pretrained=True, _print=True)
              
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
metric = ConfusionMatrix(config['num_classes'])
mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])

# mu.load_chkpt(model)
mu.load_pretrained_chkpt(model, f'/home/user01/data/talha/CMED/chkpts/{config["experiment_name"]}.pth')
#%%
###########################
#  EVALUATION
###########################

metric = ConfusionMatrix(config['num_classes'])
class_dict = {'ring': 1, 'trophozoite':2, 'schizont':3, 'gametocyte':4}

magnifications = ['400x/']#['100x/', '400x/', '1000x/']
# data_dir = "/home/user01/data/talha/bean_uda/datasets/src/"

for meg in magnifications:
    data_dir = f"/home/user01/data/talha/CMED/dataset/HCM/{meg}/"

    img_paths, lbl_paths = get_data_loaders2(data_dir)
    
    crop, weed, avg = [], [], []

    model.eval()
    for img_path, lbl_path in tqdm(zip(img_paths, lbl_paths), total=len(img_paths)):

        img, lbl, orig_h, orig_w, filename = inference_loader(img_path, lbl_path)
        img_batch = images_transform([img])
        lbl_batch = torch_resizer(masks_transform([lbl]))

        with torch.no_grad():
            if config['use_model'] == 'proposed':
                _, preds = model.forward(img_batch) # UHD_OCR
            elif config['use_model'] == 'PSP':
                preds, _ = model.forward(img_batch) # PSP
            elif config['use_model'] == 'DeepLab':
                preds = model.forward(img_batch) # Deep Lab
                
        if config['use_model'] == 'proposed':
            preds = preds['out'] # UHD_OCR

        # write_boxes_and_confidences_to_file(preds.clone(), filename, class_dict)
        write_eval_txt_files(preds.clone(), class_dict, filename, orig_h, orig_w)

        preds = preds.argmax(1)
        preds = preds.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()
        metric.update(lbl_batch, preds)
        iou = metric.get_scores()
        metric.reset()
        # crop.append(iou['iou'][1])
        # weed.append(iou['iou'][2])
        avg.append(iou['iou_mean'])

    # crop = np.array(crop)
    # weed = np.array(weed)
    avg = np.array(avg)

    # crop[crop==0.0] = np.nan
    # weed[weed==0.0] = np.nan
    avg[avg==0.0] = np.nan

    print(f'{set} @ {config["experiment_name"]}')
    print(f'Avg: {np.nanmean(avg)}')
    
#%%





























































# # %%
# files = fmu.get_all_files('/home/user01/data/talha/bean_uda/datasets/preds/imgs/')

# for i in range(len(files)):
#     img = cv2.imread(files[i])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (config['img_width'], config['img_height']), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
#     img = std_norm(img)

#     name = fmu.get_basename(files[i], False)
#     model.eval()
#     img_batch = images_transform([img])

#     with torch.no_grad():
#         # _, preds = model.forward(img_batch) # UHD_OCR
#         # preds, _ = model.forward(img_batch) # PSP
#         preds = model.forward(img_batch) # Deep Lab

#     # preds = preds['out'] # UHD_OCR
#     preds = preds.argmax(1)
#     preds = preds.cpu().numpy().squeeze()
#     preds = g2c(preds)

#     preds = cv2.cvtColor(preds, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f'/home/user01/data/talha/bean_uda/datasets/preds/adseg/{name}.png', preds)
# # %%
# batch = next(iter(test_loader))
# img_batch = images_transform(batch['img'])
# lbl_batch = batch['lbl'][0]#torch_resizer(masks_transform(batch['lbl']))

# model.eval()
# with torch.no_grad():
#     _, preds = model.forward(img_batch) 
    
# preds = preds['out'].argmax(1)
# preds = preds.cpu().numpy().squeeze()
# preds = cv2.resize(preds, (config['img_width'], config['img_height']), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
# plt.imshow(g2c(preds))
# %%
