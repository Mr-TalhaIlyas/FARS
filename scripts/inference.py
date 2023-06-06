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
from pathlib import Path
from tabulate import tabulate

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300

from data.dataloader import GEN_DATA_LISTS, CWD26
from data.utils import collate, images_transform, torch_resizer, masks_transform
from infer_utils import (inference_loader, Segmentation2Bbox, get_data_loaders,
                         get_data_paths, make_onehot)

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
import eval.map.evaluators.coco_evaluator as coco_evaluator
import eval.map.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import eval.map.utils.converter as converter
from eval.map.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)
from eval.mpq.run import get_pq

g2c = lambda x : gray2color(x, use_pallet='cityscape',
                            custom_pallet=np.asarray(config['pallet']).reshape(1,-1,3)/255)
emp = EMPatches()

class_dict = {'ring': 1, 'trophozoite':2, 'schizont':3, 'gametocyte':4}
slides = 'blood-smear-slides'

meg = config['meg']

data_dir = Path(config['data_dir'], config['trg_machine'], meg)

seg2bbox = Segmentation2Bbox(class_dict, config['predictions_dir'], config['trg_machine'], meg)

img_paths, lbl_paths = get_data_paths(data_dir)
test_loader = get_data_loaders(data_dir)

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
mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
# mu.load_chkpt(model)
mu.load_pretrained_chkpt(model, f'/home/user01/data/talha/CMED/chkpts/{config["experiment_name"]}.pth')
#%%
###########################
#  EVALUATION on Target
###########################

metric = ConfusionMatrix(config['num_classes'])

# magnifications = ['400x/']#['100x/', '400x/', '1000x/']
# for meg in magnifications:
avg = []
op = np.zeros((len(img_paths), 128, 128, 5))
orig = np.zeros((len(img_paths), 128, 128, 5))
types = []

model.eval()

i = 0
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

    seg2bbox.write_eval_txt_files(preds.clone(), filename, orig_h, orig_w)

    preds = preds.argmax(1)
    preds = preds.cpu().numpy()
    lbl_batch = lbl_batch.cpu().numpy()

    metric.update(lbl_batch, preds)
    iou = metric.get_scores()
    metric.reset()
    avg.append(iou['iou_mean'])
    # PQ array writing  
    op[i, ...] = make_onehot(preds.squeeze().astype(np.uint8), config['num_classes'])
    orig[i, ...] = make_onehot(lbl_batch.squeeze().astype(np.uint8), config['num_classes'])
    types.append(slides)
    i += 1
    
avg = np.array(avg)
avg[avg==0.0] = np.nan

###########################
#  CALCULATE mPQ/bPQ
###########################

print('Saving Output Arrays')
os.makedirs(Path(config['predictions_dir'],'PQ'), exist_ok=True)
np.save(Path(config['predictions_dir'],'PQ', 'preds.npy'), op) # cnns predictions
np.save(Path(config['predictions_dir'],'PQ', 'gts.npy'), orig) # ground truths
np.save(Path(config['predictions_dir'],'PQ','types.npy'), np.asarray(types)) # tumor tissue types
print('Done Saving Arrays')

get_pq(Path(config['predictions_dir'],'PQ'), config['iou_threshold'])

###########################
#  CALCULATE mAP
###########################

# DEFINE GROUNDTRUTHS AND DETECTIONS
dir_imgs = Path(config['data_dir'], config['trg_machine'], meg, config['split'], 'images')
dir_gts = Path(config['data_dir'], config['trg_machine'], meg, config['split'], 'xmls')
if config['orig_annotations']:
    dir_gts = Path(config['data_dir'], 'Annotations', config['trg_machine'], config['split'], meg)
dir_dets = Path(config['predictions_dir'], config['trg_machine'], meg)

# Get annotations (ground truth and detections)
gt_bbs = converter.vocpascal2bb(dir_gts)
det_bbs = converter.text2bb(dir_dets, bb_type=BBType.DETECTED, bb_format=BBFormat.XYX2Y2,
                            type_coordinates=CoordinatesType.ABSOLUTE, img_dir=dir_imgs)

# EVALUATE WITH COCO METRICS
coco_res1 = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
# coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)

# EVALUATE WITH VOC PASCAL METRICS
iou = config['iou_threshold']
dict_res = pascal_voc_evaluator.get_pascalvoc_metrics(
    gt_bbs, det_bbs, iou, generate_table=True, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)


print(30*'/')
print(f'Avg IoU: {np.nanmean(avg)}\n')
print(30*'/')
print(f'Pascal VOC mAP = {dict_res["mAP"]:0.5f}\n')
print(30*'/')
print('COCO Metrics')
print(tabulate(coco_res1.items(), headers=["Key", "Value"], tablefmt="github", floatfmt=".5f"))

print(30*'/')
print(f'{dir_imgs}\n{dir_gts}\n{dir_dets}\n')
#%%

