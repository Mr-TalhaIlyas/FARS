import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import math

import cv2, os, imgviz, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses import Entropy
from data.utils import (images_transform, masks_transform, torch_imgresizer,
                        torch_resizer)
from fda.fda_torch import DomainAdapter
from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='pannuke')

class ModelUtils(object):
    def __init__(self, num_classes, chkpt_pth, exp_name):
        self.num_classes = num_classes
        self.chkpt_pth = chkpt_pth
        self.exp_name = exp_name
    
    def save_chkpt(self, model, optimizer, epoch=0, loss=0, iou=0):
        cprint('-> Saving checkpoint', 'green')
        torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'iou': iou,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(self.chkpt_pth, f'{self.exp_name}.pth'))

    def get_model_profile(self, model, summary=False):
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model total params: {total_params/10**6}')
        print(f'Model trainable params: {trainable_params/10**6}')
        if summary:
            from torchinfo import summary
            summary(model, input_size=(config['batch_size'],config['input_channels'], config['img_width'], config['img_height']), depth=2)

    def load_chkpt(self, model, optimizer=None):
        
        try:
            print('-> Loading checkpoint')
            chkpt = torch.load(os.path.join(self.chkpt_pth, f'{self.exp_name}.pth'),
                                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
            epoch = chkpt['epoch']
            loss = chkpt['loss']
            iou = chkpt['iou']
            model.load_state_dict(chkpt['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            print(f'[INFO] Loaded Model checkpoint: epoch={epoch} loss={loss} iou={iou}')
        except FileNotFoundError:
            print('[INFO] No checkpoint found')
        except RuntimeError:
            print('[Error] Pretrained dict dont match')
    
    def load_pretrained_chkpt(self, model, pretrained_path=None):
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path,
                               map_location='cuda' if torch.cuda.is_available() else 'cpu')
            try:
                # load pretrained
                pretrained_dict = chkpt['model_state_dict']
                # load model state dict
                state = model.state_dict()
                # loop over both dicts and make a new dict where name and the shape of new state match
                # with the pretrained state dict.
                matched, unmatched = [], []
                new_dict = {}
                for i, j in zip(pretrained_dict.items(), state.items()):
                    pk, pv = i # pretrained state dictionary
                    nk, nv = j # new state dictionary
                    # if name and weight shape are same
                    if pk.strip('_orig_mod.module.') == nk.strip('_orig_mod.module.') and pv.shape == nv.shape:
                        new_dict[nk] = pv
                        matched.append(pk)
                    else:
                        unmatched.append(pk)

                state.update(new_dict)
                model.load_state_dict(state)
                print('Pre-trained state loaded successfully...')
                print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
            except:
                print(f'ERROR in pretrained_dict @ {pretrained_path}')
        else:
            print('Enter pretrained_dict path.')

class Trainer(object):
    def __init__(self, model, batch, optimizer, metric):
        self.model = model
        self.batch = batch
        self.optimizer = optimizer
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def training_step(self, batched_data):
        img_batch = images_transform(batched_data['img'])
        lbl_batch = torch_resizer(masks_transform(batched_data['lbl']))
        
        # self.optimizer.zero_grad()
        self.model.zero_grad()

        loss, preds = self.model.forward(img_batch, target=lbl_batch)

        if config['use_ocr']: # because only OCR has aux_loss.
            loss = loss['loss'] + config['AUX_LOSS_Weights'] * loss['aux_loss']

        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            loss = loss.mean()
        
        loss.backward()
        self.optimizer.step()

        preds = preds['out'].argmax(1)
        preds = preds.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, preds)

        return loss.item()

class Evaluator(object):
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def eval_step(self, data_batch):
        self.img_batch = images_transform(data_batch['img'])
        lbl_batch = torch_resizer(masks_transform(data_batch['lbl']))
        
        with torch.no_grad():
            _, preds = self.model.forward(self.img_batch) 

        preds = preds['out']
        preds = preds.argmax(1)
        self.preds = preds.cpu().numpy()
        self.lbl_batch = lbl_batch.cpu().numpy()
        self.metric.update(self.lbl_batch, self.preds)
        
    def get_sample_prediction(self):
        # get single image, lbl, pred for plotting
        self.img_batch = torch_imgresizer(self.img_batch).detach().cpu().numpy()
        
        imgs, lbls, preds = [], [], []
        for i in range(3): # show 3 images
            
            img = np.transpose(self.img_batch[i,...], (1,2,0))
            lbl = self.lbl_batch[i,...]
            pred = self.preds[i,...]
            
            imgs.append((img*255).astype(np.uint8))
            lbls.append(g2c(lbl.astype(np.uint8)))
            preds.append(g2c(pred.astype(np.uint8)))
        
        return imgs + lbls + preds

def eval_wrapper(evaluator, model, val_loader, total_avg_viou):

    model.eval() # <-set mode important
    va = []
    vbar = tqdm(val_loader)
    for step, val_batch in enumerate(vbar):
        with torch.no_grad():
            evaluator.eval_step(val_batch)
            viou = evaluator.get_scores()
            evaluator.reset_metric()

        va.append(viou['iou_mean'])
        vbar.set_description(f'Validation - v_mIOU {viou["iou_mean"]:.4f}')

    img_gt_pred = evaluator.get_sample_prediction()
    tiled = imgviz.tile(img_gt_pred, shape=(3,3), border=(255,0,0))
    tiled = cv2.resize(tiled, (224,224))# just for visulaization
    # plt.imshow(tiled)
    avg_viou = np.nanmean(va)
    total_avg_viou.append(avg_viou)
    curr_viou = np.nanmax(total_avg_viou)

    return curr_viou, avg_viou, total_avg_viou, tiled

class AdvTrain(object):
    def __init__(self, model, disc, aux_disc, model_optim, disc_optim, aux_disc_optim, metric):
        self.model = model
        self.disc = disc
        self.aux_disc = aux_disc
        self.model_optim = model_optim
        self.disc_optim = disc_optim
        self.aux_disc_optim = aux_disc_optim
        self.metric = metric
        self.src_lbl = 0
        self.trg_lbl = 1
        self.prob2ent = Entropy(config['ita'], config['charbonnier'], config['reduce_dim'])
        self.da = DomainAdapter(L=config['L'], space=config['space'])
        # self.prob2ent = nn.Softmax(dim=1)


    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def defreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def training_step(self, src_batch, trg_batch):
        simg_batch = images_transform(src_batch['img'])
        timg_batch = images_transform(trg_batch['img'])
        lbl_batch = torch_resizer(masks_transform(src_batch['lbl']))
        
        # we will transfer HCM stains texture to LCM stains.
        # timg_batch = self.da.apply_fda(timg_batch, simg_batch)
        # simg_batch = self.da.apply_fda(simg_batch, timg_batch)
        if np.random.randint(0, 100) % 2 == 0: # randomly when even
            simg_batch = self.da.apply_fda(simg_batch, timg_batch)
        else: # when odd
            timg_batch = self.da.apply_fda(timg_batch, simg_batch)
        
        self.model.zero_grad()
        self.disc.zero_grad()
        self.aux_disc.zero_grad()

        #*****************
        # Model Training
        #*****************
        self.defreeze(self.model)
        self.freeze(self.disc)
        self.freeze(self.aux_disc)

        src_loss, src_preds = self.model.forward(simg_batch, target=lbl_batch) 

        loss = src_loss['loss'] + config['AUX_LOSS_Weights'] * src_loss['aux_loss']
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            loss = loss.mean()
        loss.backward()

        # lbl_batch = torch_resizer(masks_transform(trg_batch['lbl']))
        _, trg_preds = self.model.forward(timg_batch, target=None)
        # probability -> entropy (Ix)
        ent = self.prob2ent(trg_preds['out'])
        aux_ent = self.prob2ent(trg_preds['aux_out'])

        adv_loss = self.disc(ent, self.src_lbl)
        adv_aux_loss = self.aux_disc(aux_ent, self.src_lbl)

        adv_loss = config['LAMBDA_ADV_MAIN'] * adv_loss + config['LAMBDA_ADV_AUX'] * adv_aux_loss
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            adv_loss = adv_loss.mean()
        adv_loss.backward()
        
        #************************
        # Discriminatior Training
        #************************
        self.freeze(self.model)
        self.defreeze(self.disc)
        self.defreeze(self.aux_disc)

        # with source data
        disc_loss = self.disc(self.prob2ent(src_preds['out'].detach()),
                              target = self.src_lbl) / 4 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            disc_loss = disc_loss.mean()
        disc_loss.backward()

        aux_disc_loss = self.aux_disc(self.prob2ent(src_preds['aux_out'].detach()),
                                      target = self.src_lbl) / 4 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            aux_disc_loss = aux_disc_loss.mean()
        aux_disc_loss.backward()

        # with target data
        disc_loss = self.disc(self.prob2ent(trg_preds['out'].detach()),
                              target = self.trg_lbl) / 4 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            disc_loss = disc_loss.mean()
        disc_loss.backward()
        aux_disc_loss = self.aux_disc(self.prob2ent(trg_preds['aux_out'].detach()),
                                      target = self.trg_lbl) / 4 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            aux_disc_loss = aux_disc_loss.mean()
        aux_disc_loss.backward()

        # finish training step
        self.model_optim.step()
        self.disc_optim.step()
        self.aux_disc_optim.step()

        src_pred = src_preds['out'].argmax(1)
        src_pred = src_pred.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, src_pred)

        cur_losses = {'seg_loss': src_loss['loss'].mean().item(),
                      'aux_seg_loss': src_loss['aux_loss'].mean().item(),
                      'adv_loss': adv_loss.item(),
                      'disc_loss': disc_loss.item(),
                      'aux_disc_loss': aux_disc_loss.item()}

        return cur_losses

#%%
class AdvTrain_Deeplab(object):
    def __init__(self, model, disc, model_optim, disc_optim, criterion, metric):
        self.model = model
        self.disc = disc
        self.model_optim = model_optim
        self.disc_optim = disc_optim
        self.criterion = criterion
        self.metric = metric
        self.src_lbl = 0
        self.trg_lbl = 1
        self.prob2ent = Entropy(config['ita'], config['charbonnier'], config['reduce_dim'])
        
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def defreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def training_step(self, src_batch, trg_batch):
        img_batch = images_transform(src_batch['img'])
        lbl_batch = torch_resizer(masks_transform(src_batch['lbl']))

        self.model.zero_grad()
        self.disc.zero_grad()

        #*****************
        # Model Training
        #*****************
        self.defreeze(self.model)
        self.freeze(self.disc)

        preds = self.model.forward(img_batch) 
        loss = self.criterion(preds, lbl_batch)
        loss.backward()

        img_batch = images_transform(trg_batch['img'])
        # lbl_batch = torch_resizer(masks_transform(trg_batch['lbl']))
        trg_preds = self.model.forward(img_batch)
        # probability -> entropy (Ix)
        ent = self.prob2ent(trg_preds)

        adv_loss = self.disc(ent, self.src_lbl)

        adv_loss = config['LAMBDA_ADV_MAIN'] * adv_loss
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            adv_loss = adv_loss.mean()
        adv_loss.backward()
        
        # finish training step
        self.model_optim.step()
        #************************
        # Discriminatior Training
        #************************
        self.freeze(self.model)
        self.defreeze(self.disc)

        # with source data
        disc_loss = self.disc(self.prob2ent(preds.detach()),
                              target = self.src_lbl) / 2 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            disc_loss = disc_loss.mean()
        disc_loss.backward()

        # with target data
        disc_loss = self.disc(self.prob2ent(trg_preds.detach()),
                              target = self.trg_lbl) / 2 # for slowing learning speed
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            disc_loss = disc_loss.mean()
        disc_loss.backward()
        

        # finish training step
        self.disc_optim.step()

        src_pred = preds.argmax(1)
        src_pred = src_pred.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, src_pred)

        cur_losses = {'seg_loss': loss.mean().item(),
                      'adv_loss': adv_loss.item(),
                      'disc_loss': disc_loss.item()}

        return cur_losses





class Trainer_Deeplab(object):
    def __init__(self, model, batch, optimizer, criterion, metric):
        self.model = model
        self.batch = batch
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def training_step(self, batched_data):
        img_batch = images_transform(batched_data['img'])
        lbl_batch = torch_resizer(masks_transform(batched_data['lbl']))
        
        self.optimizer.zero_grad()

        preds = self.model.forward(img_batch)
        loss = self.criterion(preds, lbl_batch)

        loss.backward()
        self.optimizer.step()

        preds = preds.argmax(1)
        preds = preds.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, preds)

        return loss.item()

class Trainer_PSP(object):
    def __init__(self, model, batch, optimizer, criterion, aux_criterion, metric):
        self.model = model
        self.batch = batch
        self.optimizer = optimizer
        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def training_step(self, batched_data):
        img_batch = images_transform(batched_data['img'])
        lbl_batch = torch_resizer(masks_transform(batched_data['lbl']))
        
        self.optimizer.zero_grad()

        preds, aux_preds = self.model.forward(img_batch)
        aux_loss = self.aux_criterion(aux_preds, lbl_batch)
        loss = self.criterion(preds, lbl_batch)

        loss = loss + config['AUX_LOSS_Weights'] * aux_loss

        loss.backward()
        self.optimizer.step()

        preds = preds.argmax(1)
        preds = preds.cpu().numpy()
        lbl_batch = lbl_batch.cpu().numpy()

        self.metric.update(lbl_batch, preds)

        return loss.item()


class Evaluator_DeepLab(object):
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def eval_step(self, data_batch):
        self.img_batch = images_transform(data_batch['img'])
        lbl_batch = torch_resizer(masks_transform(data_batch['lbl']))
        
        with torch.no_grad():
            preds = self.model.forward(self.img_batch) 

        preds = preds
        preds = preds.argmax(1)
        self.preds = preds.cpu().numpy()
        self.lbl_batch = lbl_batch.cpu().numpy()
        self.metric.update(self.lbl_batch, self.preds)
        
    def get_sample_prediction(self):
        # get single image, lbl, pred for plotting
        self.img_batch = torch_imgresizer(self.img_batch).detach().cpu().numpy()
        
        imgs, lbls, preds = [], [], []
        for i in range(3): # show 3 images
            
            img = np.transpose(self.img_batch[i,...], (1,2,0))
            lbl = self.lbl_batch[i,...]
            pred = self.preds[i,...]
            
            imgs.append((img*255).astype(np.uint8))
            lbls.append(g2c(lbl.astype(np.uint8)))
            preds.append(g2c(pred.astype(np.uint8)))
        
        return imgs + lbls + preds

class Evaluator_PSP(object):
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
    
    def get_scores(self):
        return self.metric.get_scores()

    def reset_metric(self):
        self.metric.reset()
    
    def eval_step(self, data_batch):
        self.img_batch = images_transform(data_batch['img'])
        lbl_batch = torch_resizer(masks_transform(data_batch['lbl']))
        
        with torch.no_grad():
            preds, _ = self.model.forward(self.img_batch) 

        preds = preds
        preds = preds.argmax(1)
        self.preds = preds.cpu().numpy()
        self.lbl_batch = lbl_batch.cpu().numpy()
        self.metric.update(self.lbl_batch, self.preds)
        
    def get_sample_prediction(self):
        # get single image, lbl, pred for plotting
        self.img_batch = torch_imgresizer(self.img_batch).detach().cpu().numpy()
        
        imgs, lbls, preds = [], [], []
        for i in range(3): # show 3 images
            
            img = np.transpose(self.img_batch[i,...], (1,2,0))
            lbl = self.lbl_batch[i,...]
            pred = self.preds[i,...]
            
            imgs.append((img*255).astype(np.uint8))
            lbls.append(g2c(lbl.astype(np.uint8)))
            preds.append(g2c(pred.astype(np.uint8)))
        
        return imgs + lbls + preds

def eval_wrapper(evaluator, model, val_loader, total_avg_viou):

    model.eval() # <-set mode important
    va = []
    vbar = tqdm(val_loader)
    for step, val_batch in enumerate(vbar):
        with torch.no_grad():
            evaluator.eval_step(val_batch)
            viou = evaluator.get_scores()
            evaluator.reset_metric()

        va.append(viou['iou_mean'])
        vbar.set_description(f'Validation - v_mIOU {viou["iou_mean"]:.4f}')

    img_gt_pred = evaluator.get_sample_prediction()
    tiled = imgviz.tile(img_gt_pred, shape=(3,3), border=(255,0,0))
    tiled = cv2.resize(tiled, (224,224))# just for visulaization
    # plt.imshow(tiled)
    avg_viou = np.nanmean(va)
    total_avg_viou.append(avg_viou)
    curr_viou = np.nanmax(total_avg_viou)

    return curr_viou, avg_viou, total_avg_viou, tiled