#%%
import yaml, math, os
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn.functional as F
import torch.nn as nn

from core.backbone import MSCANet
from core.decoder import DecoderHead, HamDecoder, OCRDecoder

from core.losses import FocalLoss, Entropy, LovaszSoftmax

class UHDNext(nn.Module):
    '''Different Decoder then SegNext'''
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, ls_init_val=1e-2, drop_path=0.0, drop_path_mode='row',
                 config=config):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               ls_init_val=ls_init_val, drop_path=drop_path, drop_path_mode=drop_path_mode)
        self.decoder = DecoderHead(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        
        self.init_weights()
        self.encoder_init_weights()
        

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)

        return output

    def init_weights(self):
        print('Initializing weights...')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)
                # xavier_uniform_() tf default

    def encoder_init_weights(self, pretrained_path="/home/user01/data/talha/CWD26/pretrained/clf_pretrain_4_single_seg_XL.pth"):

        print('Encoder init_weights...')
        chkpt = torch.load(pretrained_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # load pretrained
            pretrained_dict = chkpt['model_state_dict']
            # load model state dict
            state = self.encoder.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('module.') == nk.strip('module.') and pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.encoder.load_state_dict(state)
            print('Pre-trained state loaded successfully (Encoder), summary...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            print(f'ERROR in pretrained_dict @ {pretrained_path}')

#-----------------------------------END UHDNext Class------------------------------------------
#//////////////////////////////////////////////////////////////////////////////////////////////
class UHD_OCR(nn.Module):
    '''Different Decoder then SegNext'''
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=512, ls_init_val=1e-2, drop_path=0.0, drop_path_mode='row',
                 config=config):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))

        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               ls_init_val=ls_init_val, drop_path=drop_path, drop_path_mode=drop_path_mode)

        self.ocr = OCRDecoder(num_classes=num_classes, outChannels=dec_outChannels,
                              config=config, enc_embed_dims=embed_dims)
        # define loss here for balance load accross GPUs
        self.criterion = FocalLoss()
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.init_weights()
        self.encoder_init_weights()

    def forward(self, x, target=None):

        enc_feats = self.encoder(x)
        aux_out, dec_out = self.ocr(enc_feats)
        output = self.cls_conv(dec_out)
        # aux_feats = self.ocr.ocr_aux_feats

        if self.training and target is not None:
            loss = self.criterion(output, target)
            aux_loss = self.aux_criterion(aux_out, target)
            return {'loss' : loss, 'aux_loss' : aux_loss}, \
                   {'out' : output, 'aux_out' : aux_out} 
        else:
            return {}, {'out' : output, 'aux_out' : aux_out}  
    
    def init_weights(self):
        print('Initializing weights...')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)
                # xavier_uniform_() tf default

    def encoder_init_weights(self, pretrained_path="/home/user01/data/talha/CWD26/pretrained/single_seg_pretrained_4_sem_seg_XL.pth"):

        print('Encoder init_weights...')
        chkpt = torch.load(pretrained_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # load pretrained
            pretrained_dict = chkpt['model_state_dict']
            # load model state dict
            state = self.encoder.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('module.') == nk.strip('module.') and pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.encoder.load_state_dict(state)
            print('Pre-trained state loaded successfully (Encoder), summary...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            print(f'ERROR in pretrained_dict @ {pretrained_path}')

#////////////////////////////////////////////////////////////////////////////////////////////////////////

# model = UHDNext(num_classes=34, in_channnels=3, embed_dims=[32, 64, 460, 256],
#                  ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
#                   dec_outChannels=256, ls_init_val=1e-2, drop_path=0.0, config=config)
# model = model.to('cuda')
# x = torch.randn((2,3,256,512)).to('cuda')
# y = model.forward(x)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# for name, layer in model.named_modules():
#     print(name, layer)
        
# from functools import reduce

# def get_module_by_name(module, access_string):
#      names = access_string.split(sep='.')
#      return reduce(getattr, names, module)

# get_module_by_name(model, 'stage1.0.msca_block.drop_path')#.__repr__
#%%
import timm

class MaxViTB_512(nn.Module):
    def __init__(self):
        super().__init__()
        print('[INFO] Using MaxViT Base 512 timm model.')
        self.model = timm.create_model(
            'maxvit_base_tf_512.in1k',
            pretrained=True,
            features_only=True,
        )
        
    def forward(self, x):
        # input -> B*C*H*W
        out = self.model(x)
        out = out[1:]
        return out
    

class MaxViT_OCR(nn.Module):
    '''Different Decoder then SegNext'''
    def __init__(self, num_classes, in_channnels=3, embed_dims=[96,192,384,768],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=512, ls_init_val=1e-2, drop_path=0.0, drop_path_mode='row',
                 config=config):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))

        self.encoder = MaxViTB_512()

        self.ocr = OCRDecoder(num_classes=num_classes, outChannels=dec_outChannels,
                              config=config, enc_embed_dims=embed_dims)
        # define loss here for balance load accross GPUs
        self.criterion = FocalLoss(gamma=5)
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # self.aux_criterion = LovaszSoftmax(classes='present', per_image=False)

    def forward(self, x, target=None):

        enc_feats = self.encoder(x)
        aux_out, dec_out = self.ocr(enc_feats)
        output = self.cls_conv(dec_out)
        # aux_feats = self.ocr.ocr_aux_feats

        if self.training and target is not None:
            loss = self.criterion(output, target)
            aux_loss = self.aux_criterion(aux_out, target)
            return {'loss' : loss, 'aux_loss' : aux_loss}, \
                   {'out' : output, 'aux_out' : aux_out} 
        else:
            return {}, {'out' : output, 'aux_out' : aux_out}  
# %%
