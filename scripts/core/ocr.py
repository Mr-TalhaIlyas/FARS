'''
      ___           ___           ___     
     /  /\         /  /\         /  /\    
    /  /::\       /  /:/        /  /::\   
   /  /:/\:\     /  /:/        /  /:/\:\  
  /  /:/  \:\   /  /:/  ___   /  /:/~/:/  
 /__/:/ \__\:\ /__/:/  /  /\ /__/:/ /:/___
 \  \:\ /  /:/ \  \:\ /  /:/ \  \:\/:::::/
  \  \:\  /:/   \  \:\  /:/   \  \::/~~~~ 
   \  \:\/:/     \  \:\/:/     \  \:\     
    \  \::/       \  \::/       \  \:\    
     \__\/         \__\/         \__\/    

https://arxiv.org/pdf/1909.11065.pdf
'''
#%%
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import math
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F
from core.bricks import twiceConvBNRelu, ConvBNRelu, AUX_Head, resize

class SpatinalGather_Module(nn.Module):

    def __init__(self):
        super(SpatinalGather_Module, self).__init__()
    
    def forward(self, bb_feats, aux_out):

        B, K, H, W = aux_out.shape # K is num of classes
        aux_out = aux_out.view(B, K, -1) # B x K x HW 
        bb_feats = bb_feats.view(B, bb_feats.shape[1], -1) # B x OCR_Ch x HW

        bb_feats = bb_feats.permute(0, 2, 1) # B x HW x OCR_Ch
        aux_out = F.softmax(aux_out, dim=2) # spatial softmax on HW dim

        ocr_context = torch.bmm(aux_out, bb_feats) # B x K x HW @ B x HW x OCR_Ch -> B x K x OCR_Ch
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3) # -> B x OCR_Ch x K x 1

        return ocr_context

class SpatialOCR_Module(nn.Module):
    
    def __init__(self, num_classes, ocr_ch=512, ocr_qkv_ch=256):
        super(SpatialOCR_Module, self).__init__()
        self.num_classes = num_classes
        self.ocr_ch = ocr_ch
        self.ocr_qkv_ch = ocr_qkv_ch

        # set output channels as mentioned in paper.
        self.psi   = twiceConvBNRelu(self.ocr_ch, self.ocr_qkv_ch, kernel=1)
        self.phi   = twiceConvBNRelu(self.ocr_ch, self.ocr_qkv_ch, kernel=1)
        self.delta = twiceConvBNRelu(self.ocr_ch, self.ocr_qkv_ch, kernel=1)
        self.rho   = ConvBNRelu(self.ocr_qkv_ch, self.ocr_ch, kernel=1)
        self.g     = ConvBNRelu(sum([self.ocr_ch,self.ocr_ch]), self.ocr_ch, kernel=1)
    
    def forward(self, bb_feats, context):
        B, _, _, _ = bb_feats.shape
        query = self.psi(bb_feats).view(B, self.ocr_qkv_ch, -1) # B x C x H x W -> B x C x HW
        query = query.permute(0, 2, 1) # B x HW x C
        
        key = self.phi(context).view(B, self.ocr_qkv_ch, -1) # B x C x K x 1 -> B x C x K

        value = self.delta(context).view(B, self.ocr_qkv_ch, -1) # B x C x K x 1 -> B x C x K
        value = value.permute(0, 2, 1) # B x K x C
        # pixel region relation / W
        prr = torch.bmm(query, key) # B x HW x C @ B x C x K -> B x HW x K
        # scale down
        prr = (self.ocr_qkv_ch ** 0.5) * prr
        prr = F.softmax(prr, dim=-1)
        # Object context representation / X_o
        obj_context = torch.bmm(prr, value) # B x HW x K @ B x K x C -> B x HW x C
        obj_context = obj_context.permute(0, 2, 1).contiguous() # B x C x HW
        obj_context = obj_context.view(B, self.ocr_qkv_ch, *bb_feats.shape[2:]) # B x C x H x W
        obj_context = self.rho(obj_context) #              ^___this one is a fancy way to pass *args ;)
        
        X_o = torch.cat([obj_context, bb_feats], 1)
        aug_repr = self.g(X_o) # augmented representation.
        
        return aug_repr

class OCR_Block(nn.Module):
    def __init__(self, num_classes, embed_dims=[32,64,160,256], ocr_ch=512, ocr_qkv_ch=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.ocr_ch = ocr_ch
        self.ocr_qkv_ch = ocr_qkv_ch

        self.aux_head = AUX_Head(embed_dims[-2], num_classes)
        self.conv_ip = ConvBNRelu(embed_dims[-1], self.ocr_ch)
        self.sgm = SpatinalGather_Module()
        self.ocr = SpatialOCR_Module(num_classes=num_classes)
    
    def forward(self, feats): # features will be 4 stage input.

        feats = feats[2:] # drop stage 1 and 2 features b/c we don't need those
        feats = [resize(feat, size=feats[-2].shape[2:], mode='bilinear') for feat in feats]

        self.aux_feats = feats[-2]  # get 3rd stage features # B x C x H x W
        bb_feats = feats[-1] # last stage feats # B x C' x H x W

        aux_out = self.aux_head(self.aux_feats) # B x K x H x W
        bb_feats = self.conv_ip(bb_feats)## B x OCR_Ch x H x W

        ocr_context = self.sgm(bb_feats, aux_out)

        aug_repr = self.ocr(bb_feats, ocr_context)

        return aux_out, aug_repr
#%%

# from torchsummary import summary
# model = MSCANet(in_channnels=3, embed_dims=[32, 64, 460,256],
#                  ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2],
#                  num_stages = 4, ls_init_val=1e-2, drop_path=0.0)

# y = torch.randn((1,3,1024,2048))#.to('cuda' if torch.cuda.is_available() else 'cpu')
# x = model.forward(y)

# for i in range(4):
#     print(x[i].shape)

# ocr = OCR_Block(num_classes=19)
# op = ocr.forward(x)

# print(op.shape)
