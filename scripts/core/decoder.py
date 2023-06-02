#%%

from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

from core.ocr import OCR_Block
from core.hamburger import HamBurger
from core.bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize


class OCRDecoder(nn.Module):
    def __init__(self, num_classes, outChannels, config, enc_embed_dims=[32,64,460,256]):
        super().__init__()
        # get ocr
        self.ocr = OCR_Block(num_classes=num_classes, embed_dims=enc_embed_dims,
                             ocr_ch=outChannels, ocr_qkv_ch=outChannels//2)
        # for upsampling S3 feats to concat with s4 feats
        high_res_ch = 48 # as in DeepLabv3+
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # for limiting S2 feats
        self.conv1x1 = ConvBNRelu(enc_embed_dims[1], high_res_ch, kernel=1)
        # for sequeeze and align before and after ham
        self.squeeze = ConvRelu(sum([high_res_ch, outChannels]), outChannels)
        self.align = ConvRelu(outChannels, outChannels)
        # get hams
        self.ham_attn = HamBurger(outChannels, config)
    
    def forward(self, features):
        aux_out, ocr_out = self.ocr(features) # same resolution as S3 feats.
        aux_out, ocr_out = self.up4(aux_out), self.up2(ocr_out) # same resolution as S1, S2 feats.

        s2_fix = self.conv1x1(features[-3])
        s2_ocr = torch.cat([ocr_out, s2_fix], dim=1)

        s2_ocr = self.squeeze(s2_ocr)
        s2_ocr = self.ham_attn(s2_ocr)
        s2_ocr = self.align(s2_ocr)

        s2_ocr = self.up2(s2_ocr)# same resolution as S1 feats.
        # test
        # self.ocr_aux_feats = self.ocr.aux_feats
        
        return aux_out, s2_ocr

class DecoderHead(nn.Module):
    def __init__(self, outChannels, config, enc_embed_dims=[32,64,460,256]):
        super().__init__()

        ham_channels = config['ham_channels']
        # for upsampling S3 feats to concat with s4 feats
        high_res_ch = 48 # as in DeepLabv3+
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # for limiting S2 feats
        self.conv1x1 = ConvBNRelu(enc_embed_dims[1], high_res_ch, kernel=1)
        # for sequeeze and align before and after ham
        self.squeeze1 = ConvRelu(sum(enc_embed_dims[2:4]), ham_channels)
        self.align1 = ConvRelu(ham_channels, ham_channels)

        self.squeeze2 = ConvRelu(sum([high_res_ch, ham_channels]), ham_channels)
        self.align2 = ConvRelu(ham_channels, outChannels)
        # get hams
        self.ham_attn1 = HamBurger(ham_channels, config)
        self.ham_attn2 = HamBurger(ham_channels, config)
    
    def forward(self, features):
        
        s4_up = self.up2(features[-1])
        s34 = torch.cat([features[-2], s4_up], dim=1)

        s34 = self.squeeze1(s34)
        s34 = self.ham_attn1(s34)
        s34 = self.align1(s34)

        s34_up = self.up2(s34)

        s2_fix = self.conv1x1(features[-3])
        s234 = torch.cat([s34_up, s2_fix], dim=1)

        s234 = self.squeeze2(s234)
        s234 = self.ham_attn2(s234)
        s234 = self.align2(s234)

        s234 = self.up2(s234)

        return s234

class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, outChannels, config, enc_embed_dims=[32,64,460,256]):
        super().__init__()

        ham_channels = config['ham_channels']

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, outChannels)
       
    def forward(self, features):
        
        features = features[1:] # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)       

        return x


#%%

# import torch.nn.functional as F

# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):

#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# inputs = [resize(
#         level,
#         size=x[0].shape[2:],
#         mode='bilinear',
#         align_corners=False
#     ) for level in x]

# for i in range(4):
#     print(x[i].shape)
# for i in range(4):
#     print(inputs[i].shape)



# inputs = torch.cat(inputs, dim=1)
# print(inputs.shape)