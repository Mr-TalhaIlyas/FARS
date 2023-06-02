# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:31:45 2022

@author: talha
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

import cv2, random
import numpy as np
from tqdm import trange
from cvt_color import YcbcrToRgb, RgbToYcbcr

ycc2rgb = YcbcrToRgb()
rgb2ycc = RgbToYcbcr()

def extract_ampl_phase(fft_im):
    # fft_im: size should be b x 3 x h x w
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    # multiply w by 2 because we have only half the space as rFFT is used
    w *= 2
    # multiply by 0.5 to have the maximum b for L=1 like in the paper
    b = (np.floor(0.5 * np.amin((h, w)) * L)).astype(int)     # get b
    if b > 0:
        # When rFFT is used only half of the space needs to be updated
        # because of the symmetry along the last dimension
        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
        amp_src[:, :, h-b+1:h, 0:b] = amp_trg[:, :, h-b+1:h, 0:b]    # bottom left
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # get fft of both source and target
    fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[imgH, imgW])

    return src_in_trg

scale = lambda x, alpha, beta: (((beta-alpha) * (x-torch.min(x))) / (torch.max(x)-torch.min(x))) + alpha

def FDA(im_src, im_trg, L=0.001, space='ycrcb', scale_back=True):
    '''
    Where img_src and img_trg are tensors of size 1 x 3 x h x w
    '''
    # print('inside of FDA')
    if space=='ycrcb':
        im_src = rgb2ycc(im_src)
        im_trg = rgb2ycc(im_trg)
    
    src_in_trg = FDA_source_to_target( im_src, im_trg, L=L)
    
    if space=='ycrcb':
        src_in_trg = ycc2rgb(src_in_trg)

    if scale_back:
        src_in_trg = scale(src_in_trg, alpha=0, beta=1)
    
    return src_in_trg

# def resize_std_norm(img, size=(512,512)):
#     # first resize than normalize
#     img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
#     img = std_norm(img)
#     return img

# class DomainAdapt:

#     def __init__(self, trg_img_tensor):

#         print('=> Loading and Adjusting Target Domain Tensors...')
        
#         self.trg_img_tensor = np.load(trg_img_tensor)#[0:10,...] # for testing
        
#         self.trg_img_tensor = np.asarray([resize_std_norm(self.trg_img_tensor[i,...], (config['patch_size'],config['patch_size'])) \
#                                         for i in trange(self.trg_img_tensor.shape[0], desc='Preprocessing Tensor')])
        
#         self.trg_img_tensor = torch.from_numpy(self.trg_img_tensor).permute(0,3,1,2) # BCHW
#         self.indices = np.arange(self.trg_img_tensor.shape[0])
        
#         print('[INFO] Traget Domain Tensors Loaded.')
    
#     def __call__(self, img_src, L=0.002, img_trg=None, space = 'ycrcb'):

#         if img_trg is None:
#             np.random.shuffle(self.indices)
#             a = torch.from_numpy(self.indices[0:config['batch_size']])     
#             trg_imgs = torch.index_select(self.trg_img_tensor, dim=0, index=a).to('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             trg_imgs = img_trg.to('cuda' if torch.cuda.is_available() else 'cpu')

#         src_in_trg = FDA(img_src, trg_imgs, L=L, space=space)

#         return src_in_trg, trg_imgs

# class DARegulator:

#     def __init__(self, trg_img_tensor, max_prob=0.7, till_epoch=60):
#         self.da_porb = np.linspace(0, max_prob, till_epoch)
#         self.id_prob = 1 - self.da_porb
#         self.domainadapt = DomainAdapt(trg_img_tensor)

#     def get_prob(self, current_epoch):
        
#         if current_epoch < len(self.da_porb):
#             a = self.da_porb[current_epoch]
#             b = self.id_prob[current_epoch]
#         else:
#             a = self.da_porb[-1]
#             b = self.id_prob[-1]
        
#         return a, b

#     def identity(self, img_batch, L):
#         # just pass through
#         return img_batch, img_batch

#     def __call__(self, img_batch, epoch, L=0.002):

#         func_args = [
#                     (self.domainadapt, (img_batch, L)),
#                     (self.identity,    (img_batch, L))
#                     ]
        
#         prob_da, porb_id = self.get_prob(epoch)

#         (func, args), = random.choices(func_args, weights=[prob_da, porb_id])
        
#         img_batch, trg_img = func(*args)
        
#         return img_batch, trg_img, prob_da

















        # self.trg_img_tensor = F.interpolate(self.trg_img_tensor,
        #                                     (config['patch_size'],config['patch_size']),
        #                                     mode='nearest')

    # def apply(self, img_src, L=0.001, space = 'ycrcb'):
        
    #     np.random.shuffle(self.indices)
    #     idx = np.random.randint(0, self.trg_img_tensor.shape[0])
    #     trg_img = self.trg_img_tensor[idx:idx+1, ...].to('cuda' if torch.cuda.is_available else 'cpu')

    #     src_in_trg = FDA(img_src, trg_img, L=L, space=space)

    #     return src_in_trg, trg_img, idx
################################################################
#  USAGE
################################################################

# im_src = cv2.imread('C:/Users/talha/Desktop/Crops and Weeds/Domain Adaptation/code/src/synth_multi_w5_c40_w42_bg_015_pd_000187.jpg')
# im_src = cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB)
# im_src = cv2.resize(im_src, (2048,2048))
# im_src = std_norm(im_src)

# im_trg = cv2.imread("C:/Users/talha/Desktop/Crops and Weeds/Domain Adaptation/code/trg/2022-07-03_yoon_bean-openf_90_1657277062536.jpg")
# im_trg = cv2.cvtColor(im_trg, cv2.COLOR_BGR2RGB)
# im_trg = cv2.resize(im_trg, (2048,2048))
# im_trg = std_norm(im_trg)


# op = FDA(im_src, im_trg, L=0.001, space='ycrcb')
# plt.figure(1)
# plt.imshow(op)

# plt.figure(2)
# plt.imshow(std_norm(op))
# #%%
# x = np.where(op<0, 0, 1)
# plt.imshow(x[...,0])
#%%
# img = cv2.imread('E:/depth_new_rgb_color/img/img_357.png')

# mask = cv2.imread('E:/depth_new_rgb_color/depth_colormap2/img_357.png')
# plt.imshow(cv2.addWeighted(img, 0.7, mask, 0.3, 0.0))
# plt.axis('off')