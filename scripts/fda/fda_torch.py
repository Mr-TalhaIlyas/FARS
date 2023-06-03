# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:31:45 2022

@author: talha
"""

import yaml
with open('./config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
import torch.nn.functional as F
import torch.nn as nn

import cv2, random
import numpy as np
from tqdm import trange
from fda.cvt_color import YcbcrToRgb, RgbToYcbcr

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

class DomainAdapter:
    def __init__(self, L=0.01, space='ycrcb'):
        self.scale = lambda x, alpha, beta: (((beta-alpha) * (x-torch.min(x))) / (torch.max(x)-torch.min(x))) + alpha
        self.L = L
        self.space = space

    def identity(self, src_batch, trg_batch):
        # just pass through
        return src_batch
    
    def domainadapt(self, src_batch, trg_batch):
        for i in range(len(src_batch)):
            src_batch[i:i+1, ...] = FDA(src_batch[i:i+1, ...], trg_batch[i:i+1, ...],
                                        L=self.L, space=self.space)
            # FFT changes the pixel values so scale them back between 0 and 1.
            src_batch[i:i+1, ...] = self.scale(src_batch[i:i+1, ...], alpha=0, beta=1)
        
        return src_batch
    
    def apply_fda(self, src_batch, trg_batch):
        func_args = [
            (self.domainadapt, (src_batch, trg_batch)),
            (self.identity, (src_batch, trg_batch))
        ]
        
        (func, args), = random.choices(func_args, weights=[config['prob_fda'], 1-config['prob_fda']])
        
        src_batch = func(*args)
        
        return src_batch
