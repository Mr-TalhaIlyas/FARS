# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:19:24 2023

@author: talha
"""
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
from fmutils import fmutils as fmu
from fda_torch import FDA
from gray2color import gray2color

g2c = lambda x : gray2color(x, use_pallet='pannuke')

lcm = cv2.imread('C:/Users/talha/Desktop/malaria paper/processed/LCM/400x/train/images/Malaria_CM3_21Jun2021123244_0010_126.0_16.4_400x.png')
lcm = cv2.cvtColor(lcm, cv2.COLOR_BGR2RGB)
lcm = cv2.resize(lcm, (512,512))

plt.imshow(lcm)

hcm = cv2.imread('C:/Users/talha/Desktop/malaria paper/processed/HCM/400x/train/images/Malaria_CM3_21Jun2021123244_0010_126.0_16.4_400x.png')
hcm = cv2.cvtColor(hcm, cv2.COLOR_BGR2RGB)
hcm = cv2.resize(hcm, (512,512))


hcm_lbl = cv2.imread('C:/Users/talha/Desktop/malaria paper/processed/HCM/400x/train/labels/Malaria_CM3_21Jun2021123244_0010_126.0_16.4_400x.png', -1)
hcm_lbl = cv2.resize(hcm_lbl, (512,512), interpolation=cv2.INTER_NEAREST)
lbl = g2c(hcm_lbl)
# overlay lbl on hcm
ov = cv2.addWeighted(hcm, 0.5, lbl, 0.5, 0)
plt.imshow(ov)
#%%
lcm = torch.from_numpy(lcm)
# from hxwxc to 1xcxhxw
lcm = lcm.permute(2,0,1).unsqueeze(0)
# do same for hcm
hcm = torch.from_numpy(hcm)
hcm = hcm.permute(2,0,1).unsqueeze(0)


# %%

op = FDA(lcm, hcm, L=0.001, space='rgb')
# op to numpy image
op = op.squeeze(0).permute(1,2,0).numpy()

plt.imshow(op)
# %%


class_dict = {'ring': 1, 'trophozoite':2, 'schizont':3, 'gametocyte':4, 'Ring' : 1}

filename = 'test'
xml = mask_to_bounding_boxes(hcm_lbl, class_dict, filename, hcm_lbl.shape[1], hcm_lbl.shape[0], depth=3)

with open(f'C:/Users/talha/Desktop/malaria paper/{filename}.xml', 'w') as file:
    file.write(xml)
# %%
