#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:08:48 2023

@author: user01
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";

from tqdm import tqdm, trange
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
from fmutils import fmutils as fmu
from segment_anything import sam_model_registry, SamPredictor

from utils import (draw_boxes, get_info_from_xml, assign_classes,
                   create_data_dir_tree, mask_to_bounding_boxes, get_sem_bdr)

from gray2color import gray2color

g2c = lambda x : gray2color(x, use_pallet='pannuke')

from fmutils import fmutils as fmu
import os
from copy import copy

def swapPositions(list, pos1, pos2):
     
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

sam_checkpoint = "/home/user01/data/talha/SAM/chkpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

class_dict = {'ring': 1, 'trophozoite':2, 'schizont':3, 'gametocyte':4, 'Ring' : 1}

data_dir = '/home/user01/data/talha/Extracted/'
processed_data_dir = '/home/user01/data/talha/'

machines = ['HCM', 'LCM']

save_visulizations = True
write_updated_xml = True

create_data_dir_tree(processed_data_dir, save_visulizations, write_updated_xml)
#%%
for machine in machines:
    img_paths = fmu.get_all_files(f"/home/user01/data/talha/Extracted/{machine}/")
    
    for i in trange(len(img_paths), desc= 'Generating segments from B.boxes'):
        # i += 539
        
        try:
            file_name = os.path.basename(img_paths[i])[:-4]
            
            anno_path =  os.path.dirname(img_paths[i]).split('/')
            anno_path.insert(6, 'Annotations')
            if anno_path[0] == '':
                anno_path[0] = os.path.sep
            anno_path = os.path.join(*anno_path)
            
            anno_file = os.path.join(anno_path, f'{file_name}.xml')
            
            # now make the addresses for writing files in processed folder
            
            # base address of processed file
            processed_file = os.path.dirname(img_paths[i].replace(data_dir, os.path.join(processed_data_dir, 'processed/')))
            # swap the final dir structure
            processed_file = swapPositions(processed_file.split('/'),-1,-2)
            #
            if processed_file[0] == '':
                processed_file[0] = os.path.sep
            processed_img = os.path.join(*processed_file, 'images', f'{file_name}.png')
            processed_lbl = os.path.join(*processed_file, 'labels', f'{file_name}.png')
            processed_xml = os.path.join(*processed_file, 'xmls', f'{file_name}.xml')
            
            '''Start reading and processing files and annotations'''
            
            image = cv2.imread(img_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            det_classes, coords = get_info_from_xml(anno_file)
            
            # get imageembeddings
            predictor.set_image(image)
    
            #%
            input_boxes = torch.tensor(coords, device=predictor.device)
    
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    
            masks, _, _ = predictor.predict_torch(point_coords=None,
                                                  point_labels=None,
                                                  boxes=transformed_boxes,
                                                  multimask_output=False)
    
            seg = masks.to(torch.uint8).permute(2,3,0,1).cpu().numpy().squeeze() # Cx1xHxW -> HxWxC
    
            # insure binary
            bin_seg = np.sum(seg, axis=-1).astype(np.uint8)
            ret, bin_seg = cv2.threshold(bin_seg, 0, 1, cv2.THRESH_BINARY)
    
            # convert to semantic
            sem_seg = assign_classes(bin_seg, coords, det_classes, class_dict)
            
            # write updated tight bouding boxes.
            if write_updated_xml:
                xml = mask_to_bounding_boxes(sem_seg, class_dict, file_name, sem_seg.shape[1], sem_seg.shape[0], depth=3)
    
                with open(processed_xml, 'w') as file:
                    file.write(xml)
                
            shutil.copy2(Path(img_paths[i]), Path(processed_img))
            cv2.imwrite(str(Path(processed_lbl)), sem_seg)
            
            if save_visulizations:
                # clr_sem_seg = g2c(sem_seg)
                confidences = np.zeros(det_classes.shape)
                op, _, _, _ = draw_boxes(image, confidences, coords, det_classes, list(class_dict.keys()))
                x = get_sem_bdr(sem_seg, op)
                # x = cv2.addWeighted(op, 0.6, clr_sem_seg, 0.4, 1)
                
                cv2.imwrite(str(Path(processed_data_dir) / 'processed' / 'visualize' / f'{machine}_{file_name}.png'),
                            cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        except:
            print(f'\nError in xml file check {img_paths[i]}')
            pass
        
    print(f'[INFO] {machine} Machine done.')
            

