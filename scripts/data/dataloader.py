
import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.utils.data as data
from fmutils import fmutils as fmu
from empatches import EMPatches
from tabulate import tabulate
# from PIL import Image
import cv2
import numpy as np
import os, random, time

import torch
from data.augmenters import data_augmenter
from data.utils import std_norm
from pathlib import Path

emp = EMPatches()

class GEN_DATA_LISTS():
    
    def __init__(self, root_dir, sub_dirname):
        '''
        Parameters
        ----------
        root_dir : TYPE
            root directory containing [train, test, val] folders.
        sub_dirname : TYPE
            sub directories inside the main split (train, test, val) folders
        get_lables_from : TYPE
            where to get the label from either from dir_name of file_name.


        '''
        self.root_dir = root_dir
        self.sub_dirname = sub_dirname
        self.splits = ['train', 'val', 'test']
        
    def get_splits(self):
        
        print('Directories loadded:')
        self.split_files = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[0]))
            self.split_files.append(os.path.join(self.root_dir, split, self.sub_dirname[0]))
        print('\n')
        self.split_lbls = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            self.split_lbls.append(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            
        
        self.train_f = fmu.get_all_files(self.split_files[0])
        self.val_f = fmu.get_all_files(self.split_files[1])
        self.test_f = fmu.get_all_files(self.split_files[2])

        self.train_l = fmu.get_all_files(self.split_lbls[0])
        self.val_l = fmu.get_all_files(self.split_lbls[1])
        self.test_l = fmu.get_all_files(self.split_lbls[2])
        
        train, val, test = [self.train_f, self.train_l], [self.val_f, self.val_l], [self.test_f, self.test_l]
        
        return train, val, test
    
    def get_classes(self):
        
        cls_names = []
        for i in range(len(self.train_f)):
            cls_names.append(fmu.get_basename(self.train_f[i]).split('_')[1])
        classes = sorted(list(set(cls_names)), key=fmu.numericalSort)
        return classes
    
    def get_filecounts(self):
        print('\n')
        result = np.concatenate((np.asarray(['train', 'val', 'test']).reshape(-1,1),
                                np.asarray([len(self.train_f), len(self.val_f), len(self.test_f)]).reshape(-1,1),
                                np.asarray([len(self.train_l), len(self.val_l), len(self.test_l)]).reshape(-1,1))
         , 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Split", "Images", "Labels"], tablefmt="github"))
        return None

class CWD26(data.Dataset):
    def __init__(self, img_paths, mask_paths, img_height, img_width, augment_data=False, normalize=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_height = img_height
        self.img_width = img_width
        self.augment_data = augment_data
        self.normalize = normalize
        self.my_epoch = 1
        self.idx = 0

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        data_sample = {}
        # print(self.img_paths[index])
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        lbl = cv2.imread(self.mask_paths[index], 0)
        lbl = cv2.resize(lbl, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        if self.augment_data:
            img, lbl, a, b = data_augmenter(img, lbl, self.my_epoch)
        else:
            a, b = 0, 0 # no augmentation

        if self.normalize:
            img = std_norm(img)
        
        assert len(np.unique(lbl)) <= config['num_classes'], f'A total of {len(np.unique(lbl))} labels found in {self.mask_paths[index]}.'
        
        self.idx += 1
        if len(self.img_paths) / self.idx < 2: # one iteration over data finished b/c (x+1)/x < 2.
            self.my_epoch += 1
            self.idx = 0 # reset

        data_sample['img'] = img
        data_sample['lbl'] = lbl
        data_sample['geo_augs'] = a
        data_sample['noise_augs'] = b
        
        return data_sample 
    
    def return_epoch(self):
        return self.my_epoch

def inference_loader(img_path, lbl_path=None):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    filename = Path(img_path).stem
    orig_h, orig_w = img.shape[:2] # save origina height and width for writing predictions to file

    img = cv2.resize(img, (config['img_width'], config['img_height']), interpolation=cv2.INTER_LINEAR)
    img = std_norm(img)

    if lbl_path is not None:
        lbl = cv2.imread(lbl_path, 0)
        lbl = cv2.resize(lbl, (config['img_width'], config['img_height']), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        return img, lbl, orig_h, orig_w, filename
    else:
        return img, orig_h, orig_w, filename
    
def get_segment_boxes_and_confidences(seg_mask, class_dict, relative_coords=True):
    H, W, C = seg_mask.shape
    boxes = []
    confidences = []
    classes = []

    for i, class_name in enumerate(class_dict.keys()):
        class_idx = class_dict[class_name]
        class_mask = seg_mask[:, :, class_idx]

        # Convert to binary mask
        binary_mask = (class_mask > 0.5).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        for j in range(1, num_labels):
            x, y, w, h, _ = stats[j]

            if relative_coords:
                # Convert to relative coordinates
                x_min, y_min, x_max, y_max = x/W, y/H, (x + w)/W, (y + h)/H
                boxes.append([x_min, y_min, x_max, y_max])  # xmin, ymin, xmax, ymax
            else:
                boxes.append([x, y, x + w, y + h])  # xmin, ymin, xmax, ymax

            # Get segment confidence
            segment_mask = (labels == j)
            segment_confidence = torch.tensor(class_mask[segment_mask]).mean().item()
            confidences.append(segment_confidence)
            
            # Append class name
            classes.append(class_name)

    return boxes, confidences, classes

def write_boxes_and_confidences_to_file(filename, boxes, confidences, classes, orig_h, orig_w):
    with open(f'/home/user01/data/talha/CMED/preds/{filename}.txt', 'w') as f:
        for class_name, confidence, box in zip(classes, confidences, boxes):
            x_min, y_min, x_max, y_max = box
            f.write(f'{class_name} {np.round(confidence, 5)} {int(x_min*orig_w)} {int(y_min*orig_h)} {int(x_max*orig_w)} {int(y_max*orig_h)}\n')


def write_eval_txt_files(preds, class_dict, filename, orig_h, orig_w):
    probs = preds.softmax(1).permute(0,2,3,1).cpu().numpy().squeeze()
    boxes, confidences, classes = get_segment_boxes_and_confidences(probs, class_dict)
    write_boxes_and_confidences_to_file(filename, boxes, confidences, classes, orig_h, orig_w)


# import csv
# import os

# def write_boxes_and_confidences_to_file(preds, filename, class_dict):
#     probs = preds.softmax(1).permute(0,2,3,1).cpu().numpy().squeeze()
#     boxes, confidences, classes = get_segment_boxes_and_confidences(probs, class_dict)
#     # Check if file exists, write headers only if it doesn't
#     file_exists = os.path.isfile(f'/home/user01/data/talha/CMED/lcm_400x_test.csv')
    
#     with open(f'/home/user01/data/talha/CMED/lcm_400x_test.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
        
#         if not file_exists:
#             writer.writerow(["filename", "class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]) # write header
            
#         for class_name, confidence, box in zip(classes, confidences, boxes):
#             x_min, y_min, x_max, y_max = box
#             writer.writerow([filename, class_name, confidence, x_min, y_min, x_max, y_max])
