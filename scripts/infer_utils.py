import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
import cv2, os
import numpy as np
from data.dataloader import std_norm
from pathlib import Path
from data.dataloader import GEN_DATA_LISTS, CWD26
from torch.utils.data import DataLoader
from data.utils import collate

def get_data_loaders(data_dir):
    data_lists = GEN_DATA_LISTS(data_dir, config['sub_directories'])
    _, test_paths, _ = data_lists.get_splits()
    classes = data_lists.get_classes()
    data_lists.get_filecounts()

    test_data = CWD26(test_paths[0], test_paths[1], config['img_height'], config['img_width'],
                        False, config['Normalize_data'])

    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate)
    return test_loader

def get_data_paths(data_dir):

    data_lists = GEN_DATA_LISTS(data_dir, config['sub_directories'])
    _, _, test_paths = data_lists.get_splits()
    img_paths, lbl_paths = test_paths
    return img_paths, lbl_paths


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

class Segmentation2Bbox():
    def __init__(self, class_dict, output_folder, trg_machine, magnification):
        self.class_dict = class_dict
        # os.makedirs(Path(output_folder, trg_machine), exist_ok=True)
        os.makedirs(Path(output_folder, trg_machine, magnification), exist_ok=True)
        self.output_folder = Path(output_folder, trg_machine, magnification)

    def get_segment_boxes_and_confidences(self, seg_mask, relative_coords=True):
        H, W, C = seg_mask.shape
        boxes = []
        confidences = []
        classes = []

        for i, class_name in enumerate(self.class_dict.keys()):
            class_idx = self.class_dict[class_name]
            class_mask = seg_mask[:, :, class_idx]

            binary_mask = (class_mask > 0.5).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

            for j in range(1, num_labels):
                x, y, w, h, _ = stats[j]
                if relative_coords:
                    x_min, y_min, x_max, y_max = x/W, y/H, (x + w)/W, (y + h)/H
                    boxes.append([x_min, y_min, x_max, y_max])
                else:
                    boxes.append([x, y, x + w, y + h])

                segment_mask = (labels == j)
                segment_confidence = torch.tensor(class_mask[segment_mask]).mean().item()
                confidences.append(segment_confidence)
                
                classes.append(class_name)

        return boxes, confidences, classes

    def write_boxes_and_confidences_to_file(self, filename, boxes, confidences, classes, orig_h, orig_w):
        with open(f'{self.output_folder}/{filename}.txt', 'w') as f:
            for class_name, confidence, box in zip(classes, confidences, boxes):
                x_min, y_min, x_max, y_max = box
                f.write(f'{class_name} {np.round(confidence, 5)} {int(x_min*orig_w)} {int(y_min*orig_h)} {int(x_max*orig_w)} {int(y_max*orig_h)}\n')

    def write_eval_txt_files(self, preds, filename, orig_h, orig_w):
        probs = preds.softmax(1).permute(0,2,3,1).cpu().numpy().squeeze()
        boxes, confidences, classes = self.get_segment_boxes_and_confidences(probs)
        self.write_boxes_and_confidences_to_file(filename, boxes, confidences, classes, orig_h, orig_w)


# def get_segment_boxes_and_confidences(seg_mask, class_dict, relative_coords=True):
#     H, W, C = seg_mask.shape
#     boxes = []
#     confidences = []
#     classes = []

#     for i, class_name in enumerate(class_dict.keys()):
#         class_idx = class_dict[class_name]
#         class_mask = seg_mask[:, :, class_idx]

#         # Convert to binary mask
#         binary_mask = (class_mask > 0.5).astype(np.uint8)

#         # Find connected components
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

#         for j in range(1, num_labels):
#             x, y, w, h, _ = stats[j]

#             if relative_coords:
#                 # Convert to relative coordinates
#                 x_min, y_min, x_max, y_max = x/W, y/H, (x + w)/W, (y + h)/H
#                 boxes.append([x_min, y_min, x_max, y_max])  # xmin, ymin, xmax, ymax
#             else:
#                 boxes.append([x, y, x + w, y + h])  # xmin, ymin, xmax, ymax

#             # Get segment confidence
#             segment_mask = (labels == j)
#             segment_confidence = torch.tensor(class_mask[segment_mask]).mean().item()
#             confidences.append(segment_confidence)
            
#             # Append class name
#             classes.append(class_name)

#     return boxes, confidences, classes

# def write_boxes_and_confidences_to_file(filename, boxes, confidences, classes, orig_h, orig_w):
#     with open(f'/home/user01/data/talha/CMED/preds/{filename}.txt', 'w') as f:
#         for class_name, confidence, box in zip(classes, confidences, boxes):
#             x_min, y_min, x_max, y_max = box
#             f.write(f'{class_name} {np.round(confidence, 5)} {int(x_min*orig_w)} {int(y_min*orig_h)} {int(x_max*orig_w)} {int(y_max*orig_h)}\n')


# def write_eval_txt_files(preds, class_dict, filename, orig_h, orig_w):
#     probs = preds.softmax(1).permute(0,2,3,1).cpu().numpy().squeeze()
#     boxes, confidences, classes = get_segment_boxes_and_confidences(probs, class_dict)
#     write_boxes_and_confidences_to_file(filename, boxes, confidences, classes, orig_h, orig_w)


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
