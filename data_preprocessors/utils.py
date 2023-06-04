#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:54:09 2023

@author: user01
"""

import xmltodict
import seaborn as sns
import cv2, glob, os
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import numpy as np
from scipy.ndimage import label
import xml.etree.ElementTree as ET
from xml.dom import minidom
from skimage.segmentation import find_boundaries
from gray2color import gray2color
from skimage import measure
# g2c = lambda x : gray2color(x, use_pallet='pannuke')

def get_info_from_xml(xml_path):
    '''
    Parameters
    ----------
    xml_path : path to corresponding xml file
    boxes : all the b_box coordinates array
    Returns:
    ----------
    '''
    # reading xml file and converting into dictionary
    filepath = xml_path
    
    try:
        full_dict = xmltodict.parse(open( filepath , 'rb' ))
    except (FileNotFoundError, OSError):
        full_dict = xmltodict.parse(filepath)
    
    # Extracting the coords and class names from xml file
    names = []
    coords = []
    
    obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
    # file_name = os.path.basename(filepath)[:-4]+'.jpg'#full_dict[ 'annotation' ][ 'filename' ]
    # imgH = int(float(full_dict[ 'annotation' ][ 'size' ]['height']))
    # imgW = int(float(full_dict[ 'annotation' ][ 'size' ]['width']))
    for i in range(len(obj_boxnnames)):
        # 1st get the name and indices of the class
        try:
            obj_name = obj_boxnnames[i]['name']
        except:
            obj_name = obj_boxnnames['name']  # if the xml file has only one object key
            
        # 2nd get tht bbox coord and append the class name at the end
        try:
            obj_box = obj_boxnnames[i]['bndbox']
        except:
            obj_box = obj_boxnnames['bndbox'] # if the xml file has only one object key
        bounding_box = [0.0] * 4                    # creat empty list
        bounding_box[0] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
        bounding_box[1] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
        bounding_box[2] = int(float(obj_box['xmax']))# still convert them to int
        bounding_box[3] = int(float(obj_box['ymax']))
        # convert absolute coords to relative coords
        # bounding_box[0] = bounding_box[0] * 1 / imgW
        # bounding_box[1] = bounding_box[1] * 1 / imgH
        # bounding_box[2] = bounding_box[2] * 1 / imgW
        # bounding_box[3] = bounding_box[3] * 1 / imgH
        names.append(obj_name)
        coords.append(bounding_box)
        
    return np.asarray(names).astype('<U16'), np.asarray(coords)

def draw_boxes(image_in, confidences, nms_box, det_classes, classes, modified=False):
    '''
    Parameters
    ----------
    image : RGB image original shape will be resized
    confidences : confidence scores array, shape (None,)
    nms_box : all the b_box coordinates array after NMS, shape (None, 4) => order [y_min, x_min, y_max, x_max]
    det_classes : shape (None,), names  of classes detected
    classes : all classes names in dataset
    '''
    # boxes = (boxes).astype(np.uint16)
    
    image = image_in / 255
    boxes = (nms_box).astype(np.uint16)
    i = 1

    colors =  sns.color_palette("bright") #+ sns.color_palette("tab10")
    # colors.pop(2) # remove Green
    # colors.pop(2) # remove Red
    # colors.pop(5) # remove Gray
    colors.pop(0) 
    [colors.extend(colors) for i in range(6)]
    bb_line_tinkness = 2
    for result in zip(confidences, boxes, det_classes, colors):
        conf = float(result[0])
        facebox = result[1].astype(np.int16)
        #print(facebox)
        name = result[2]

        if modified: # pred
            color = (1., 0., 0.) # red  
            bb_line_tinkness = 2
            label = 'M'
        if not modified: # gt
            color = (0., 0.647, 1.)  # cyan 
            bb_line_tinkness = 2
            label = 'O'
        
        
        cv2.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), color, bb_line_tinkness)#255, 0, 0

        
        
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX   , 0.7, 1)
            
        cv2.rectangle(image, (facebox[2]-2, facebox[1] - label_size[1]), # top left cornor
                  (facebox[2] + label_size[0]-2, facebox[1] + base_line-1),# bottom right cornor
                  color, cv2.FILLED)
        op = cv2.putText(image, label, (facebox[2], facebox[1]),
               cv2.FONT_HERSHEY_DUPLEX   , 0.7, (0, 0, 0))
                
             
        i = i+1
    return (image*255).astype(np.uint8), boxes, det_classes, np.round(confidences, 3)

def calculate_iou(box1, box2):
    '''
    Parameters
    ----------
    box1 :  array/list/tuple
        original bbox coordinates.
    box2 : array/list/tuple
        blob's bbox coordinates.

    Returns
    -------
    float
        IoU.

    '''
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)
    if (dx >= 0) and (dy >= 0):
        intersection = dx * dy
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        union = area1 + area2 - intersection
        return intersection / union
    else:
        return 0

def blob2bbox(labeled_cc, label_id):
    '''
    Parameters
    ----------
    labeled_cc : connected component label output by scipy.ndimage.lable function.
    label_id : int - which labels bounding box is nedded from all the detected blobs.

    Returns
    -------
     bounding box coordinates in form => xmin, ymin, xmax, ymax
    '''
    # Get blob's bounding box
    blob_coords = np.argwhere(labeled_cc == label_id)
    ymin, xmin = blob_coords.min(axis=0)
    ymax, xmax = blob_coords.max(axis=0)
    
    return xmin, ymin, xmax, ymax


def assign_classes(binary_mask, coords, det_classes, class_dict):
    '''
    Parameters
    ----------
    binary_mask : H*W array only FG and BG class.
    coords : array of all detected/labelled boxes.
    det_classes : array/list of all detected classes in same order as in coords.
    class_dict : dict of all classes present in the dataset.

    Returns
    -------
    s : semantic segmentation mask H*W, each class having unique pixel value.

    '''
    # Assuming 'm' is your binary mask of shape (H, W)
    s = np.zeros_like(binary_mask)  # Initialize semantic mask same as binary mask, all zeros
    
    # Perform connected component labeling on the binary mask
    labeled, num_features = label(binary_mask)
    
    # Iterate through blobs
    for i in range(1, num_features+1):
        # Get blob's bounding box
        xmin, ymin, xmax, ymax = blob2bbox(labeled, i)
    
        # Calculate IoU with original bounding boxes and find the one with max IoU
        max_iou = 0
        max_class = 0
        for j in range(len(coords)):
            iou = calculate_iou(coords[j], (xmin, ymin, xmax, ymax))
            if iou > max_iou:
                max_iou = iou
                max_class = class_dict[det_classes[j]]
    
        # Assign class to blob in semantic segmentation mask
        s[labeled == i] = max_class
    
    return s


def create_data_dir_tree(base_dir, visualize_dir=False, xml_dir=True):
    try:
        base_dir = os.path.join(base_dir, 'processed')
        os.mkdir(base_dir)
    except FileExistsError:
        print('[Warning] Dir already exist in the base dir. kindly chekc it.')
        pass
    # Define the top-level directory
    top_dirs = ['HCM', 'LCM']
    
    # Define the second-level directories
    second_level_dirs = ['100x', '400x', '1000x']
    
    # Define the third-level directories
    third_level_dirs = ['train', 'test', 'val']
    
    # Define the fourth-level directories
    if xml_dir:
        fourth_level_dirs = ['images', 'labels', 'xmls']
    else:
        fourth_level_dirs = ['images', 'labels']
    
    # Iterate through the levels and create directories
    for top_dir in top_dirs:
        for second_dir in second_level_dirs:
            for third_dir in third_level_dirs:
                for fourth_dir in fourth_level_dirs:
                    # Construct the directory path
                    dir_path = os.path.join(base_dir, top_dir, second_dir, third_dir, fourth_dir)
                    
                    # Create the directory
                    os.makedirs(dir_path, exist_ok=True)
    
    if visualize_dir: # just for verification
        try:
            os.mkdir(os.path.join(base_dir,'visualize'))
        except FileExistsError:
            print('[Warning] Visualization Dirs already exist in the base dir. kindly chekc it.')
            pass
        
    print("[INFO] Folders created successfully.")
    
    return None

def mask_to_bounding_boxes(mask, filename, width, height, depth=3):
    
    class_dict = {'ring': 1, 'trophozoite':2, 'schizont':3, 'gametocyte':4}
    # Initialize the XML tree
    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = "LabelledImages"
    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'path').text = '/path/to/' + filename  # Add correct path here

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    ET.SubElement(root, 'segmented').text = '0'

    # Generate a bounding box for each class
    for class_name, class_value in class_dict.items():
        class_mask = (mask == class_value)

        if not class_mask.any():  # If there is no instance of this class in the mask, skip to next class
            continue

        # Label each individual segment
        labeled, num_labels = measure.label(class_mask, connectivity=2, return_num=True)

        for i in range(1, num_labels+1):
            coords = np.where(labeled == i)

            # Calculate the bounding box
            xmin, ymin = np.min(coords[1]), np.min(coords[0])
            xmax, ymax = np.max(coords[1]), np.max(coords[0])

            # Add the object to the XML tree
            object_elem = ET.SubElement(root, 'object')
            ET.SubElement(object_elem, 'name').text = class_name.lower()
            ET.SubElement(object_elem, 'pose').text = 'Unspecified'
            ET.SubElement(object_elem, 'truncated').text = '0'
            ET.SubElement(object_elem, 'difficult').text = '0'

            bndbox = ET.SubElement(object_elem, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # Return the prettified XML string
    xml_string = ET.tostring(root, 'utf-8')
    parsed_xml = minidom.parseString(xml_string)
    return parsed_xml.toprettyxml(indent="  ")

def gray2encoded(y_true, num_class):
    '''
    Parameters
    ----------
    y_true : 2D array of shape [H x W] containing unique pixel values for all N classes i.e., [0, 1, ..., N] 
    num_class : int no. of classes inculding BG
    Returns
    -------
    encoded_op : one-hot encoded 3D array of shape [H W N] where N=num_class
    '''

    # Ensure y_true is integer type
    y_true = y_true.astype('int32')
    
    # One-hot encoding
    encoded_op = np.eye(num_class)[y_true]
    
    return encoded_op.astype('uint8')

def get_sem_bdr(sem, img, blend=True, custom_pallet=None):
    '''
    Parameters
    ----------
    sem : a 2D array of shape [H, W] where containing unique value for each class.
    img : Original RGB image for overlaying the semantic seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each class have a unique RGB border. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    # if you get shape mismatch error try swaping the (w,h) argument of the line below.
    # i.e., from (x.shape[0], x.shape[1]) to (x.shape[1], x.shape[0]).
    # img = cv2.resize(img, (sem.shape[1], sem.shape[0]), interpolation=cv2.INTER_LINEAR) 
    # 1-hot encode all classes 
    sem_enc = gray2encoded(sem, num_class=5)
    # as the in encoded output the 0th channel will be BG we don't need it so
    sem_enc = sem_enc[:,:,1:]
    # get boundaries of thest isolated instances
    temp = np.zeros(sem_enc.shape)
    
    for i in range(sem_enc.shape[2]):
        temp[:,:,i] = find_boundaries(sem_enc[:,:,i], connectivity=1, mode='thick', background=0)
    
    dummy = np.zeros((temp.shape[0], temp.shape[1], 1))
    temp =  np.concatenate((dummy, temp), axis=-1)
        
    sem_bdr = np.argmax(temp, axis=-1).astype(np.uint8)
    sem_bdr_rgb = gray2color(sem_bdr, use_pallet='pannuke', custom_pallet=custom_pallet)
    if blend:
        inv = 1 - cv2.threshold(sem_bdr.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, sem_bdr_rgb)
    else:
        blend = sem_bdr_rgb
    return blend.astype(np.uint8)
