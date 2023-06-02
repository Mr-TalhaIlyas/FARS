import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import cv2, imgviz

# torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
                                 # this transfrom converts BHWC -> BCHW and 
                                 # also divides the image by 255 by default if values are in range 0..255.
                                 transforms.ToTensor(),
                                ])
stride = config['output_stride']
torch_resizer = transforms.Compose([transforms.Resize(size=(config['img_height']//stride, config['img_width']//stride),
                                                interpolation=transforms.InterpolationMode.NEAREST)])
torch_imgresizer = transforms.Compose([transforms.Resize(size=(config['img_height']//stride, config['img_width']//stride),
                                                interpolation=transforms.InterpolationMode.BILINEAR)])
def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    
    img = [ b['img'] for b in batch ] # w, h
    lbl = [ b['lbl'] for b in batch ]
    a = [ b['geo_augs'] for b in batch ]
    b = [ b['noise_augs'] for b in batch ]
    
    return {'img': img, 'lbl': lbl, 'geo_augs': a, 'noise_augs': b}

normalize = lambda x, alpha, beta : (((beta-alpha) * (x-np.min(x))) / (np.max(x)-np.min(x))) + alpha
standardize = lambda x : (x - np.mean(x)) / np.std(x)

def std_norm(img, norm=True, alpha=0, beta=1):
    '''
    Standardize and Normalizae data sample wise
    alpha -> -1 or 0 lower bound
    beta -> 1 upper bound
    '''
    img = standardize(img)
    if norm:
        img = normalize(img, alpha, beta)
        
    return img

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets) 
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().to('cuda' if torch.cuda.is_available() else 'cpu')

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs


def decode_preds(mask, img, font_size=60):
    
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                       interpolation = cv2.INTER_NEAREST)
    
    op = imgviz.label2rgb(label=mask,
                          image=(img*255).astype(np.uint8),
                          alpha=0.5,
                          label_names=config['sub_classes'],
                          font_size=font_size,
                          thresh_suppress=0,
                          colormap=np.asarray(config['pallet_sub_class']),
                          loc="rb",
                          font_path=None,
                          )
    return op