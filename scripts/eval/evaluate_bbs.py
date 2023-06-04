

import matplotlib.pyplot as plt
import numpy as np
import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)

#############################################################
# DEFINE GROUNDTRUTHS AND DETECTIONS
#############################################################
dir_imgs = 'C:/Users/talha/Desktop/malaria paper/processed/LCM/400x/test/images/'
dir_gts = 'C:/Users/talha/Desktop/malaria paper/processed/LCM/400x/test/xmls/'
dir_dets = 'C:/Users/talha/Desktop/malaria paper/processedv2/LCM/400x/test/preds/'


# Get annotations (ground truth and detections)
gt_bbs = converter.vocpascal2bb(dir_gts)
det_bbs = converter.text2bb(dir_dets, bb_type=BBType.DETECTED, bb_format=BBFormat.XYX2Y2,type_coordinates=CoordinatesType.ABSOLUTE, img_dir=dir_imgs)


#############################################################
# EVALUATE WITH COCO METRICS
#############################################################
coco_res1 = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)
#############################################################
# EVALUATE WITH VOC PASCAL METRICS
#############################################################
iou = 0.5
dict_res = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs, det_bbs, iou, generate_table=True, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
