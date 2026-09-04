# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import os
import sys
import csv

import numpy as np
import torch, torchvision
import cv2
import logging
import datetime
import pickle
from detectron2.utils.visualizer import Visualizer
import torchvision.transforms.functional as F

from collections import defaultdict
from dropclick.data.dataset_mappers.utils import getOneclickCoords


def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]

def get_gt_clicks_coords_eval_oneclick(masks, image_shape, max_num_points=1,
                             first_click_center=True, t= 0, keypoints=None):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    NOTE: ignoring timestamp as we do oneclick
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)

    I, H, W = masks.shape
    trans_h, trans_w = image_shape
    ratio_h = trans_h/H
    ratio_w = trans_w/W
    num_clicks_per_object = [0]*I
    orig_fg_coords_list = []
    fg_coords_list = []
    keypoints = [ None for _ in range(len(masks)) ] if keypoints is None else keypoints
    for i, (_m,_k) in enumerate(zip(masks, keypoints)):
        orig_coords = []
        coords = []
        if first_click_center:
            center_coords = getOneclickCoords( _m, is_eval=True, keypoint=_k[:2][[1,0]].numpy() if (_k is not None and _k.sum()) else None )
            orig_coords.append([center_coords[0], center_coords[1], t])
            coords.append([center_coords[0]*ratio_h, center_coords[1]*ratio_w, t])
            num_clicks_per_object[i]+=1
        orig_fg_coords_list.append(orig_coords) 
        fg_coords_list.append(coords)
       
    return num_clicks_per_object, fg_coords_list, orig_fg_coords_list


import json
def log_underoneclick(res, dataset_name, testmode,):
    logger = logging.getLogger(__name__)

    summary_stats = {}
    summary_stats["dataset"] = dataset_name

    for k,v in res.items():
        current_metric = {}
        for _d in v:
            current_metric.update(_d)
        summary_stats[k] = current_metric

    avgs = defaultdict(list)
    key_with_all_imgids = 'missing100_all_miou' if 'missing100_all_miou' in summary_stats else 'missing000_all_miou' 
    n_images = len(summary_stats[key_with_all_imgids])
    n_objects_avg = sum([ len(v[-1]) for v in summary_stats[key_with_all_imgids].values() ]) / n_images
    print(f'total number of images: {n_images}')
    print(f'average number of objects per image: {n_objects_avg}')
    nclicks_per_imgid = {}
    for _image_id in summary_stats[key_with_all_imgids].keys():
        for k,v in summary_stats.items():
            if any(k==x for x in ('dataset',)) or not any(k.startswith(x) for x in ('missing000', 'missing100')):
                continue
            if _image_id in v:
                assert len(v[_image_id])==1
                v_ = v[_image_id][-1]
                if isinstance(v_, list):
                    avgs[k].extend(v_)
                else:
                    avgs[k].append(v_)
    res_out = {}
    for k,v in avgs.items():
        v_ = [ x for x in v if x!=-1 ]
        avgs[k] = (sum(v_) / len(v_)).item() if len(v_) > 0 else -1
        if any(x==k for x in []) or '_map' in k or '_n_' in k or '_miou' in k:
            res_out[f'{testmode}_{k}'] = round(avgs[k],4)
        else:
            logger.info(f'did not save metric: {k}')
    
    ### thresholded by target ratios
    target_ratios = [int(n) for n in list(np.linspace(5,95,19))]
    data_per_ratio = defaultdict(list)
    for target_ratio in target_ratios:
        ## which logged metric to pick for per target ratio
        if target_ratio == 0:
            actual_ratio_per_id = { id_.split('_')[0]: 0 for id_ in summary_stats[key_with_all_imgids].keys() }            
        else:
            actual_ratio_per_id = {}
            for metric, v in summary_stats.items():
                if any(metric==x for x in ('dataset',)) or metric.startswith('missing000'):
                    continue
                missing_ratio = int(metric.split('missing')[1][:3])
                for id_ in v.keys():
                    id_ = id_.split('_')[0]
                    if id_ not in actual_ratio_per_id:
                        actual_ratio_per_id[id_] = missing_ratio
                    if ( missing_ratio > target_ratio and missing_ratio < actual_ratio_per_id[id_]):
                        actual_ratio_per_id[id_] = missing_ratio
        ## now get
        for id_, missing_ratio in actual_ratio_per_id.items():
            selected_metrics = [metric for metric in summary_stats.keys() if metric.startswith(f'missing{missing_ratio:03d}') ]
            for metric in selected_metrics:
                data = summary_stats[metric][id_]; assert len(data)==1; data = data[-1]
                metric_short = '_'.join(metric.split('_')[1:])
                if isinstance(data, list):
                    data_per_ratio[f'{testmode}_missingthresh{target_ratio:03d}_{metric_short}'].extend(data)
                else:
                    data_per_ratio[f'{testmode}_missingthresh{target_ratio:03d}_{metric_short}'].append(data)
        if f'{testmode}_missingthresh{target_ratio:03d}_n_objects' and f'{testmode}_missingthresh{target_ratio:03d}_n_missing' in data_per_ratio:
            n_objects = data_per_ratio[f'{testmode}_missingthresh{target_ratio:03d}_n_objects']
            n_missing = data_per_ratio[f'{testmode}_missingthresh{target_ratio:03d}_n_missing']
            actual_drop_ratio = [sum(n_missing)/sum(n_objects)]
            data_per_ratio[f'{testmode}_missingthresh{target_ratio:03d}_actual_drop_ratio'] = actual_drop_ratio 
    ## mean
    data_per_ratio = { k: (sum(v)/len(v)).item() if len(v)>0 else -1 for k,v in data_per_ratio.items() }
    res_out.update(data_per_ratio)

    out_str = ''
    for k,v in sorted(res_out.items()):
        out_str += f'\n{k} {v}'
    logger.info(out_str)
    return res_out
            

