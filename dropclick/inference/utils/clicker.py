# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import torchvision
import torch
import numpy as np
import cv2
import os
from dropclick.utils.misc import color_map
from pathlib import Path
import json
from pycocotools import mask as mask_utils
from skimage import measure

from scipy.optimize import linear_sum_assignment
import random
import copy


color_map = np.array([
    [220,  20,  60],   # crimson
    [173, 255,  47],   # greenyellow
    [30,  144, 255],   # dodgerblue
    [255, 127,  80],   # coral
    [148,   0, 211],   # darkviolet
    [50,  205,  50],   # limegreen
    [255, 255,   0],   # yellow
    [255, 105, 180],   # hotpink
    [0,   191, 255],   # deepskyblue
    [199,  21, 133],   # mediumvioletred
    [0,   250, 154],   # mediumspringgreen
    [255, 215,   0],   # gold
    [0,     0, 205],   # mediumblue
    [255, 165,   0],   # orange
    #
    [192, 192, 192],   # silver
    [64,  224, 208],   # turquoise
    [123, 104, 238],   # mediumslateblue
    [210, 180, 140],   # tan
    [250, 128, 114],   # salmon
    [152, 251, 152],   # palegreen
    [244, 164,  96],   # sandybrown
    [127, 255, 212],   # aquamarine
    [219, 112, 147],   # palevioletred
    [60, 179, 113],    # mediumseagreen
    [255, 140,   0],   # darkorange
    [100, 149, 237],   # cornflowerblue
    # matplotlib "tab10" palette
    [31, 119, 180],    # tab:blue
    [255,127,  14],    # tab:orange
    [44, 160,  44],    # tab:green
    [214, 39,  40],    # tab:red
    [148,103,189],     # tab:purple
    #### [140, 86,  75],    # tab:brown
    [227,119,194],     # tab:pink
    [127,127,127],     # tab:gray
    [188,189, 34],     # tab:olive
    [23, 190, 207],    # tab:cyan
    ] * 10 )


class ClickerOneclick:

    def __init__(self, inputs, matching_threshold_iou=.4):
        
        self.inputs = inputs
        
        self.num_insts = []
        self.num_clicks_per_object = []
        self.fg_coords = []
        self.fg_orig_coords = []
        self.pred_masks = None
        self.missing_clicks_idcs = []
        self._set_gt_info()
        
        self.pred_masks_clicked = None
        self.pred_masks_learnt = None
        self.idcs_kept_preds = []
        
        self.matching_threshold_iou = matching_threshold_iou
        self.ioustr = str(int(self.matching_threshold_iou*100))
    
    def _set_gt_info(self):

        self.gt_masks = self.inputs[0]['instances'].gt_masks.to('cpu')
        self.gt_click_missing = self.inputs[0]['instances'].gt_click_missing
        self.gt_masks_clicked = self.gt_masks[[True if not mc else False for mc in self.gt_click_missing]]
        self.gt_masks_learnt = self.gt_masks[self.gt_click_missing]
        ## actually missing clicks
        self.num_clicks = sum( [ True if not cm else False for cm in self.gt_click_missing ] )
        
        self.num_instances, self.orig_h, self.orig_w = self.gt_masks.shape[:]
        self.num_instances_clicked = self.gt_masks_clicked.shape[0]
        self.num_instances_learnt = self.gt_masks_learnt.shape[0]

        if 'num_clicks_per_object' in self.inputs[0]:
            for x in self.inputs:
                self.num_clicks_per_object.append(x['num_clicks_per_object'])
                self.fg_coords.append(x['fg_click_coords'])
                self.missing_clicks_idcs.append(x['missing_clicks_idcs'])
                self.num_insts.append(len(x['num_clicks_per_object']))

        self.trans_h, self.trans_w = self.inputs[0]['image'].shape[-2:]
        
        self.ratio_h = self.trans_h/self.orig_h
        self.ratio_w = self.trans_w/self.orig_w

        self.fg_orig_coords = self.inputs[0]['orig_fg_click_coords']


    def compute_iou(self):

        ious = []
        for gt_mask, pred_mask in zip(self.gt_masks, self.pred_masks):
            intersection = (gt_mask * pred_mask).sum()
            union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
            ious.append(intersection/union)
        return ious


    def compute_iou_clicked(self):

        ious = []
        for gt_mask, pred_mask in zip(self.gt_masks_clicked, self.pred_masks_clicked):
            intersection = (gt_mask * pred_mask).sum()
            union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
            ious.append(intersection/union)
        return ious


    def compute_iou_learnt_and_get_next_click(self, use_click_scheme, save_preds_info=None):
        ious = []
        fn_gt_idcs = []
        pred_masks = self.pred_masks_learnt
        gt_masks = self.gt_masks_learnt
        pad_size = gt_masks.shape[0] - pred_masks.shape[0]
        n_preds_before_pad = pred_masks.shape[0]
        if pad_size > 0:
            pred_masks = torch.cat((pred_masks, torch.zeros((pad_size,pred_masks.shape[1],pred_masks.shape[2]))))  # padding empty pred to ensure a match for all gts
        ## iou based hungarian matching
        cost_matrix = np.zeros((pred_masks.shape[0], gt_masks.shape[0]))
        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                intersection = (gt_mask * pred_mask).sum()
                union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
                iou = intersection / union if union > 0 else 0
                cost_matrix[i, j] = 1 - iou  # convert iou to cost
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        matching_idcs = list(zip(pred_indices, gt_indices))
        matching_idcs_dict = dict(zip(pred_indices, gt_indices))
        ## get ious of matches
        for i_pred, i_gt in matching_idcs:
            gt_mask = gt_masks[i_gt]
            pred_mask = pred_masks[i_pred]
            intersection = (gt_mask * pred_mask).sum()
            union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
            iou = intersection/union
            ious.append(iou)
            if iou<self.matching_threshold_iou:
                fn_gt_idcs.append(i_gt)
        ## logging fp/fn
        clicker_metrics = {}
        clicker_metrics[f'n_tp{self.ioustr}'] = len([x for x in ious if x>=self.matching_threshold_iou])
        clicker_metrics[f'n_fn{self.ioustr}'] = len([x for x in ious if x<self.matching_threshold_iou])  # this automatically counts padded ones as fns if they're matched (intended)
        clicker_metrics[f'n_fp{self.ioustr}'] = clicker_metrics[f'n_fn{self.ioustr}'] - pad_size  # matched fns naturally are fps (1 pad => 1 less FP). pad_size needs to be considered (positive = not real pred, negative = additional FP)
        ## storing which preds to save (also needed elsewhere for vis)
        self.idcs_kept_preds = [i_pred for iou, (i_pred, _) in zip(ious, matching_idcs) if iou>=self.matching_threshold_iou]  # iou>=self.matching_threshold_iou removes such FPs that have been included through matching algo. Other FPs excluded there already. All counted.
        if save_preds_info:
            preds_to_save = torch.cat((self.pred_masks_clicked, pred_masks[self.idcs_kept_preds]))
            save_masks_to_coco(
                predictions=[{'masks': preds_to_save, 'labels': torch.ones(preds_to_save.shape[0])}],
                output_path=os.path.join(
                    save_preds_info["vis_path"],
                    str(save_preds_info["image_id"]),
                    f'{save_preds_info["image_id"]}_missing{int(save_preds_info["missing_click_ratio"] * 100):03d}_nmiss{save_preds_info["n_missing"]}_nclk{save_preds_info["n_clicks"]}_nfp{self.ioustr}-{clicker_metrics[f"n_fp{self.ioustr}"]}_nfn{self.ioustr}-{clicker_metrics[f"n_fn{self.ioustr}"]}.json'),
                image_ids=[save_preds_info['image_id']],
                file_names=[save_preds_info['file_name']],
                )
            
        if len(ious) and use_click_scheme:
            ## first do missing ones
            if min(ious)<self.matching_threshold_iou:
                next_click = sorted( [ (iou, i_gt) for iou, (_, i_gt) in zip(ious, matching_idcs) ], key=lambda x: x[0] )[0][1]
            ## then find most most overlapping (cluttered) ones
            else:
                overlaps = {}
                for i, gt_mask in enumerate(gt_masks):
                    overlaps[i] = []
                    for j, pred_mask in enumerate(pred_masks):
                        if j in matching_idcs_dict and matching_idcs_dict[j] == i:
                            continue
                        intersection = (gt_mask * pred_mask).sum()
                        union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
                        iou = intersection / union if union > 0 else 0
                        overlaps[i].append(iou)
                clutter = [ (len([ x for x in overlaps[i_gt] if x>0 ]), sum(overlaps[i_gt]), i_gt) for i_gt in overlaps ]  # (n_clutter, clutter_iou, i_gt)
                clutter = [ x for x in clutter if x[0]>0 ]
                if len(clutter):
                    next_click = sorted(clutter, key=lambda x: (x[1], x[0]))[-1][2]  # sort by clutter_iou then n_clutter, take highest
                else:
                    next_click = random.randint(0,gt_masks.shape[0]-1)
        else:
            next_click = None
        for k,v in clicker_metrics.items():
            clicker_metrics[k] = torch.tensor(v)
        return [ iou if iou>=self.matching_threshold_iou else torch.tensor(0) for iou in ious ], clicker_metrics, next_click, fn_gt_idcs
    

    def save_visualization(self, save_results_path, ious=None, num_interactions=None, alpha_blend =0.6, click_radius=None, fn_gt_idcs=None):

        if click_radius is None:
            click_radius = 15   
            click_radius = int(round( click_radius * (self.orig_w / 640) ))
            
        if ( num_interactions==0 and ious==[0] ):
            print( 'WARNING: evaluator called clicker with num_interactions==0 and ious==[0] as in deprecated versions. If this happens a lot, sth is wrong.' )
        if num_interactions==0 and not len(ious):
            is_gt = True
            result_masks_for_vis = self.gt_masks
        else:
            is_gt = False
            result_masks_for_vis = self.pred_masks

        image = np.asarray(self.inputs[0]['image'].permute(1,2,0))
        image = cv2.resize(image, (self.orig_w, self.orig_h))
        image_gt = copy.deepcopy(image)
        image_1col = copy.deepcopy(image)
        image_3col = copy.deepcopy(image)
        image_3col_clicked = copy.deepcopy(image)
        image_clicksonly = copy.deepcopy(image)
        image_clicksonly_given = copy.deepcopy(image)
        image_clicksonlyalpha = copy.deepcopy(image)
        image_clicksonlyalpha = self.apply_mask(image_clicksonlyalpha, np.ones_like(image[:,:,0]), np.array([1,1,1]), .5)

        result_masks_for_vis = result_masks_for_vis.to(device ='cpu')
    
        pred_masks =np.asarray(result_masks_for_vis,dtype=np.uint8)
        c = []
        for i in range(pred_masks.shape[0]):
            if i>=79:
                i -= 79
            c.append(color_map[i]/255.0)
        bg_mask = np.ones((image.shape[0],image.shape[1]))
        i_preds_sorted_by_type = {}
        for i in range(pred_masks.shape[0]):  # gt always gets 'fp' here but we don't use that anyways.
            if i<self.num_clicks:
                i_preds_sorted_by_type[i] = 'clicked'
            elif ( i>=self.num_clicks and i-self.num_clicks in self.idcs_kept_preds ):
                i_preds_sorted_by_type[i] = 'tp'
            else:
                i_preds_sorted_by_type[i] = 'fp'
        order =  [ 'tp',  'clicked',  'fp' ]
        colors = [(.9,.5,0), (0,.9,0), (.9,0,.5)]  # orange, green, raspberry
        i_preds_sorted_by_type = dict(sorted(i_preds_sorted_by_type.items(), key=lambda item: order.index(item[1]) ))

        for i, type_ in i_preds_sorted_by_type.items():
            bg_mask[pred_masks[i]==1] = 0
            image = self.apply_mask(image, pred_masks[i], c[i], alpha_blend)
            if is_gt:
                image_gt = self.apply_mask(image_gt, pred_masks[i], c[i], 1)
            else:
                color = colors[order.index(type_)]
                color = tuple( max(0, min(0.9, v*random.uniform(0.8, 1.2))) for v in color )  # randomize existing color a bit to have different shades
                image_3col = self.apply_mask(image_3col, pred_masks[i],
                                            color,
                                            alpha_blend)
                color_1col = colors[order.index('tp')]
                color_1col = tuple( max(0, min(0.9, v*random.uniform(0.8, 1.2))) for v in color_1col )  # randomize existing color a bit to have different shades
                image_1col = self.apply_mask(image_1col, pred_masks[i],
                                            color_1col,
                                            alpha_blend)
                contours, _ = cv2.findContours(pred_masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for j, cnt in enumerate(contours):
                    cv2.drawContours(image_3col, [cnt], -1, tuple(int(c*255) for c in color), 2)
                for j, cnt in enumerate(contours):
                    cv2.drawContours(image_1col, [cnt], -1, tuple(int(c*255) for c in color_1col), 2)

        if fn_gt_idcs is not None:
            img_4col = copy.deepcopy(image_3col)
            for fn in fn_gt_idcs:
                fn_mask = self.gt_masks[fn].to(device='cpu').numpy().astype(np.uint8)
                color_fn = tuple( max(0, min(0.9, v*random.uniform(0.8, 1.2))) for v in (0.1176, 0.5647, 1.0) )  # randomize existing color a bit to have different shades
                img_4col = self.apply_mask(img_4col, fn_mask,
                                            color_fn,
                                            alpha_blend)
                contours, _ = cv2.findContours(fn_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for j, cnt in enumerate(contours):
                    cv2.drawContours(img_4col, [cnt], -1, tuple(int(c*255) for c in color_fn), 2)

        image_1col_clicked = copy.deepcopy(image_1col)
        if fn_gt_idcs is not None:
            img_4col_clicked = copy.deepcopy(img_4col)

        image = self.apply_mask(image, bg_mask, np.array([1,1,1]), .5)
        if is_gt:
            image_gt = self.apply_mask(image_gt, bg_mask, np.array([0,0,0]), 1)
        
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255

        if len(self.fg_orig_coords) and len(ious):
            missed = []
            clicked = []
            for i, c in enumerate(self.fg_orig_coords):
                if i in self.missing_clicks_idcs[0]:
                    missed.append(c)
                else:
                    clicked.append(c)
            for j, fg_coords_per_mask in enumerate(missed+clicked):
                assert len(fg_coords_per_mask)==1  # supports oneclick only
                for i, coords in enumerate(fg_coords_per_mask):
                    if j < len(missed):
                        color = (255,125,0)
                    else:
                        color = (0,255,0) 
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
                    image_clicksonly = cv2.circle(image_clicksonly, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
                    image_clicksonlyalpha = cv2.circle(image_clicksonlyalpha, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
                    if not j < len(missed):
                        image_clicksonly_given = cv2.circle(image_clicksonly_given, (int(coords[1]), int(coords[0])), click_radius, tuple((0,255,0) ), -1)
                        image_1col_clicked = cv2.circle(image_1col_clicked, (int(coords[1]), int(coords[0])), click_radius, tuple((0,255,0) ), -1)
                        if fn_gt_idcs is not None:
                            img_4col_clicked = cv2.circle(img_4col_clicked, (int(coords[1]), int(coords[0])), click_radius, tuple((0,255,0) ), -1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_dir = os.path.join(save_results_path, str(self.inputs[0]['image_id']))
        os.makedirs(save_dir, exist_ok=True)
        try:
            iou_val = np.round(sum(ious)/len(ious),4)*100
        except:
            iou_val = 'nan'
        cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}.png"), image)
        if is_gt:
            image_gt = cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"gt_only.png"), image_gt)
        else:
            image_3col = cv2.cvtColor(image_3col, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_preds_only_3col.png"), image_3col)
            image_1col = cv2.cvtColor(image_1col, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_preds_only_1col.png"), image_1col)
            image_1col_clicked = cv2.cvtColor(image_1col_clicked, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_preds_only_1col_clicked.png"), image_1col_clicked)
            if fn_gt_idcs is not None:
                img_4col = cv2.cvtColor(img_4col, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_preds_only_4col.png"), img_4col)
                img_4col_clicked = cv2.cvtColor(img_4col_clicked, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_preds_only_4col_clicked.png"), img_4col_clicked)
            image_clicksonly = cv2.cvtColor(image_clicksonly, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_clicksonly.png"), image_clicksonly)
            image_clicksonly_given = cv2.cvtColor(image_clicksonly_given, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_clicksonly_given.png"), image_clicksonly_given)
            image_clicksonlyalpha = cv2.cvtColor(image_clicksonlyalpha, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}_clicksonlyalpha.png"), image_clicksonlyalpha)
    
    def apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image

    def set_pred_masks(self, pred_masks, pred_scores):
        self.pred_masks = pred_masks
        self.pred_scores = pred_scores
        self.pred_masks_clicked = pred_masks[:self.num_clicks,:,:]
        self.pred_scores_clicked = pred_scores[:self.num_clicks]  # is already always 1.
        self.pred_masks_learnt = pred_masks[self.num_clicks:,:,:]
        self.pred_scores_learnt = pred_scores[self.num_clicks:]


def mask_to_polygons(binary_mask):
    """
    Convert a binary mask to COCO-style polygon representation.
    
    Args:
        binary_mask (numpy.ndarray): A binary mask (1s and 0s).

    Returns:
        list: A list of polygon points (COCO format).
    """
    polygons = []
    contours = measure.find_contours(binary_mask, 0.5)

    for contour in contours:
        contour = np.flip(contour, axis=1)
        polygon = contour.ravel().tolist()
        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons

def mask_to_bbox(binary_mask):
    """
    Compute the bounding box [x, y, width, height] from a binary mask.

    Args:
        binary_mask (numpy.ndarray): A binary mask.

    Returns:
        list: Bounding box [x, y, w, h].
    """
    pos = np.where(binary_mask > 0)
    if pos[0].size == 0:
        return [0, 0, 0, 0]  # Empty mask case

    x_min, y_min = np.min(pos[1]), np.min(pos[0])
    x_max, y_max = np.max(pos[1]), np.max(pos[0])

    return [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, width, height]

def save_masks_to_coco(predictions, output_path, image_ids, file_names):
    """
    Save segmentation masks in COCO format.

    Args:
        predictions (list of dict): Each dict should contain:
            - 'masks' (Tensor[N, H, W]): Binary masks
            - 'labels' (Tensor[N]): Category IDs
        output_path (str): Path to save the JSON file
        image_ids (list of int): List of image IDs corresponding to each prediction
    """
    
    coco_results = {
        'images':[],
        'annotations':[],
        'categories':[],
        }
    
    for i, pred in enumerate(predictions):
        image_id = image_ids[i]
        file_name = file_names[i]
        masks = pred["masks"].cpu().numpy().astype(np.uint8)
        labels = pred["labels"].tolist()

        for j in range(len(masks)):
            binary_mask = masks[j]
            segmentation = mask_to_polygons(binary_mask)
            if not segmentation:
                continue  # Skip empty masks

            bbox = mask_to_bbox(binary_mask)
            area = int(binary_mask.sum())  # Count nonzero pixels

            if not any( img['id']==image_id for img in coco_results['images'] ):
                coco_results['images'].append({
                    "id": image_id,
                    "file_name": file_name,
                    # "category_ids": [],
                    # "annotated": False,
                    # "annotating": [],
                    # "num_annotations": 0,
                    # "metadata": {},
                    # "deleted": False,
                    # "milliseconds": 0,
                    # "events": [],
                    # "regenerate_thumbnail": False
                })
            if not any( cat['id']==int(labels[j]) for cat in coco_results['categories'] ):
                coco_results['categories'].append({
                    "id": int(labels[j]),
                    "name": "object",
                    "supercategory": "",
                    "color": "#f90660",
                    "metadata": {},
                    "keypoint_colors": []
                })
            coco_results['annotations'].append({
                "image_id": image_id,
                "category_id": int(labels[j]),
                "segmentation": segmentation,
                "area": int(area),
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0  # Assume no crowd annotations
            })

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_results, f)

