# Modified by Amit Rana from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import copy
import logging

import numpy as np
import torch
import torchvision
import pycocotools.mask as mask_util
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode
from detectron2.structures.masks import PolygonMasks

from dropclick.data.dataset_mappers.utils import convert_coco_poly_to_mask, build_transform_gen
from dropclick.inference.utils.eval_utils import get_gt_clicks_coords_eval_oneclick

import random


__all__ = ["EvaluationDatasetMapperOneclick"]

# This is specifically designed for the COCO dataset.
class EvaluationDatasetMapperOneclick:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the required format.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    5. Prepare a list of foreground clicks (one click per object) for all the objects in the image
    """

    @configurable
    def __init__(
        self,
        is_train=False,
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[EvaluationDatasetMapperOneclick] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.dataset_name = dataset_name
    
    @classmethod
    def from_config(cls, cfg, is_train=False, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        orig_image_shape = image.shape[:2]
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            if "CKA_sugar_beet" not in dataset_dict["file_name"]:  # we use keypoint annos in sb20 (stem locations)
                for anno in dataset_dict["annotations"]:
                    anno.pop("keypoints", None)

            annos = [
                original_res_annotations(obj, orig_image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            keypoints = None
            if len(annos) and 'keypoints' in annos[0]:
                keypoints = []
                for a in annos:
                    keypoints.append([a['keypoints']])
                    del a['keypoints']
            # USER: Implement additional transformations if you have other types of data

            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            if self.dataset_name == "coco_2017_val" or "CKA_" in self.dataset_name:
                instances = utils.annotations_to_instances(annos, orig_image_shape)
            else:
                instances = utils.annotations_to_instances(annos, orig_image_shape,  mask_format="bitmask")
           
            if not hasattr(instances, 'gt_masks'):
                return None
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            
            if keypoints is not None:  # do the keypoint/click scheme as in panoptic one-click
                keypoints_ = []
                for kps_ins_ in keypoints:  # loop over the instances
                    kps_ins = [ list(kp) for kp in kps_ins_ ]
                    kps_ins = [ [ int(round(kp[0])),int(round(kp[1])),int(kp[2]) ] for kp in kps_ins ]
                    if len(kps_ins)==0:
                        keypoints_.append( [0,0,0] )
                        continue
                    keypoints_.append( kps_ins[0] )
                keypoints_= torch.as_tensor(keypoints_)
                instances.set('gt_keypoints', keypoints_)
            
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            
            if len(instances) == 0:
                return None
            # Generate masks from polygon
            h, w = instances.image_size
        
            if hasattr(instances, 'gt_masks'):
                if self.dataset_name == "coco_2017_val":
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = instances.gt_masks.tensor

                ## Here we handle occlusions, e.g. small objects in front of big ones.
                cnt = 1
                imap = torch.zeros_like( gt_masks[0], dtype=torch.uint8 )
                new_keypoints = []
                for curr_mask, keypoint in zip(gt_masks, instances.gt_keypoints if hasattr(instances, 'gt_keypoints') else [None for _ in range(gt_masks.shape[0])]):
                    overlap_is, counts = torch.unique(imap[curr_mask>0], return_counts=True)
                    overlap_is, counts = list(overlap_is), list(counts)
                    if 0 in overlap_is:  # skip background
                        counts.pop(overlap_is.index(0))
                        overlap_is = [ x for x in overlap_is if x!=0 ]
                    if len(overlap_is):
                        for oi, c in zip(overlap_is, counts):
                            size_current = curr_mask.sum()
                            size_existing = (imap==oi).sum()
                            if torch.equal( curr_mask, (imap==oi) ):
                                curr_mask = torch.zeros_like(curr_mask)  # delete current mask and move on
                                skipped_deliberately = True  # adding a safety check
                                continue
                            if size_existing <= size_current:
                                curr_mask[imap==oi] = 0  # giving priority to smaller objects by setting the new overlaping mask to 0
                    if curr_mask.sum():  # only add/append/count if anything is left (or wasn't there to begin with)
                        imap[curr_mask>0] = cnt
                        new_keypoints.append( keypoint )
                        cnt += 1
                    else:
                        assert skipped_deliberately
                assert torch.unique(imap).max()+1 == torch.unique(imap).shape[0], f'{torch.unique(imap).max()+1} vs. {torch.unique(imap).shape[0]}'
                new_gt_masks = []
                for id_ in torch.unique(imap):
                    if id_==0:  # skip background
                        continue
                    new_gt_masks.append( (imap==id_).float() )
                new_gt_masks = torch.stack(new_gt_masks, dim=0)
                
                all_masks = torch.zeros((new_gt_masks.shape[-2:]), dtype=torch.int16)
                for _id, m in enumerate(new_gt_masks):
                    all_masks = torch.logical_or(all_masks, m)

                new_gt_classes = [0]*new_gt_masks.shape[0]
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                new_instances = Instances(image_size=image_shape)
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
               
                if 'ignore_mask' in dataset_dict:
                    raise NotImplementedError
                    
                if hasattr(instances, 'gt_keypoints'):
                    (num_clicks_per_object, fg_coords_list, orig_fg_coords_list) = get_gt_clicks_coords_eval_oneclick(new_gt_masks, image_shape, keypoints=new_keypoints )
                else:
                    (num_clicks_per_object, fg_coords_list, orig_fg_coords_list) = get_gt_clicks_coords_eval_oneclick(new_gt_masks, image_shape)

                dataset_dict["orig_fg_click_coords"] = orig_fg_coords_list
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["num_clicks_per_object"] = num_clicks_per_object
                assert len(num_clicks_per_object) == new_instances.gt_masks.shape[0]
                dataset_dict["bg_mask"] = torch.where(all_masks!=1,1,0).to(dtype = torch.uint8)
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict


def original_res_annotations(
    annotation, image_size
):
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in polygons
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask

    return annotation