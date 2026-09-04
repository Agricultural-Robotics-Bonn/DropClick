# Modified by Amit Rana from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import copy
import logging
import pickle
import numpy as np
import torch
import random
from functools import lru_cache
import cv2
from copy import deepcopy
from detectron2.structures import BoxMode, Boxes
from fvcore.common.timer import Timer
from pycocotools import coco
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from dropclick.data.dataset_mappers.utils import get_clicks_coords, build_transform_gen


__all__ = ["COCOLVISDatasetMapperOneclick"]


class COCOLVISDatasetMapperOneclick:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the required format.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    5. Generate a boolean list to randomly select some objects as "missing clicks"
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        min_area,
    ):
        """

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image
            min_ares: minimum mask area for an object/instance
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOLVISDatasetMapperOneclick] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_area = min_area
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "min_area": cfg.INPUT.MIN_AREA_FOR_MASK,
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

            # USER: Implement additional transformations if you have other types of data
            keypoints = None
            if "CKA_" in dataset_dict["file_name"]:
                annos = [
                    transform_instance_annotations(obj, transforms, image.shape[:2])
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                if '_sugar_beet' in dataset_dict["file_name"]:  # dataset uses keypoints for stem location clicks
                    keypoints = []
                    for a in annos:
                        keypoints.append(a['keypoints'])
                        del a['keypoints']
                else:
                    for a in annos:
                        del a['keypoints']
            else:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if (obj.get("iscrowd", 0) == 0 and obj.get("isThing") and obj.get("area",0) > self.min_area)
                ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,  mask_format="bitmask")
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
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
                
                gt_masks = instances.gt_masks.tensor.to(dtype=torch.uint8)

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
                unique_ids = torch.unique(imap)
                expected_ids = torch.arange(0, cnt, dtype=unique_ids.dtype, device=unique_ids.device)
                if 0 not in unique_ids:
                    # background fully covered by foreground masks
                    expected_ids = torch.arange(1, cnt, dtype=unique_ids.dtype, device=unique_ids.device)
                assert torch.equal(torch.sort(unique_ids)[0], expected_ids), \
                    f'{unique_ids} vs expected {expected_ids}'

                new_gt_masks = []
                for id_ in torch.unique(imap):
                    if id_==0:  # skip background
                        continue
                    new_gt_masks.append( (imap==id_).float() )
                new_gt_masks = torch.stack(new_gt_masks, dim=0)

                all_masks = dataset_dict["padding_mask"].int()
             
                new_instances = Instances(image_size=image_shape)

                new_gt_classes = [0]*new_gt_masks.shape[0]
                new_gt_classes = torch.tensor(new_gt_classes, dtype=torch.int64)
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
               
                for _id, m in enumerate(new_gt_masks):
                    all_masks = torch.logical_or(all_masks, m)
                
                if hasattr(instances, 'gt_keypoints'):
                    (num_clicks_per_object, fg_coords_list) = get_clicks_coords(new_gt_masks, all_masks=all_masks, keypoints=new_keypoints)
                else:
                    (num_clicks_per_object, fg_coords_list) = get_clicks_coords(new_gt_masks, all_masks=all_masks)

                ## select the missing clicks
                n = len(fg_coords_list)
                if random.random() < .5:
                    if random.random() < .5:
                        n_missing = 0
                    else:
                        n_missing = n
                else:
                    assert n > 0
                    min_ = 1 if n>1 else 0
                    max_ = n-1 if n>1 else 1
                    n_missing = random.randint(min_, max_)
                missing_clicks_idcs = random.sample( range( 0, n ), n_missing )
                
                dataset_dict["missing_clicks_idcs"] = missing_clicks_idcs
                new_instances.set('gt_click_missing', [ True if i in missing_clicks_idcs else False for i in range(new_gt_masks.shape[0]) ]) 
                
                dataset_dict["bg_mask"] = torch.logical_not(all_masks).to(dtype = torch.uint8)
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["num_clicks_per_object"] = num_clicks_per_object

                assert len(num_clicks_per_object) == new_instances.gt_masks.shape[0]
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict
    

##### BELOW IS FROM detectron2.data.detection_utils
### slightly modified as we use keypoints differently
### (we do not associate meanings to keypoints (e.g. left eye, right eye), hence we do not need to consider that horizontal flip)
### original meaning gets clear in https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/detectron2/data/datasets/builtin_meta.py#L157
def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size):
    """
    Transform keypoint annotations of an image.
    If a keypoint is transformed out of image boundary, it will be marked "unlabeled" (visibility=0)

    Args:
        keypoints (list[float]): Nx3 float in Detectron2's Dataset format.
            Each point is represented by (x, y, visibility).
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        # keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
        #     When `transforms` includes horizontal flip, will use the index
        #     mapping to flip keypoints.
    """
    # (N*3,) -> (N, 3)
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints_xy = transforms.apply_coords(keypoints[:, :2])

    # Set all out-of-boundary points to "unlabeled"
    inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)
    keypoints[:, :2] = keypoints_xy
    keypoints[:, 2][~inside] = 0

    # Maintain COCO convention that if visibility == 0 (unlabeled), then x, y = 0
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints

