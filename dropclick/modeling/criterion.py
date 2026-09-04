# Modified by Amit Rana from: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import copy


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetFinalCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    V1,V2
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 num_static_fg_queries,
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        assert num_classes==1, 'multi-class not implemented'
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        self.num_static_fg_queries = num_static_fg_queries


    def loss_labels_learnt(
        self,
        outputs_learnt,
        targets_learnt,
        outputs_clicked,
        targets_clicked,
        num_masks,
        indices,
        num_clicks_per_object,
        ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        ## labels for learnt fg queries
        assert outputs_clicked is None and targets_clicked is None and num_masks is None and num_clicks_per_object is None
        assert "pred_logits" in outputs_learnt
        src_logits_learnt = outputs_learnt["pred_logits"].float()  ## b, q, n_classes+1

        idx = self._get_src_permutation_idx(indices)
        target_classes_learnt_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_learnt, indices)])
        target_classes_learnt = torch.full(
            src_logits_learnt.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits_learnt.device
        )
        target_classes_learnt[idx] = target_classes_learnt_o  ## b, q;   max(labels)+1 (==n_classes) where no object 
        
        loss_ce_learnt = F.cross_entropy(src_logits_learnt.transpose(1, 2), target_classes_learnt, self.empty_weight)

        losses = {"loss_ce_learnt": loss_ce_learnt}

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def loss_mask_clicked(self, outputs_learnt, targets_learnt, outputs_clicked, targets_clicked, num_masks, indices, num_clicks_per_object):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert outputs_learnt is None and targets_learnt is None and indices is None
        assert "pred_masks" in outputs_clicked

        ## clicked
        # Accumulate mask for each object (as there might be multiple clicks per object) and background
        new_outputs = []
        for i in range(outputs_clicked['pred_masks'].shape[0]):
            temp_out = []
            clicks_per_image = copy.deepcopy(num_clicks_per_object[i])
            clicks_per_image.append(outputs_clicked['pred_masks'][i].shape[0] - sum(num_clicks_per_object[i]))
            split_masks = torch.split(outputs_clicked['pred_masks'][i], clicks_per_image, dim=0)
            for m in split_masks:
                temp_out.append(torch.max(m, dim=0).values)
            new_outputs.append(torch.stack(temp_out))
        src_masks = torch.cat(new_outputs,dim=0)

        masks_clicked = [t["masks"] for t in targets_clicked]
        target_masks = torch.cat(masks_clicked,dim=0).to(dtype=torch.float32)
        
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_clicked": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice_clicked": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses


    def loss_mask_learnt(self, outputs_learnt, targets_learnt, outputs_clicked, targets_clicked, num_masks, indices, num_clicks_per_object):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert outputs_clicked is None and targets_clicked is None and num_clicks_per_object is None
        assert "pred_masks" in outputs_learnt

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs_learnt["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets_learnt]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_learnt": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice_learnt": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses


    def get_loss(
        self,
        loss,
        outputs_learnt,
        targets_learnt,
        outputs_clicked,
        targets_clicked,
        num_masks,
        indices,
        num_clicks_per_object,
        ):
        loss_map = {
            'labels_learnt': self.loss_labels_learnt,
            'masks_clicked': self.loss_mask_clicked,
            'masks_learnt': self.loss_mask_learnt,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs_learnt, targets_learnt, outputs_clicked, targets_clicked, num_masks, indices, num_clicks_per_object)

    def forward(self, outputs, targets, num_clicks_per_object = None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        #### Dynamite is appending bg target, keep this initially so to not have to mess with indices later
        added_bg = True
        for i,t in enumerate(targets):
            
            target_bg_mask = t['bg_mask']
            targets[i]["masks"] = torch.cat((t["masks"], target_bg_mask.unsqueeze(0)), dim=0)

        #### split outputs and targets
        num_clicks_per_object = [ [n for n,is_missing in zip(nc,t['click_missing']) if not is_missing] for t,nc in zip(targets, num_clicks_per_object) ]
        outputs_learnt = filter_outputs( outputs, 'learnt', self.num_static_fg_queries )  # learnt fg queries only
        outputs_clicked = filter_outputs( outputs, 'clicked', self.num_static_fg_queries )  # those include learnt bg queries
        targets_learnt = filter_targets( targets, 'learnt', added_bg=added_bg )
        targets_clicked = filter_targets( targets, 'clicked', added_bg=added_bg )
        
        #### non-clicked objects
        outputs_without_aux = {k: v for k, v in outputs_learnt.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets_learnt)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks_clicked = sum(len(t["labels"]) for t in targets_clicked)
        num_masks_clicked = torch.as_tensor(
            [num_masks_clicked], dtype=torch.float, device=next(iter(outputs_clicked.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_clicked)
        num_masks_clicked = torch.clamp(num_masks_clicked / get_world_size(), min=1).item()

        num_masks_learnt = sum(len(t["labels"]) for t in targets_learnt)
        num_masks_learnt = torch.as_tensor(
            [num_masks_learnt], dtype=torch.float, device=next(iter(outputs_learnt.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_learnt)
        num_masks_learnt = torch.clamp(num_masks_learnt / get_world_size(), min=1).item()


        # Compute all the requested losses
        losses = {}
        losses.update( self.get_loss('labels_learnt', outputs_learnt, targets_learnt, None, None, None, indices, None) )
        losses.update( self.get_loss('masks_clicked', None, None, outputs_clicked, targets_clicked, num_masks_clicked, None, num_clicks_per_object) )
        losses.update( self.get_loss('masks_learnt', outputs_learnt, targets_learnt, None, None, num_masks_learnt, indices, None) )
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, (aux_outputs_learnt, aux_outputs_clicked) in enumerate(zip(outputs_learnt["aux_outputs"],outputs_clicked["aux_outputs"])):
                indices = self.matcher(aux_outputs_learnt, targets_learnt)
                l_dict = {}
                l_dict.update( self.get_loss('labels_learnt', aux_outputs_learnt, targets_learnt, None, None, None, indices, None) )
                l_dict.update( self.get_loss('masks_clicked', None, None, aux_outputs_clicked, targets_clicked, num_masks_clicked, None, num_clicks_per_object) )
                l_dict.update( self.get_loss('masks_learnt', aux_outputs_learnt, targets_learnt, None, None, num_masks_learnt, indices, None) )
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def filter_outputs( outputs, mode, num_static_fg_queries ):
    if mode=='learnt':
        start = None
        end = num_static_fg_queries
    elif mode=='clicked':
        start = num_static_fg_queries
        end = None
    outputs_filtered = {}  # learnt fg queries only
    outputs_filtered['pred_masks'] = outputs['pred_masks'][:,start:end,:,:]
    outputs_filtered['pred_logits'] = outputs['pred_logits'][:,start:end,:]
    outputs_filtered['aux_outputs'] = []
    for i,aux_output in enumerate(outputs['aux_outputs']):
        aux_output_filtered = {}
        aux_output_filtered['pred_masks'] = aux_output['pred_masks'][:,start:end,:,:]
        aux_output_filtered['pred_logits'] = aux_output['pred_logits'][:,start:end,:]
        outputs_filtered['aux_outputs'].append(aux_output_filtered)
    return outputs_filtered

def filter_targets( targets, mode, added_bg=True ):
    targets_filtered = []
    for i,ts_img in enumerate(targets):
        ts_img_filtered = {}
        if mode=='learnt':
            mask = copy.deepcopy(ts_img['click_missing'])
            if added_bg:
                mask.append(False)  # excluding the previously appended bg mask
        elif mode=='clicked':
            mask = copy.deepcopy([ not x for x in ts_img['click_missing'] ])
            if added_bg:
                mask.append(True)  # including bg mask
        for k,v in ts_img.items():
            if k=='click_missing':
                continue
            elif isinstance(v,list):
                ts_img_filtered[k] = [ t for j,t in enumerate(v) if mask[j] ]  # labels
            elif isinstance(v,torch.Tensor):
                if len(v.size())==3:  # masks
                    ts_img_filtered[k] = v[mask]
                elif len(v.size())==2:  # bg_mask, padding_mask
                    ts_img_filtered[k] = v
                elif len(v.size())==1:  # labels
                    ts_img_filtered[k] = v[mask[:-1]] if added_bg else v[mask]
        targets_filtered.append( ts_img_filtered )
    return targets_filtered

