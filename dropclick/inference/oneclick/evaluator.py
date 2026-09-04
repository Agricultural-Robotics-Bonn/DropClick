# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe

import csv
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from contextlib import ExitStack, contextmanager
import contextlib
import numpy as np
from collections import defaultdict
import torch
import torchvision

from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from ..utils.clicker import ClickerOneclick
from ..utils.predictor import PredictorOneclick
from pathlib import Path


def evaluate(
    model,
    data_loader,
    vis_path = None,
    missing_click_ratios=[],  # empty list will use clicker to drop clicks one by one
    ids_to_skip = [],
):
    """
    Run model on the data_loader and return a dict, later used to calculate
    all the metrics for semi-automated one-click segmentation
    (iou clicked/learnt and various fp/fn counts).
    The model will be used in eval mode.

    Arguments:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        vis_path: str
            Path to save visualization of masks with clicks during evaluation

    Returns:
        Dict with metrics as keys
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
   
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0
    total_eval_time = 0
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        final_iou_per_object = defaultdict(lambda: defaultdict(list))
        clicker_metrics = defaultdict(lambda: defaultdict(list))

        start_data_time = time.perf_counter()

        assert isinstance(missing_click_ratios, list)
        if len(missing_click_ratios) > 0:
            assert any(isinstance(x, float) for x in missing_click_ratios) and all(0. <= x <= 1. for x in missing_click_ratios)
            use_click_scheme = False
        else:
            use_click_scheme = True

        if vis_path is not None:
            vis_path = os.path.join( vis_path, f'variable_missing_clicks' )
            
        for idx, inputs in enumerate(data_loader):
            
            if inputs[0]['image_id'] in ids_to_skip:
                logger.info(f"skipping specified image_id: {inputs[0]['image_id']}")
                continue

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()

            assert isinstance(missing_click_ratios, (float, str, list))

            num_instances = inputs[0]['instances'].gt_masks.to('cpu').shape[0]
            if use_click_scheme:
                missing_click_ratios = [ round(((x+1)/num_instances),2) for x in reversed(range(num_instances)) ]
                missing_click_ratios.append(0.)
            
            predictor = PredictorOneclick(model)  # initializing predictor here already will re-use backbone features
            for i, missing_click_ratio in enumerate(missing_click_ratios):
                
                ## modify input in regards to clicks
                if use_click_scheme:
                    if i == 0:  # initialize all clicks as missing before adding them one by one each iteration
                        missing_clicks_idcs = list(range(num_instances))
                        inputs[0]['missing_clicks_idcs'] = missing_clicks_idcs
                    inputs[0]['instances'].set('gt_click_missing', [ True if i in missing_clicks_idcs else False for i in range(num_instances) ])
                else:
                    if missing_click_ratio == 0.:
                        missing_clicks_idcs = []
                        inputs[0]['missing_clicks_idcs'] = missing_clicks_idcs
                        inputs[0]['instances'].set('gt_click_missing', [ False for i in range(num_instances) ])
                    elif missing_click_ratio == 1.:
                        missing_clicks_idcs = list(range(num_instances))
                        inputs[0]['missing_clicks_idcs'] = missing_clicks_idcs
                        inputs[0]['instances'].set('gt_click_missing', [ True for i in range(num_instances) ])
                    else:
                        raise NotImplementedError

                clicker = ClickerOneclick(inputs)
                
                if i==0 and vis_path is not None:  # save gt vis
                    clicker.save_visualization(vis_path, ious=[], num_interactions=0)

                pred_masks, pred_scores = predictor.get_prediction(clicker)
                clicker.set_pred_masks(pred_masks, pred_scores)

                ious_clicked = clicker.compute_iou_clicked()
                num_clicks_missing = len(missing_clicks_idcs)
                num_instances_clicked = num_instances - num_clicks_missing
                save_preds = True if vis_path else False
                ious_learnt, clicker_metrics_iter, next_click, fn_gt_idcs = clicker.compute_iou_learnt_and_get_next_click(
                    use_click_scheme=use_click_scheme,
                    save_preds_info={
                        'file_name':Path(inputs[0]['file_name']).name,
                        'image_id':inputs[0]['image_id'],
                        'missing_click_ratio':missing_click_ratio,
                        'vis_path':vis_path,
                        'n_missing':num_clicks_missing,
                        'n_clicks':num_instances_clicked,
                    } if save_preds else None,
                )
                ious_all = ( ious_clicked + ious_learnt ) if ious_learnt is not None else ious_clicked
                
                if vis_path is not None:
                    clicker.save_visualization(
                        vis_path,
                        ious=ious_all if ious_all is not None else ious_clicked,
                        num_interactions=num_instances_clicked,
                        fn_gt_idcs=fn_gt_idcs,
                        )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
                final_iou_per_object[f'missing{int(missing_click_ratio * 100):03d}_clicked_miou'][f"{inputs[0]['image_id']}"].append(ious_clicked)
                final_iou_per_object[f'missing{int(missing_click_ratio * 100):03d}_learnt_miou'][f"{inputs[0]['image_id']}"].append(ious_learnt)
                final_iou_per_object[f'missing{int(missing_click_ratio * 100):03d}_all_miou'][f"{inputs[0]['image_id']}"].append(ious_all)
                for k,v in clicker_metrics_iter.items():
                    clicker_metrics[f'missing{int(missing_click_ratio * 100):03d}_learnt_{k}'][f"{inputs[0]['image_id']}"].append(v)
                clicker_metrics[f'missing{int(missing_click_ratio * 100):03d}_n_objects'][f"{inputs[0]['image_id']}"].append(torch.tensor(num_instances_clicked+num_clicks_missing))
                clicker_metrics[f'missing{int(missing_click_ratio * 100):03d}_n_clicked'][f"{inputs[0]['image_id']}"].append(torch.tensor(num_instances_clicked))
                clicker_metrics[f'missing{int(missing_click_ratio * 100):03d}_n_missing'][f"{inputs[0]['image_id']}"].append(torch.tensor(num_clicks_missing))

                if use_click_scheme:  # drop a click for next iteration
                    if next_click is not None:
                        missing_clicks_idcs.pop(next_click)
            
            
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        ),
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = {}
    for k,v in final_iou_per_object.items():
        results[k] = [v]
    for k,v in clicker_metrics.items():
        results[k] = [{k_:v_ for k_,v_ in v.items() if len(v_)}]
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
