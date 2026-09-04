#Adapted by Amit Rana from: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe
import csv
import yaml
import numpy as np

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"",
    category=FutureWarning,  # ignore all FutureWarnings
)

import copy
import itertools
import logging

from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
    TrainerBase,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref
from dropclick.utils.misc import default_argument_parser

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dropclick import (
    COCOLVISDatasetMapperOneclick,
    EvaluationDatasetMapperOneclick,
    add_maskformer2_config,
    add_hrnet_config
)
from dropclick.inference.utils.eval_utils import log_underoneclick
from detectron2.utils.events import EventStorage, JSONWriter


from rich import print
from datetime import datetime
import os
from pathlib import Path
from detectron2.data.samplers import InferenceSampler
from detectron2.config import CfgNode as CN


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to Mask2Former.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        sampler = None
        datset_mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        if datset_mapper_name == "coco_lvis_oneclick":
            mapper = EvaluationDatasetMapperOneclick(cfg, False, dataset_name)
        else:
            mapper=None
        sampler=None  # for quick prototyping use: sampler=InferenceSampler(n_samples); n_samples must be divisible by the number of workers
        assert mapper is not None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper, sampler=sampler)
        
    @classmethod
    def build_train_loader(cls, cfg):
        datset_mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        if datset_mapper_name == "coco_lvis_oneclick":
            mapper = COCOLVISDatasetMapperOneclick(cfg,True)
        else:
            mapper = None
        assert mapper is not None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    @classmethod
    def test(cls, cfg, model, iter_, evaluators=None):
        """
        Method is called after every evaluation Checkpoint iteration.
        You can evaluate on any dataset and log the results/metrics 
        for debugging and performance measure puposes.
        """
        res = cls.interactive_evaluation(cfg, model, iter_)
        return res

    @classmethod
    def interactive_evaluation(cls, cfg, model, iter_, args=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """

        logger = logging.getLogger(__name__)

        ## VALIDATION
        if not args:
            assert len(cfg.DATASETS.TEST)<=1, f"Proper metric logging is not implemented for multi dataset testing {cfg.DATASETS.TEST}"
            if not comm.get_world_size() == 1:
                print('WARNING: skipped validation as num_gpus is > 1')
                return

            vis_path = os.path.join( cfg.OUTPUT_DIR, 'test_val', f'{iter_:0>8}' )

            from dropclick.inference.oneclick.evaluator import evaluate

            res = None
            dataset_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST)>0 else None
            if dataset_name is not None:
                print(f'\nVALIDATING one-click for dataset {dataset_name}...\n')
                data_loader = cls.build_test_loader(cfg, dataset_name)
                res_out = {}
                results_i = evaluate(
                    model,
                    data_loader,
                    vis_path=vis_path if args.visualize_testing else None,
                    missing_click_ratios=[0., 1.],  # in val use fix ratios to be faster. parse [] for full click scheme
                )
                res_out.update( log_underoneclick(results_i, dataset_name=dataset_name, testmode='valid',) )
                return res_out
            else:
                print('WARNING: skipped validation as no dataset is given in cfg.DATASETS.TEST')
                return

        ## EVALUATION
        if args and args.eval_weights:
            eval_datasets = args.eval_datasets
            assert len(eval_datasets)<=1, f"Proper metric logging is not implemented for multi dataset testing; eval_datasets: {eval_datasets}"
            vis_path = cfg.CUSTOM.VIS_PATH
        
            # for dataset_name in eval_datasets:
            dataset_name = eval_datasets[0] if len(eval_datasets)>0 else None
            if dataset_name is not None:
                print(f'\nEVALUATING one-click for dataset {dataset_name}...\n')

                from dropclick.inference.oneclick.evaluator import evaluate
                
                data_loader = cls.build_test_loader(cfg, dataset_name)
                
                res_out = {}
                results_i = evaluate(
                    model,
                    data_loader,
                    vis_path=vis_path if args.visualize_testing else None,
                    missing_click_ratios=[] if not ('lvis' in cfg.OUTPUT_DIR and 'coco' in dataset_name) else [0., 1.],
                    ids_to_skip=args.eval_skip_ids
                )
                results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                if comm.is_main_process():
                    # sum the values with same keys
                    assert len(results_i) > 0
                    res_gathered = results_i[0]
                    results_i.pop(0)
                    for _d in results_i:
                        for k in _d.keys():
                            res_gathered[k] += _d[k]
                    res_out.update( log_underoneclick(res_gathered, dataset_name=dataset_name, testmode='eval') )
                ## save
                json_writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))
                with EventStorage() as storage:
                    for k,v in res_out.items():
                        storage.put_scalar(k,v)
                    json_writer.write()

                return res_out
            else:
                print('WARNING: skipped evaluation as no dataset is given in args.eval_datasets')
                return


    def build_hooks(self):
        hooks = super().build_hooks()
        ## replace eval_func
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, self.iter)
            return self._last_eval_results
        from detectron2.engine.hooks import EvalHook
        eval_hook = [ h for h in hooks if isinstance(h, EvalHook) ][0]
        idx = hooks.index(eval_hook)
        hooks[idx] = EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results)
        return hooks


def merge_custom_list(cfg, opts):
    '''
        This function merges custom cli arguments into detectron2 config.
        It takes config keys with arbitrary depth and creates new nodes if needed.
        Currently if you want to use custom arguments, they need to be defined in here.
    '''
    custom_opts = [
        'SOLVER.MAX_EPOCHS',
        'TEST.VAL_EVERY_N_EPOCHS',
        'TEST.SAVE_EVERY_N_VAL',
        'SOLVER.DECAY_EPOCHS',
        'SOLVER.CHECKPOINT_EPOCHS',
    ]
    not_given = []
    for opt in custom_opts:
        if opt not in opts:
            not_given.append(opt)
            continue
        else:
            opt_i = opts.index( opt )
        keys = opt.split('.')
        parent = cfg
        for i,k in enumerate(keys):
            node = parent.get(k)
            if node is not None:
                parent = node
                continue
            else:
                if i < len(keys)-1:  # if not deepest: create node
                    setattr( parent, k, CN()  )
                else:  # if deepest
                    v = CN._decode_cfg_value(opts[opt_i+1])
                    setattr( parent, k, v )
                    opts.pop(opt_i) ; opts.pop(opt_i)
            parent = parent.get(k)
    print( f'not given costum opts: {not_given}' )
    
    
def handle_epoch_opts( cfg, opts ):
    '''
        handling epochs/iters
    '''
    if hasattr( cfg.SOLVER, 'MAX_EPOCHS' ):
        print( 'max_epochs:', cfg.SOLVER.MAX_EPOCHS )
        from detectron2.data import MetadataCatalog, DatasetCatalog
        print( DatasetCatalog )
        N_TRAIN = sum( [ len(DatasetCatalog.get(dataset_train)) for dataset_train in cfg.DATASETS.TRAIN ] )
        print( 'Number of images in training set(s):', N_TRAIN )
        ITERS_PER_EPOCH = (N_TRAIN//cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.MAX_ITER = int( ITERS_PER_EPOCH * cfg.SOLVER.MAX_EPOCHS )
        if hasattr( cfg.TEST, 'VAL_EVERY_N_EPOCHS'):
            cfg.TEST.EVAL_PERIOD = int( ITERS_PER_EPOCH * cfg.TEST.VAL_EVERY_N_EPOCHS )
            if hasattr( cfg.TEST, 'SAVE_EVERY_N_VAL'):
                cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.SAVE_EVERY_N_VAL * cfg.TEST.EVAL_PERIOD
        if hasattr( cfg.SOLVER, 'CHECKPOINT_EPOCHS' ):
            cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.CHECKPOINT_EPOCHS * ITERS_PER_EPOCH
        cfg.SOLVER.STEPS = tuple( [ x*ITERS_PER_EPOCH for x in cfg.SOLVER.DECAY_EPOCHS ] )
        print( 'max_iters:', cfg.SOLVER.MAX_ITER )
        print( 'eval_period', cfg.TEST.EVAL_PERIOD )
        print( 'checkpoint_period', cfg.SOLVER.CHECKPOINT_PERIOD )
        

def setup(args, timestamp):
    """
    Create configs and perform basic setups.
    """
    ## get cfg
    cfg = get_cfg()
    cfg.CUSTOM = CN()
    ## TRAIN
    if not args.eval_weights:
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_hrnet_config(cfg)
        cfg.merge_from_file(args.config_file)
        merge_custom_list(cfg, args.opts)
        cfg.merge_from_list(args.opts)  # check merge_custom_list() if error occurs here
        ### no. of steps calculation
        handle_epoch_opts( cfg, args.opts )
        ## add timestamp to output folder
        cfg.OUTPUT_DIR = os.path.join( cfg.OUTPUT_DIR, f'{timestamp}_{cfg.MODEL.MASK_FORMER.INTERACTIVE_TRANSFORMER_NAME}' )
    ## EVAL
    elif args.eval_weights:
        if args.config_file and 'swin-base.yaml' in args.config_file:
            output_dir = 'output/base_models/swin-base'
            cfg.set_new_allowed(True)
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            add_hrnet_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            cfg.MODEL.WEIGHTS = args.eval_weights
            cfg.OUTPUT_DIR = output_dir
            cfg.CUSTOM.VIS_PATH = os.path.join( output_dir, 'visualization' )
            print(f'exporting visualizations to: {cfg.CUSTOM.VIS_PATH}')
        else:
            assert not args.config_file
            output_dir = Path(args.eval_weights).parent
            config_path = os.path.join(output_dir, 'config.yaml') if not args.config_file else args.config_file
            import re
            eval_iter = Path(args.eval_weights).stem.split("_")[-1]  # new folder per evaluation
            eval_path = os.path.join( Path(config_path).parent, f'test_eval_{eval_iter}_{timestamp}' )
            cfg.set_new_allowed(True)
            cfg.merge_from_file(config_path)
            ## override with command line opts
            if len(args.opts):
                merge_custom_list(cfg, args.opts)
                cfg.merge_from_list(args.opts)  # check merge_custom_list() if error occurs here
            cfg.MODEL.WEIGHTS = args.eval_weights
            cfg.OUTPUT_DIR = eval_path  # saves new dir in case it changed (moved/renamed after training)
            cfg.CUSTOM.VIS_PATH = os.path.join( eval_path, 'visualization' )
            print(f'exporting visualizations to: {cfg.CUSTOM.VIS_PATH}')
    ### FOR ALL MODES
    # TODO currently still requires manual per dataset subdirectory handling here
    if any('coco_lvis' in x for x in cfg.DATASETS.TRAIN):
        if '/lvis/' not in cfg.OUTPUT_DIR:
            cfg.OUTPUT_DIR = os.path.join(Path(cfg.OUTPUT_DIR).parent, 'lvis', Path(cfg.OUTPUT_DIR).name)
            print(f'WARNING: changed output dir according to dataset: {cfg.OUTPUT_DIR}')
    elif any('sb20' in x for x in cfg.DATASETS.TRAIN):
        assert len(cfg.DATASETS.TRAIN)==1
        if '/sb20/' not in cfg.OUTPUT_DIR:
            cfg.OUTPUT_DIR = os.path.join(Path(cfg.OUTPUT_DIR).parent, 'sb20', Path(cfg.OUTPUT_DIR).name)
            print(f'WARNING: changed output dir according to dataset: {cfg.OUTPUT_DIR}')
    elif any('bup20' in x for x in cfg.DATASETS.TRAIN):
        assert len(cfg.DATASETS.TRAIN)==1
        if '/bup20/' not in cfg.OUTPUT_DIR:
            cfg.OUTPUT_DIR = os.path.join(Path(cfg.OUTPUT_DIR).parent, 'bup20', Path(cfg.OUTPUT_DIR).name)
            print(f'WARNING: changed output dir according to dataset: {cfg.OUTPUT_DIR}')
    else:
        raise NotImplementedError(f'output dir handling not implemented for dataset: {cfg.DATASETS.TRAIN}')
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dropclick")
    return cfg


def main(args, timestamp):

    cfg = setup(args, timestamp)
    if args.eval_weights:
        print(cfg.OUTPUT_DIR)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        res = Trainer.interactive_evaluation(cfg, model, None, args)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    return trainer.train()


if __name__ == "__main__":
    
    parser = default_argument_parser()
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url='auto',
        args=(args,timestamp),
    )
