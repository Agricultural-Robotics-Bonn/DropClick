

#### RUN FINETUNING ON TASK SPECIFIC DATASETS


## SELECT DATASET
DATASETS_TRAIN="('sb20_single_subset05imgs_train',)"
# DATASETS_TRAIN="('bup20_single_subset05imgs_train',)"

## SET PRETRAINED WEIGHTS (provided)
PRETRAINED_WEIGHTS=./output/lvis/v0/model_final.pth  # provided!


cd "$(dirname "${BASH_SOURCE[0]}")"
. .venv/bin/activate

export NCCL_P2P_LEVEL=NVL  # use nvlink if available
export PYTORCH_JIT=0
python train_net.py \
  --config-file configs/coco_lvis/swin/swin-base.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.000005 \
  MODEL.WEIGHTS $PRETRAINED_WEIGHTS \
  DATASETS.TRAIN $DATASETS_TRAIN \
  SOLVER.MAX_EPOCHS 1000 SOLVER.DECAY_EPOCHS "()"