

#### RUN PRE-TRAINING ON COCO/LVIS (weights already provided!)


cd "$(dirname "${BASH_SOURCE[0]}")"
. ./.venv/bin/activate

export NCCL_P2P_LEVEL=NVL  # use nvlink if available
export PYTORCH_JIT=0
python train_net.py \
  --config-file configs/coco_lvis/swin/swin-base.yaml \
  --num-gpus 8 \
  SOLVER.MAX_EPOCHS 60 SOLVER.CHECKPOINT_EPOCHS 5 SOLVER.DECAY_EPOCHS "(54,58)"









