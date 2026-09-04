

#### RUN EVALUATION ON TASK SPECIFIC DATASETS


### SELECT DATASET
## SB20 eval set
DATASETS_TEST="sb20_single_eval"; EVAL_WEIGHTS=./output/sb20/v0/model_final.pth; SKIP_IDS="()"
## SB20 train set (special case of evaluating on the remaining n-5 training images, for semi-supervised downstream tasks)
# DATASETS_TEST="sb20_single_train"; EVAL_WEIGHTS=./output/sb20/v0/model_final.pth; SKIP_IDS="(1887, 1900, 1909, 1916, 1931)"

## BUP20 eval set
# DATASETS_TEST="(bup20_single_eval)"; EVAL_WEIGHTS=./output/bup20/v0/model_final.pth; SKIP_IDS=""
## BUP20 train set (special case of evaluating on the remaining n-5 training images, for semi-supervised downstream tasks)
# DATASETS_TEST="(bup20_single_train)"; EVAL_WEIGHTS=./output/bup20/v0/model_final.pth; SKIP_IDS="(304, 309, 420, 429, 320)"


cd "$(dirname "${BASH_SOURCE[0]}")"
. .venv/bin/activate

export NCCL_P2P_LEVEL=NVL  # use nvlink if available
python train_net.py \
  --num-gpus 1 \
  --eval-datasets $DATASETS_TEST \
  --eval-weights $EVAL_WEIGHTS \
  --eval-skip-ids "$SKIP_IDS" \
  DATALOADER.NUM_WORKERS 0
  # --visualize-testing \  # put above dataloader line to enable result visualization
  