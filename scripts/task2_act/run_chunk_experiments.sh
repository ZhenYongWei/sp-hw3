#!/bin/bash
# Launch chunk=10 and chunk=50 training in parallel

cd /mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3

# chunk=10 on GPUs 1,2,3
CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_act_official.py \
  --envs A --eval-envs D --epochs 200 --batch-size 256 --lr 1e-4 \
  --chunk-size 10 --kl-weight 10 --save-every 50 \
  --output-dir task2_output_official/envA_chunk10 \
  > task2_output_official/envA_chunk10_train.log 2>&1 &
PID10=$!
echo "chunk=10 PID: $PID10"

# chunk=50 on GPUs 4,5,6
CUDA_VISIBLE_DEVICES=4,5,6 python scripts/train_act_official.py \
  --envs A --eval-envs D --epochs 200 --batch-size 256 --lr 1e-4 \
  --chunk-size 50 --kl-weight 10 --save-every 50 \
  --output-dir task2_output_official/envA_chunk50 \
  > task2_output_official/envA_chunk50_train.log 2>&1 &
PID50=$!
echo "chunk=50 PID: $PID50"

echo "Waiting for both to finish..."
wait $PID10
echo "chunk=10 done with exit code $?"
wait $PID50
echo "chunk=50 done with exit code $?"
echo "ALL DONE"
