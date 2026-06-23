#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/.hf_cache
export TRANSFORMERS_CACHE=/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/.hf_cache
cd /tmp/threestudio
python launch.py --config /mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectB_config/dreamfusion-sd-v1-5.yaml --export
