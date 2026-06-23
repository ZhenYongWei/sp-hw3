#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/.hf_cache
export TRANSFORMERS_CACHE=/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/.hf_cache
cd /tmp/threestudio
python launch.py --config /mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectC_config/stable-zero123-custom.yaml --export
