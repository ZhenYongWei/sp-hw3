#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/.hf_cache
huggingface-cli download xiaoma26/calvin-lerobot --repo-type dataset --local-dir /root/public/data/calvin-lerobot
