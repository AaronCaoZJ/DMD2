#!/bin/bash

# 简单推理脚本 - 生成5张图片

cd /home/zhijun/Code/DMD2

export PYTHONPATH=/home/zhijun/Code/DMD2:$PYTHONPATH
export HF_HOME=/home/zhijun/Code/DMD2/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMD2/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMD2/ckpt/transformers

# 设置checkpoint路径
CHECKPOINT_DIR="/data/sdv15/sdv15_decoupled_4step/time_1769772371_seed10"
# CHECKPOINT_DIR="/data/sdv15/sdv15_decoupled_4step/time_1769762143_seed10"
# CHECKPOINT_DIR="/data/sdv15/sdv15_decoupled_4step"
# CHECKPOINT_DIR="/home/zhijun/Code/DMD2/ckpt/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# 找到最新的checkpoint
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*checkpoint_model_*/pytorch_model.bin 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# 运行推理
python main/simple_inference.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --output_dir "/data/sdv15/test_output/decoupled_dmd_1000" \
    --seed 42 \
    --device cuda \
    --prompts \
        "a beautiful landscape with mountains and a lake at sunset" \
        "a cute orange cat sitting on a windowsill" \
        "a futuristic cyberpunk city with neon lights" \
        "a portrait of a woman with flowers in her hair" \
        "a steaming cup of coffee on a wooden table"

echo "Inference completed! Check outputs at /data/sdv15/test_output"
