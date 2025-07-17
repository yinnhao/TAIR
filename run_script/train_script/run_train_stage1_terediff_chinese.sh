#!/bin/bash

# Stage 1: 训练图像恢复模块（支持中文）
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config configs/train/train_stage1_terediff_chinese.yaml \
    --config_testr testr/configs/TESTR/TESTR_R_50_Polygon_Chinese.yaml