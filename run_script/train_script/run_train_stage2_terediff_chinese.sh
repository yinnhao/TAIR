#!/bin/bash

# Stage 2: 训练文本识别模块（支持中文）
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
    --config configs/train/train_stage2_terediff_chinese.yaml \
    --config_testr testr/configs/TESTR/TESTR_R_50_Polygon_Chinese.yaml