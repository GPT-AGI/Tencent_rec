#!/bin/bash

# 设置所有必需的环境变量
export RUNTIME_SCRIPT_DIR=/root/GR/Tencent_rec
export TRAIN_DATA_PATH=/root/GR/Tencent_rec/TencentGR_1k
export TRAIN_LOG_PATH=/root/GR/Tencent_rec/logs
export TRAIN_TF_EVENTS_PATH=/root/GR/Tencent_rec/tensorboard
export TRAIN_CKPT_PATH=/root/GR/Tencent_rec/checkpoints

# 进入工作目录
cd ${RUNTIME_SCRIPT_DIR}


# 执行训练脚本
python -u main.py --use_mixed_optimizer
