#!/bin/bash

# 设置所有必需的环境变量
export RUNTIME_SCRIPT_DIR=/home/leyankun/workspace/Tencent/Tencent_lyk
export TRAIN_LOG_PATH=/home/leyankun/workspace/Tencent/Tencent_lyk/logs
export TRAIN_TF_EVENTS_PATH=/home/leyankun/workspace/Tencent/Tencent_lyk/tensorboard
export TRAIN_CKPT_PATH=/home/leyankun/workspace/Tencent/Tencent_lyk/checkpoints

# 进入工作目录
cd ${RUNTIME_SCRIPT_DIR}

# 创建必要的目录（如果不存在）
mkdir -p ${TRAIN_LOG_PATH}
mkdir -p ${TRAIN_TF_EVENTS_PATH}
mkdir -p ${TRAIN_CKPT_PATH}

# 执行训练脚本
python -u main.py --use_mixed_optimizer --local_test
