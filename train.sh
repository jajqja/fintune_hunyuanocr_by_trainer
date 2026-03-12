#!/bin/bash

set -e  

# ==================== Configuration ====================
MODEL_PATH="tencent/HunyuanOCR"
DATA_PATH="data/"
OUTPUT_DIR="HunYuanOCR-SFT"

# Change to 1 if using Google Colab
NUM_GPUS=1  

# Training Hyperparameters
EPOCHS=5                   
BATCH_SIZE=2               
GRAD_ACCUM_STEPS=4         
LEARNING_RATE=2e-5
MAX_LENGTH=3000            

# ==================== Environment Setup ====================
echo "Setting up environment variables..."

# For Colab, usually only device 0 is available
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting Training..."

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --logging_steps 10 \

echo "Training Completed!"