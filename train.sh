#!/bin/bash

set -e  

# ==================== Configuration ====================
MODEL_PATH="tencent/HunyuanOCR"
DATA_TRAIN="datatrain"
DATA_TEST="datatest"
OUTPUT_DIR="HunYuanOCR-VNCLUR"
PROMPTS_FILE="prompts.json"  # JSON file mapping folder names to prompts

# Change to 1 if using Google Colab
NUM_GPUS=1  

# Training Hyperparameters
EPOCHS=8               
BATCH_SIZE=2           
GRAD_ACCUM_STEPS=4      
LEARNING_RATE=2e-4
MAX_LENGTH=None        

# ==================== Environment Setup ====================
echo "Setting up environment variables..."

# For Colab, usually only device 0 is available
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting Training..."

accelerate launch \
    --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_train "$DATA_TRAIN" \
    --data_test "$DATA_TEST" \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps 10 \

echo "Training Completed!"
