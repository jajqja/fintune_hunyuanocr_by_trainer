import torch
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
import json
from PIL import Image
import os 
import argparse
from datasets import Dataset

from dataloader import load_dataset

# Prompt
PROMPT = "Extract all information of the document image and represent it in markdown format. Ensure the parsing follows the logical reading order. Do not describe or extract any figures, signatures, or seals."

def scale_image_limit(image: Image.Image, max_pixels: int = 3000000) -> Image.Image:
    """
    Scale ảnh sao cho tổng số pixel không vượt quá max_pixels mà vẫn giữ nguyên tỉ lệ.
    """
    width, height = image.size
    current_pixels = width * height

    if current_pixels <= max_pixels:
        return image  # Ảnh đã nhỏ sẵn rồi, không cần scale

    # Tính toán tỷ lệ scale dựa trên diện tích
    # ratio = sqrt(max_pixels / current_pixels)
    import math
    ratio = math.sqrt(max_pixels / current_pixels)

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Sử dụng Resampling.LANCZOS để giữ chất lượng ảnh tốt nhất khi thu nhỏ
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def create_sft_collate_fn(processor):
    tokenizer = processor.tokenizer
    IGNORE = -100 

    user_id = tokenizer.convert_tokens_to_ids("<｜hy_User｜>")
    assistant_id = tokenizer.convert_tokens_to_ids("<｜hy_Assistant｜>")

    def collate_fn(batch_samples):
        batch_input_ids = []
        batch_token_type_ids = []
        batch_imgs_pos = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        for sample in batch_samples:
            messages = json.loads(sample["messages_json"])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            image_path = sample["images"]
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                continue
                
            try:
                # Scale image to avoid OOM, but keep the aspect ratio
                image = [scale_image_limit(Image.open(image_path)).convert("RGB")]
            except Exception as e:
                print(f"Error: Can't open {image_path}: {e}")
                continue

            batch = processor(
                text=text,
                images=image,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=False,
            )

            batch_input_ids.append(batch["input_ids"])
            
            if "pixel_values" in batch.keys():
                batch_pixel_values.append(batch["pixel_values"])
                batch_imgs_pos.append(batch["imgs_pos"])
                batch_token_type_ids.append(batch["token_type_ids"])
                batch_image_grid_thw.append(batch["image_grid_thw"])
        
        if not batch_input_ids:
            return {} 
        
        input_ids = pad_cat_sequences(batch_input_ids, "right", processor.pad_id)
        token_type_ids = pad_cat_sequences(batch_token_type_ids, "right", 0)

        attention_mask = input_ids != processor.pad_id
        labels = torch.full_like(input_ids, IGNORE)

        data_dict ={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            if user_id in ids and assistant_id in ids:
                user_pos = ids.index(user_id)
                assistant_pos = len(ids) -1 - ids[::-1].index(assistant_id)
                labels[i, user_pos +1: assistant_pos + 1] = input_ids[i, user_pos +1: assistant_pos + 1] 

        data_dict["labels"] = labels
        
        ## It is debug
        # non_ignore_count = (labels != -100).sum().item()
        # print(f"DEBUG: Batch has {non_ignore_count} tokens to learn.")

        # if non_ignore_count == 0:
        #     print(f"CRITICAL WARNING: No labels found! Check token IDs.")
        #     print(f"User ID: {user_id}, Assistant ID: {assistant_id}")
        #     print(f"First 50 IDs in input: {input_ids[0][:50].tolist()}")
        
        batch_size, seq_len = input_ids.shape

        x_dim = 4
        position_ids = torch.arange(seq_len, device=input_ids.device , dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, x_dim,1)
        data_dict["position_ids"] = position_ids

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)
            imgs_pos = torch.cat(batch_imgs_pos, dim=0)

            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
            data_dict["imgs_pos"] = imgs_pos

        return data_dict
    return collate_fn


def pad_cat_sequences(sequences, padding_side='right', padding_value=0):
    assert padding_side in ["right", "left"]

    if not sequences:
        return torch.tensor([])

    max_len = max(seq.shape[1] for seq in sequences)

    outputs = []
    for i, seq in enumerate(sequences):
        pad_len = max_len - seq.shape[1]
        if padding_side == "right":
            seq = torch.nn.functional.pad(seq, (0, pad_len), value=padding_value)
        else:
            seq = torch.nn.functional.pad(seq, (pad_len, 0), value=padding_value)
        outputs.append(seq)
    outputs = torch.cat(outputs, dim=0)

    return outputs


def load_ocr_datasets(data_path):
    #list[dict]
    data_list = load_dataset(data_path)
    
    # Tạo Hugging Face Dataset
    full_ds = Dataset.from_list(data_list)
    
    # Chia train/test (ví dụ 90/10)
    ds_split = full_ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    ds_train, ds_test = ds_split["train"], ds_split["test"]

    column_names = ds_train.column_names

    ds_train = ds_train.map(
        format_data,
        num_proc=4, 
        remove_columns=column_names
    )

    ds_test = ds_test.map(
        format_data,
        num_proc=4, 
        remove_columns=column_names
    )
    
    return ds_train, ds_test

def format_data(sample):
    image_path = sample['image_path']

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant", 
            "content": sample['ground_truth']
        },
    ]
    messages_json = json.dumps(messages, ensure_ascii=False)
    return {
        "images": image_path,
        "messages_json": messages_json
    }

def parse_args():
    parser = argparse.ArgumentParser(description="HunYuanOCR SFT Training")
    
    # Path settings
    parser.add_argument("--model_name_or_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="HunYuanOCR-SFT")
    
    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=4096) # Ưu tiên set cụ thể cho OCR
    parser.add_argument("--logging_steps", type=int, default=10)

    # Distributed settings
    parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.local_rank != -1:
        # Thiết lập Distributed Data Parallel
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Processor & Model
    print(f"Loading model: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=False)
    
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager", 
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Distributed Data Parallel
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters
        )
    
    # Load Dataset
    print("Loading dataset...")
    train_dataset, eval_dataset = load_ocr_datasets(args.data_path)
    
    # Data collator
    data_collator = create_sft_collate_fn(processor)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,  
        num_train_epochs=args.num_train_epochs,  
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size,  
        gradient_accumulation_steps=args.gradient_accumulation_steps,  
        remove_unused_columns=False, 
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        max_length=args.max_length,
        dataset_text_field=None, 
        optim="adamw_torch_fused",  
        learning_rate=args.learning_rate,  
        
        logging_steps=args.logging_steps, 
        eval_strategy="epoch",  
        save_strategy="epoch",  
        
        bf16=True,   
        warmup_ratio=0.03,  
        report_to=["tensorboard"],  
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("--- Starting Training ---")
    trainer.train()
    
    if args.local_rank <= 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Model and processor saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
