#!/bin/bash

# ==== 設定變數 ====
base_model="meta-llama/Llama-3.2-1B-Instruct" # [重要] 這裡要填 Base Model
NUM_RANDOM_ITEMS=50
RANDOM_SEED=42
CHUNK_SIZE=100

# ==== 執行 Python 腳本 ====
for category in "Goodreads"
do
    # 假設你的 LoRA 權重路徑
    lora_weights="./models/SFT_4096/${category}"
    id2name_path="./eval/${category}/id2name.json"
    
    # [修正 1] 移除等號後的空白
    output_dir="./models/Random${NUM_RANDOM_ITEMS}_Similarity/${category}"

    echo "Starting metric computation..."
    echo "Category: ${category}"
    # [修正 2] 變數名稱改為正確的小寫引用，並確保路徑正確
    echo "Data Path (Train): ./data/${category}/train.json"
    echo "LoRA Weights: ${lora_weights}"
    echo "Output Dir: ${output_dir}"

    # [修正 3] 確保輸出目錄存在 (使用正確的變數名稱)
    mkdir -p "${output_dir}"

    # [注意] 這裡我修改了參數，讓它同時傳入 base_model 和 lora_weights
    # 請記得同步修改 Python 程式 (見下方)
    
    # 1. Run for Training Data
    echo "Processing Training Data..."
    python ./train/compute_similarity.py \
        --data_path "./data/${category}/train.json" \
        --id2name_path "${id2name_path}" \
        --model_path "${base_model}" \
        --lora_path "${lora_weights}" \
        --output_dir "${output_dir}" \
        --output_prefix "train_item_pref_similarity" \
        --num_random_items ${NUM_RANDOM_ITEMS} \
        --random_seed ${RANDOM_SEED} \
        --chunk_size ${CHUNK_SIZE}

    # 2. Run for Validation Data
    echo "Processing Validation Data..."
    python ./train/compute_similarity.py \
        --data_path "./data/${category}/valid.json" \
        --id2name_path "${id2name_path}" \
        --model_path "${base_model}" \
        --lora_path "${lora_weights}" \
        --output_dir "${output_dir}" \
        --output_prefix "valid_item_pref_similarity" \
        --num_random_items ${NUM_RANDOM_ITEMS} \
        --random_seed ${RANDOM_SEED} \
        --chunk_size ${CHUNK_SIZE}

    echo "Done with ${category}!"
done