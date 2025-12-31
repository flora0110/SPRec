import re
import json
import sys
import fire
# import gradio as gr
import numpy as np
import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer
import random
import transformers
from tqdm import tqdm
import json
import os
import glob # [新增] 用於搜尋檔案
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,AutoTokenizer
from transformers import LlamaForCausalLM

# [修改] 根據新的邏輯調整了預設參數與新增路徑參數
def main(
    train_json_file : str  = "", # [修改] 指定 SFT 來源
    similarity_chunk_dir : str = "", # [新增] 相似度 Chunk 檔案的資料夾
    output_dir : str = "", # [新增] 輸出結果的資料夾
    # 以下參數保留但在此邏輯中可能用不到，為了兼容性保留
    base_model: str = "",
    lora_weights: str = "",
    batch_size:int = 4,
    train_sample_size:int = 1024, # 這裡的語意變為：要處理多少筆資料，或者可以忽略直接處理全部
    valid_sample_size:int = 128,
    load_8bit: bool = False,
    random_neg: bool = False,
):
    
    # [修改] 註解掉模型加載部分，因為我們不需要跑推論
    """
    # generate responses from model
    tokenizer =  AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    load_8bit = False
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer.padding_side = "left"

    model.eval()
    """

    # [新增] 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # [新增] 定義我們要處理的 Metrics 列表
    metrics = [
        "minus_normalized_edit_distances",
        "ches_scores",
        "ln_ches_scores",
        "last_hidden_embedding_inner_prods"
    ]
    
    # [新增] 初始化結果容器，每個 Metric 都有 Farthest (Max) 和 Nearest (Min)
    results = {}
    for metric in metrics:
        results[f"farthest_{metric}"] = [] # Max
        results[f"nearest_{metric}"] = []  # Min

    # [新增] 步驟 1: 讀取 SFT Train Data 作為基準
    print(f"Loading SFT data from {train_json_file}...")
    sft_data_map = {} # 用 prompt 作為 key 來對齊
    with open(train_json_file, 'r') as f:
        # 逐行讀取 JSONL
        for line in f:
            item = json.loads(line)
            # 重新建構 prompt 以便與 Similarity 檔案對齊
            prompt_text = generate_prompt(item['instruction'], item.get('input', ''))
            sft_data_map[prompt_text] = item

    # [新增] 步驟 2: 讀取 Similarity Chunk Files (000-049)
    # 這裡假設檔案命名格式固定為 train_item_pref_similarity_chunkXXX.json
    print(f"Loading Similarity Chunks from {similarity_chunk_dir}...")
    
    similarity_data_map = {}
    chunk_files = sorted(glob.glob(os.path.join(similarity_chunk_dir, "train_item_pref_similarity_chunk*.json")))
    
    for chunk_file in tqdm(chunk_files, desc="Reading Chunks"):
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
            for item in chunk_data:
                # 假設 chunk 資料裡的 prompt 已經是完整格式，如果不是，需要自行調整
                # 根據輸入格式範例，這裡有 prompt 欄位
                similarity_data_map[item['prompt']] = item

    # [新增] 步驟 3: 對齊資料並選取 Rejected
    print("Processing and selecting candidates...")
    
    # 遍歷 SFT 資料 (確保順序或集合與 SFT 一致)
    for prompt_key, sft_item in tqdm(sft_data_map.items(), desc="Matching"):
        
        if prompt_key not in similarity_data_map:
            continue # 如果在相似度檔案中找不到對應的 prompt，則跳過

        sim_item = similarity_data_map[prompt_key]
        
        # 準備基礎 DPO 結構
        base_dpo_entry = {
            "prompt": prompt_key,
            "chosen": sim_item['chosen'], # 使用 Similarity 檔案中的 chosen (通常帶有換行符號)
            # rejected 待填入
        }

        # 針對每一個 Metric 計算 Max 和 Min
        for metric in metrics:
            scores_dict = sim_item.get(metric, {})
            if not scores_dict:
                continue

            # 找出最大值與最小值的 Candidate
            # key 是 candidate string, value 是分數
            
            # Farthest (Max Score)
            max_candidate = max(scores_dict, key=scores_dict.get)
            
            # Nearest (Min Score)
            min_candidate = min(scores_dict, key=scores_dict.get)

            # 格式化 Rejected 字串 (加上引號，模擬原始邏輯)
            # 原始邏輯：item_names = re.findall... formatted = f'\"{item}\"'
            # 這裡直接將 candidate 字串加上引號並換行
            
            # Farthest Entry
            farthest_entry = base_dpo_entry.copy()
            farthest_entry['rejected'] = f"\"{max_candidate}\"\n"
            results[f"farthest_{metric}"].append(farthest_entry)

            # Nearest Entry
            nearest_entry = base_dpo_entry.copy()
            nearest_entry['rejected'] = f"\"{min_candidate}\"\n"
            results[f"nearest_{metric}"].append(nearest_entry)

    # [新增] 步驟 4: 寫入檔案
    print("Saving results...")
    for key, data_list in results.items():
        output_subdir = os.path.join(output_dir, key)
        
        # [關鍵修正]：如果該子資料夾不存在，則建立它
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        # 設定完整的檔案路徑
        output_filename = os.path.join(output_subdir, "train.jsonl")
        
        with open(output_filename, 'w') as f:
            for item in data_list:
                json.dump(item, f)
                f.write('\n')
        
        print(f"Saved {len(data_list)} items to {output_filename}")


    # [修改] 註解掉原始的 evaluate 和寫入邏輯
    """
    outputs = []
    # ... (原始的大段 inference 邏輯) ...
    with open(result_json_dpo_data_valid, 'w') as f:
        for item in dpo_valid_data:
            json.dump(item, f)  
            f.write('\n')
    """

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)