import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel

import glob
import re

def recover_progress(output_prefix_pt):
    """
    掃描輸出目錄，找出已經完成的 chunk 以及已處理過的 sample_indices。
    """
    # 搜尋所有符合格式的 .pt 檔案
    pattern = f"{output_prefix_pt}_chunk*.pt"
    files = glob.glob(pattern)
    
    processed_indices = set()
    max_chunk_idx = -1
    
    if not files:
        return 0, processed_indices, 0

    print(f"Found {len(files)} existing chunks. Loading progress...")
    
    for fpath in tqdm(files, desc="Recovering progress"):
        # 1. 從檔名解析 chunk index (例如 ..._chunk036.pt -> 36)
        match = re.search(r"_chunk(\d+)\.pt", fpath)
        if match:
            c_idx = int(match.group(1))
            if c_idx > max_chunk_idx:
                max_chunk_idx = c_idx
        
        # 2. 讀取 .pt 檔案內容以獲取 sample_indices
        # 使用 map_location='cpu' 避免佔用 GPU 記憶體
        try:
            chunk_data = torch.load(fpath, map_location="cpu")
            for item in chunk_data:
                # 原始程式碼中 sample_indices 存的是 Tensor([idx])
                if "sample_indices" in item:
                    idx = item["sample_indices"].item()
                    processed_indices.add(idx)
        except Exception as e:
            print(f"Warning: Could not read {fpath}. Error: {e}")

    next_chunk_idx = max_chunk_idx + 1
    effective_count_start = len(processed_indices)
    
    print(f"Resume summary: Next chunk index = {next_chunk_idx}, "
          f"Already processed samples = {effective_count_start}")
    
    return next_chunk_idx, processed_indices, effective_count_start

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


def normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)
        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


def save_chunk(chunk_idx, results_chunk, json_chunk, output_prefix_pt, output_prefix_json):
    if not results_chunk:
        return

    pt_path = f"{output_prefix_pt}_chunk{chunk_idx:03d}.pt"
    json_path = f"{output_prefix_json}_chunk{chunk_idx:03d}.json"

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)

    torch.save(results_chunk, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_chunk, f, ensure_ascii=False, indent=2)

    print(f"[Chunk {chunk_idx}] Saved {len(results_chunk)} samples.")
    print(f"  PT   -> {pt_path}")
    print(f"  JSON -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for LLM generation")

    # Path arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data json file")
    parser.add_argument("--id2name_path", type=str, required=True, help="Path to the id2name json file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--output_prefix", type=str, default="random50_item_pref_similarity", help="Prefix for output filenames")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")

    # Parameter arguments
    parser.add_argument("--num_random_items", type=int, default=50, help="Number of random items to sample")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of samples per chunk file")
    
    args = parser.parse_args()

    # Derived paths
    output_prefix_pt = os.path.join(args.output_dir, args.output_prefix)
    output_prefix_json = os.path.join(args.output_dir, args.output_prefix)

    random.seed(args.random_seed)

    # ==== 3) 讀 GoodReads data & id2name ====

    print(f"Loading data from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading id2name from {args.id2name_path}")
    with open(args.id2name_path, "r", encoding="utf-8") as f:
        id2name = json.load(f)

    all_titles = list(id2name.values())  # 全部 item pool

    max_effective_samples = len(data)

    # ==== 4) 載入 model / tokenizer ====

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} to {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, # 建議加上 float16 以節省顯存
        device_map="auto"
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    start_chunk_idx, processed_indices, start_effective_count = recover_progress(output_prefix_pt)

    # 用來存「目前這一批 chunk」的結果
    results_chunk = []
    json_chunk = []
    chunk_idx = start_chunk_idx        # 接續下一個 chunk 編號
    effective_count = start_effective_count  # 接續目前的有效樣本數
    # chunk_idx = 0

    effective_count = 0  # 累積的有效 sample 數量

    for sample_idx, example in tqdm(list(enumerate(data)), desc="Processing samples"):

        if sample_idx in processed_indices:
            continue

        if effective_count >= max_effective_samples:
            break  # 已經有足夠有效資料就停

        instruction_text = example.get("instruction", "")
        input_text = example.get("input", "") # 安全獲取 input，若無則為空字串
        prompt = generate_prompt(instruction_text, input_text)


        chosen_raw = example["output"]
        chosen_title = chosen_raw.strip('"')

        # ==== 5) 從全體 item 中隨機抽 NUM_RANDOM_ITEMS 本（排除 chosen 自己）====

        candidate_pool = [t for t in all_titles if t != chosen_title]
        if not candidate_pool:
            continue

        k = min(args.num_random_items, len(candidate_pool))
        sampled_titles = random.sample(candidate_pool, k=k)

        minus_norm_ed = {}
        ches_scores = {}
        ln_ches_scores = {}
        last_inner_prods = {}

        for r_title in sampled_titles:
            query = prompt
            text_w = query + chosen_title
            text_l = query + r_title

            # ---- tokenize ----
            q_ids = tokenizer(query, padding=False, truncation=False, add_special_tokens=False).input_ids
            w_ids = tokenizer(text_w, padding=False, truncation=False, add_special_tokens=False).input_ids
            l_ids = tokenizer(text_l, padding=False, truncation=False, add_special_tokens=False).input_ids

            query_len = len(q_ids)
            pref_ids = w_ids[query_len:]
            dispref_ids = l_ids[query_len:]

            if len(pref_ids) == 0 or len(dispref_ids) == 0:
                continue

            # ==== 7) normalized edit distance ====
            d = normalized_edit_distance(pref_ids, dispref_ids)
            minus_norm_ed[r_title] = -float(d)

            # ==== 8) CHES / ln-CHES / last hidden inner product ====

            pref_tensor = torch.tensor(w_ids, dtype=torch.long, device=device)
            dispref_tensor = torch.tensor(l_ids, dtype=torch.long, device=device)

            with torch.no_grad():
                pref_out = model(input_ids=pref_tensor.unsqueeze(0), output_hidden_states=True)
                dispref_out = model(input_ids=dispref_tensor.unsqueeze(0), output_hidden_states=True)

            hidden_w = pref_out.hidden_states[-1][0]   # [L_w, H]
            hidden_l = dispref_out.hidden_states[-1][0]  # [L_l, H]

            preferred_hidden_embed = hidden_w[query_len - 1:]
            dispreferred_hidden_embed = hidden_l[query_len - 1:]

            if preferred_hidden_embed.shape[0] == 0 or dispreferred_hidden_embed.shape[0] == 0:
                continue

            preferred_hidden_embed = preferred_hidden_embed.to(torch.float32)
            dispreferred_hidden_embed = dispreferred_hidden_embed.to(torch.float32)

            S_w = preferred_hidden_embed.sum(dim=0)
            S_l = dispreferred_hidden_embed.sum(dim=0)
            T_w = preferred_hidden_embed.shape[0]
            T_l = dispreferred_hidden_embed.shape[0]

            ches = (S_w * S_l).sum() - torch.norm(S_w) ** 2

            pref_dispref = (S_w * S_l).sum() / (T_w * T_l)
            pref_only = torch.norm(S_w) ** 2 / (T_w ** 2)
            ln_ches = pref_dispref - pref_only

            last_inner = torch.inner(
                preferred_hidden_embed[-1],
                dispreferred_hidden_embed[-1]
            )

            ches_scores[r_title] = float(ches.detach().cpu())
            ln_ches_scores[r_title] = float(ln_ches.detach().cpu())
            last_inner_prods[r_title] = float(last_inner.detach().cpu())

        # 這個 sample 要求「至少有一個 candidate 算出數值」才算有效
        if len(ches_scores) == 0:
            continue

        # ==== 有效 sample，加入目前 chunk ====

        sample_result = {
            "sample_indices": torch.tensor([sample_idx]),
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        }
        results_chunk.append(sample_result)

        json_chunk.append({
            "prompt": prompt,
            "chosen": example["output"],
            "random_candidates": list(ches_scores.keys()),
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        })

        effective_count += 1

        # 每累積 CHUNK_SIZE 個有效 sample 就存一次
        if len(results_chunk) >= args.chunk_size:
            save_chunk(chunk_idx, results_chunk, json_chunk, output_prefix_pt, output_prefix_json)
            chunk_idx += 1
            results_chunk = []
            json_chunk = []

        # 如果剛好達到 MAX_EFFECTIVE_SAMPLES，也可以在這裡再檢查一次，安全起見
        if effective_count >= max_effective_samples:
            break

    # 迴圈結束後，若還有殘餘未滿 chunk_size 的 chunk，一樣要存
    if results_chunk:
        save_chunk(chunk_idx, results_chunk, json_chunk, output_prefix_pt, output_prefix_json)

    print(f"Total effective samples: {effective_count}")


if __name__ == "__main__":
    main()