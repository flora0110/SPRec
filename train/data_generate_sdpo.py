import re
import json
import sys
import fire
# import gradio as gr
import numpy as np
# import torch
# torch.set_num_threads(1)
# from sentence_transformers import SentenceTransformer
import random
# import transformers
from tqdm import tqdm
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# from peft import PeftModel
# from transformers import GenerationConfig,AutoTokenizer
# from transformers import LlamaForCausalLM
# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

def main(
    n_negatives: int = 1,
    result_json_sft_data_train: str = "./models/SPRec/Goodreads_2048_0.00002/it0/data/sft_train.jsonl",
    result_json_sft_data_valid: str = "./models/SPRec/Goodreads_2048_0.00002/it0/data/sft_valid.jsonl",
    result_json_sdpo_data_train: str = "",
    result_json_sdpo_data_valid: str = "",
    seed: int = 42,
    train_sample_size:int = 1024,
    valid_sample_size:int = 128,
    id2name_path: str = "./eval/Goodreads/id2name.json",
):
    print(f"n_negatives: {n_negatives}")

    # # generate responses from model
    # tokenizer =  AutoTokenizer.from_pretrained(base_model)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # load_8bit = False
    # if device == "cuda":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         load_in_8bit=load_8bit,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         torch_dtype=torch.float16,
    #         device_map="auto"
    #     )
    # tokenizer.padding_side = "left"

    # model.eval()

    #emb_model = SentenceTransformer('/data/chenruijun/code/models/paraphrase-MiniLM-L3-v2')

    # [新增] 讀取 id2name 並準備隨機池
    print(f"Loading id2name from {id2name_path}...")
    with open(id2name_path, 'r') as f:
        id2name_data = json.load(f)
    all_book_names = list(id2name_data.values())

    # [新增] 設定隨機種子
    random.seed(seed)

    # [修改] 定義處理 SDPO 資料的函數
    def process_sdpo_data(input_file, output_file):
        print(f"Processing {input_file} -> {output_file}")
        sdpo_data = []
        
        # 讀取現有的 SFT jsonl 檔案
        with open(input_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines):
            item = json.loads(line)
            
            # 建構 SDPO 格式
            sdpo_case = {}
            
            # 1. Prompt: 組合 Instruction 與 Input
            sdpo_case['prompt'] = generate_prompt(item['instruction'], item['input'])
            
            # 2. Chosen: 原始正確答案
            sdpo_case['chosen'] = item['output']
            
            # 3. Rejected: 從 id2name 隨機選 n 個 (排除掉正確答案可選，但這裡依指示純隨機)
            # [新增] 隨機選取 n 個負樣本
            sampled_negatives = random.sample(all_book_names, n_negatives)
            # if n_negatives == 1:
            #     # 如果只有 1 個，直接存為字串，不要 list
            #     sdpo_case['rejected'] = sampled_negatives[0]
            # else:
            sdpo_case['rejected'] = sampled_negatives
            
            sdpo_data.append(sdpo_case)

        # 寫入結果
        with open(output_file, 'w') as f:
            for item in sdpo_data:
                json.dump(item, f)  
                f.write('\n')
        print(f"Saved {len(sdpo_data)} items to {output_file}")

    # [修改] 執行處理邏輯 (取代原本的訓練/預測流程)
    if result_json_sft_data_train and result_json_sdpo_data_train:
        process_sdpo_data(result_json_sft_data_train, result_json_sdpo_data_train)
        
    if result_json_sft_data_valid and result_json_sdpo_data_valid:
        process_sdpo_data(result_json_sft_data_valid, result_json_sdpo_data_valid)

    # def evaluate(
    #     instructions,
    #     inputs=None,
    #     temperature=0,
    #     top_p=0.9,
    #     top_k=40,
    #     num_beams=1,
    #     max_new_tokens=128,
    #     **kwargs,
    # ):
    #     prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
    #     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    #     generation_config = GenerationConfig(
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         num_beams=num_beams,
    #         num_return_sequences=num_beams,
    #         **kwargs,
    #     )
    #     with torch.no_grad():
    #         generation_output = model.generate(
    #             **inputs,
    #             generation_config=generation_config,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             max_new_tokens=max_new_tokens,
    #             pad_token_id = tokenizer.eos_token_id
    #         )
    #     s = generation_output.sequences
    #     output = tokenizer.batch_decode(s, skip_special_tokens=True)
    #     output = [_.split('Response:\n')[-1] for _ in output]
    #     real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
    #     return real_outputs
    
    # outputs = []
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # with open(train_json_file, 'r') as f:
    #     train_data = json.load(f)
    #     train_data = random.sample(train_data, train_sample_size)
    #     sft_train_data = train_data
    # with open(valid_json_file, 'r') as f:
    #     valid_data = json.load(f)
    #     valid_data = random.sample(valid_data, valid_sample_size)
    #     sft_valid_data = valid_data
    # with open(result_json_sft_data_train, 'w') as f:
    #     for item in sft_train_data:
    #         json.dump(item, f) 
    #         f.write('\n') 
    # with open(result_json_sft_data_valid, 'w') as f:
    #     for item in sft_valid_data:
    #         json.dump(item, f) 
    #         f.write('\n')    
    # data = train_data + valid_data
    # instructions = [_['instruction'] for _ in data]
    # inputs = [_['input'] for _ in data]
    # def batch(list, batch_size=batch_size):
    #     chunk_size = (len(list) - 1) // batch_size + 1
    #     for i in range(chunk_size):
    #         yield list[batch_size * i: batch_size * (i + 1)]
    # for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
    #     instructions, inputs = batch
    #     output = evaluate(instructions, inputs)
    #     outputs = outputs + output
        
    # for i, test in tqdm(enumerate(data)):
    #     data[i]['predict'] = outputs[i]

    # dpo_data = []

    # for data_point in data:
    #     dpo_case = {}
    #     dpo_case['prompt'] = data_point['instruction'] + data_point['input']
    #     dpo_case['chosen'] = data_point['output']
    #     pattern = r'"(.*?)"'
    #     item_names = re.findall(pattern, data_point['predict'][0])
    #     formatted_item_names = [f'\"{item}\"' for item in item_names]
    #     if len(formatted_item_names) > 0:
    #         dpo_case['rejected'] = formatted_item_names[0]+"\n"
    #     else:
    #         dpo_case['rejected'] = "\n"
    #     dpo_data.append(dpo_case)
        
    # # random.shuffle(dpo_data)
    # dpo_train_data = dpo_data[:train_sample_size]
    # dpo_valid_data = dpo_data[train_sample_size:]


    # with open(result_json_dpo_data_train, 'w') as f:
    #     for item in dpo_train_data:
    #         json.dump(item, f)  
    #         f.write('\n')  

    # with open(result_json_dpo_data_valid, 'w') as f:
    #     for item in dpo_valid_data:
    #         json.dump(item, f)  
    #         f.write('\n')


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
