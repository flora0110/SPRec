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
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,AutoTokenizer
from transformers import LlamaForCausalLM
from utils import save_run_script_content, set_seed

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# new trie class for constrained decoding
class Trie:
    def __init__(self, eos_token_id):
        self.children = {}
        self.is_end = False
        self.eos_token_id = eos_token_id

    def insert(self, token_ids):
        node = self
        for token in token_ids:
            if token not in node.children:
                node.children[token] = Trie(self.eos_token_id)
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix):
        node = self
        for token in prefix:
            if token in node.children:
                node = node.children[token]
            else:
                return []
        return list(node.children.keys()) + ([self.eos_token_id] if node.is_end else [])

# new function to filter rejected candidates
def filter_rejected_candidates(candidates, correct_answer, input_text):
    # history_books = set(re.findall(r'"(.*?)"', input_text))
    filtered = set()
    for cand in candidates:
        cleaned = cand.strip()
        match = re.search(r'"(.*?)"', cleaned)
        cand_name = match.group(1) if match else cleaned
        
        # not null and not the correct answer and not in history(removed)
        if cleaned and cand_name != correct_answer.strip() and cleaned != '""':
             if cand_name != correct_answer.strip('"'):
                filtered.add(cleaned)
    return list(filtered)

def main(
    train_json_file : str  = "",
    valid_json_file : str = "",
    result_json_dpo_data_train: str = "",
    result_json_dpo_data_valid: str = "",
    result_json_sft_data_train: str = "",
    result_json_sft_data_valid: str = "",
    base_model: str = "",
    lora_weights: str = "",
    batch_size:int = 4,
    train_sample_size:int = 1024,
    valid_sample_size:int = 128,
    load_8bit: bool = False,
    random_neg: bool = False,
    name2genre_path: str = "", # NEW: path to name2genre json file
    num_beams: int = 5, # NEW: number of beams for generation
    num_return_sequences: int = 3,  # [新增] 想要保留幾個候選者
    diverse_beam_search: bool = False, # [新增] 是否啟用多樣化束搜索
    diversity_penalty: float = 1.0,    # [新增] 多樣性懲罰係數
    seed: int = 42, 
    sh_file_path: str = "",
):  
    set_seed(seed)
    save_run_script_content(sh_file_path, result_json_dpo_data_train)

    # generate responses from model
    tokenizer =  AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    load_8bit = False
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
            device_map="auto"
        )
    tokenizer.padding_side = "left"

    model.eval()

    print("Building Trie...")
    trie = None
    if name2genre_path:
        with open(name2genre_path, "r", encoding="utf-8") as f:
            name2genre = json.load(f)
        trie = Trie(tokenizer.eos_token_id)
        for name in name2genre.keys():
            token_ids = tokenizer.encode(name, add_special_tokens=False)
            trie.insert(token_ids)
    else:
        print("Warning: name2genre_path not provided. Trie constraint will be disabled.")

    #emb_model = SentenceTransformer('/data/chenruijun/code/models/paraphrase-MiniLM-L3-v2')

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=num_beams, # NEW: need to increase num_beams for more candidates
        num_return_sequences=num_return_sequences,
        diverse_beam_search=diverse_beam_search,
        diversity_penalty=diversity_penalty,
        max_new_tokens=128,
        trie_constraint=None, # NEW: ad
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs_tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # NEW: for multi batch
        tokenizer.padding_side = "left"
        # prompt_len = inputs_tokenized['input_ids'].shape[1]

        # NEW
        prompt_end_text = "### Response:\n"
        prompt_end_ids = tokenizer.encode(prompt_end_text, add_special_tokens=False)

        def find_response_start(input_ids, prompt_end_ids):
            """動態尋找 Prompt 結束的位置"""
            n = len(prompt_end_ids)
            for i in range(len(input_ids) - n + 1):
                if input_ids[i:i+n] == prompt_end_ids:
                    return i + n
            return None

        prefix_allowed_tokens_fn = None

        if trie_constraint:
            def constraint_fn(batch_id, input_ids):

                # NEW
                input_ids_list = input_ids.tolist()
                response_start = find_response_start(input_ids_list, prompt_end_ids)
                if response_start is None:
                     return list(range(tokenizer.vocab_size))
                generated_part = input_ids_list[response_start:]
                allowed = trie_constraint.get_allowed_tokens(generated_part)
                return allowed if allowed else [tokenizer.eos_token_id]

                # generated_part = input_ids[prompt_len:].tolist()
                # allowed = trie_constraint.get_allowed_tokens(generated_part)
                # return allowed if allowed else [tokenizer.eos_token_id]
            
            prefix_allowed_tokens_fn = constraint_fn

        gen_kwargs = {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id
        }

        base_gen_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": False, # Code A 顯式關閉採樣，使用確定性 Beam Search
        }

        if diverse_beam_search:
            # Code A 的 Diverse Beam Search 邏輯
            base_gen_config.update({
                "num_beams": num_return_sequences * 2, # 通常 Beam 數要大於回傳數
                "num_beam_groups": num_return_sequences, # 分組數通常等於回傳數
                "diversity_penalty": diversity_penalty,
                "num_return_sequences": num_return_sequences
            })
        else:
            # 普通 Beam Search
            base_gen_config.update({
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences # 這裡允許只回傳部分 Beam
            })

        generation_config = GenerationConfig(**base_gen_config, **kwargs)
        gen_kwargs["generation_config"] = generation_config

        if prefix_allowed_tokens_fn:
             gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs_tokenized,
                **gen_kwargs
            )

        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)

        actual_return_num = generation_config.num_return_sequences
        
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = [output[i * actual_return_num: (i + 1) * actual_return_num] for i in range(len(output) // actual_return_num)]
        return real_outputs

        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        #     num_beams=num_beams,
        #     num_return_sequences=num_beams,
        #     **kwargs,
        # )
        # with torch.no_grad():
        #     gen_kwargs = {
        #         "generation_config": generation_config,
        #         "return_dict_in_generate": True,
        #         "output_scores": True,
        #         "max_new_tokens": max_new_tokens,
        #         "pad_token_id": tokenizer.eos_token_id
        #     }

        #     if prefix_allowed_tokens_fn:
        #          gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            
        #     generation_output = model.generate(
        #         **inputs_tokenized,
        #         **gen_kwargs
        #     )

        # s = generation_output.sequences
        # output = tokenizer.batch_decode(s, skip_special_tokens=True)

        # output = [_.split('Response:\n')[-1] for _ in output]
        # real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        # return real_outputs
    
    outputs = []
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
        train_data = random.sample(train_data, train_sample_size)
        sft_train_data = train_data
    with open(valid_json_file, 'r') as f:
        valid_data = json.load(f)
        valid_data = random.sample(valid_data, valid_sample_size)
        sft_valid_data = valid_data
    with open(result_json_sft_data_train, 'w') as f:
        for item in sft_train_data:
            json.dump(item, f) 
            f.write('\n') 
    with open(result_json_sft_data_valid, 'w') as f:
        for item in sft_valid_data:
            json.dump(item, f) 
            f.write('\n')    
    data = train_data + valid_data
    instructions = [_['instruction'] for _ in data]
    inputs = [_['input'] for _ in data]
    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]

    def batch_generator(list_data, batch_size=batch_size):
        chunk_size = (len(list_data) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list_data[batch_size * i: batch_size * (i + 1)]
            
    for i, batch_data in tqdm(enumerate(zip(batch_generator(instructions), batch_generator(inputs)))):
        batch_instructions, batch_inputs = batch_data
        # NEW
        output = evaluate(
            batch_instructions, 
            batch_inputs, 
            trie_constraint=trie, 
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            diverse_beam_search=diverse_beam_search,
            diversity_penalty=diversity_penalty
        )
        outputs = outputs + output
        
    for i, test in tqdm(enumerate(data)):
        data[i]['predict'] = outputs[i]

    dpo_data = []

    for data_point in data:
        dpo_case = {}
        dpo_case['prompt'] = data_point['instruction'] + data_point['input']
        dpo_case['chosen'] = data_point['output']

        # pattern = r'"(.*?)"'
        # item_names = re.findall(pattern, data_point['predict'][0])
        # formatted_item_names = [f'\"{item}\"' for item in item_names]
        # if len(formatted_item_names) > 0:
        #     dpo_case['rejected'] = formatted_item_names[0]+"\n"
        # else:
        #     dpo_case['rejected'] = "\n"
        # dpo_data.append(dpo_case)

        # [FIX] Replaced old logic with filter_rejected_candidates
        # data_point['predict'] contains a list of 'num_beams' candidates
        raw_candidates = data_point['predict']
        
        # Prepare candidates for filtering (extract book name if possible, or use raw)
        candidates_formatted = []
        for raw in raw_candidates:
             match = re.search(r'"(.*?)"', raw)
             if match:
                 candidates_formatted.append(f'"{match.group(1)}"')
             else:
                 candidates_formatted.append(raw)

        # Apply filtering
        rejected_list = filter_rejected_candidates(
            candidates_formatted, 
            data_point['output'], 
            data_point['input']
        )

        # if len(rejected_list) > 0:
        #     dpo_case['rejected'] = rejected_list[0] + "\n"
        # else:
        #     dpo_case['rejected'] = "\n" # Or handle empty rejected case (e.g. skip)
            
        # dpo_data.append(dpo_case)
        dpo_case['rejected'] = rejected_list 
        
        dpo_data.append(dpo_case)
        
    # random.shuffle(dpo_data)
    dpo_train_data = dpo_data[:train_sample_size]
    dpo_valid_data = dpo_data[train_sample_size:]


    with open(result_json_dpo_data_train, 'w') as f:
        for item in dpo_train_data:
            json.dump(item, f)  
            f.write('\n')  

    with open(result_json_dpo_data_valid, 'w') as f:
        for item in dpo_valid_data:
            json.dump(item, f)  
            f.write('\n')


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
