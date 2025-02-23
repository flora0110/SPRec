import os
import torch
import warnings
import re
import wandb
from typing import List, Optional
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
# from utils import find_all_linear_names, print_trainable_parameters
import pandas as pd
from accelerate import Accelerator
import numpy as np
import torch
import bitsandbytes as bnb
import fire
import json

def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)

def gh_tr(category:str,test_data,name2genre:dict,genre_dict:dict):
    for data in tqdm(test_data,desc="Processing category data......"):
        input = data['input']
        names = re.findall(r'"([^"]+)"', input)
        for name in names:
            if name in name2genre:
                genres = name2genre[name]
            else:
                continue
            for genre in genres:
                if genre in genre_dict:
                    genre_dict[genre] += 1/len(genres)
    gh = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x/sum(gh) for x in gh]
    return gh_normalize

def gh_ta(category:str,test_data,name2genre:dict,genre_dict:dict):
    for data in tqdm(test_data,desc="Processing category data......"):
        input = data['output']
        names = re.findall(r'"([^"]+)"', input)
        for name in names:
            if name in name2genre:
                genres = name2genre[name]
            else:
                # print(f"Not exist in name2genre:{name}")
                continue
            for genre in genres:
                if genre in genre_dict:
                    genre_dict[genre] += 1/len(genres)
    gh = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x/sum(gh) for x in gh]
    return gh_normalize

def weight_dict(category:str,test_data,name2genre:dict,genre_dict:dict):
    GH_tr = gh_tr(category,test_data,name2genre,genre_dict)
    GH_ta = gh_ta(category,test_data,name2genre,genre_dict)
    weight_dict = {}
    idx = 0
    for category in genre_dict:
        weight_dict[category] = GH_tr[idx] / GH_ta[idx]
        idx += 1

    return weight_dict

def cal_weight(category:str,test_data,name2genre:dict,genre_dict:dict):
    weights = []
    w_dict = weight_dict(category,test_data,name2genre,genre_dict)
    print(f"Length of data:{len(test_data)}")
    for data in tqdm(test_data,desc="Processing category data......"):
        weight = []
        target_item = data['output'].strip("\n").strip("\"")
        if target_item in name2genre :
            genres = name2genre[target_item]
            for genre in genres:
                if genre in genre_dict:
                    weight.append(w_dict[genre])
            if len(weight)>0:
                weight = sum(weight) / len(weight)
                weights.append(weight)
            else:
                weights.append(1)
        else:
            weights.append(1)
    print(f"Length of weights:{len(weights)}")
    return weights

class IFTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = torch.mean(weights * torch.mean(loss_fct(shift_logits, shift_labels).view(weights.shape[0], -1)))

        
        return (loss, outputs) if return_outputs else loss
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("weight")
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"

        columns = [k for k in signature_columns if k in dataset.column_names]
        x = dataset.remove_columns(ignored_columns)
        return x
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "weight": element['weight']}

        signature_columns = ["input_ids", "labels", "attention_mask","weight"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

from transformers import DataCollatorWithPadding
import torch

def train(
    # path
    output_dir="",
    base_model ="",
    train_dataset="",
    valid_dataset="",
    train_sample_size:int = 1024,
    resume_from_checkpoint: str = "base_model",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    gradient_accumulation_steps: int = 1,
    batch_size: int = 8,
    num_train_epochs: int = 5,
    learning_rate: float = 2e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  
    category: str = "CDs_and_Vinyl",
    seed = 0
):
    os.environ['WANDB_PROJECT'] = wandb_project

    def formatting_prompts_func(examples):
        output_text = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]

            if len(input_text) >= 2:
                text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Input:
                {input_text}
                
                ### Response:
                {response}
                '''
            else:
                text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Response:
                {response}
                '''
            output_text.append(text)

        return output_text

    def get_train_weight(data, category:str,test_data,name2genre:dict,genre_dict:dict,w_dict):
        weight = []
        target_item = data['output'].strip("\n").strip("\"")
        if target_item in name2genre :
            genres = name2genre[target_item]
            for genre in genres:
                if genre in genre_dict:
                    weight.append(w_dict[genre])
            if len(weight)>0:
                weight = sum(weight) / len(weight)
                return {"weight":weight}
            else:
                return {"weight":1}
        else:
            return {"weight":1}


    name2genre = read_json(f"./eval/{category}/name2genre.json")
    genre_dict = read_json(f"./eval/{category}/genre_dict.json")
    w_dict = weight_dict(category,test_data=read_json(train_dataset),name2genre=name2genre,genre_dict=genre_dict)
    train_weights = cal_weight(category,test_data=read_json(train_dataset),name2genre=name2genre,genre_dict=genre_dict)

    val_sample_size = int(train_sample_size / 8)
    dataset = load_dataset('json', data_files=train_dataset)
    dataset = {"train": dataset['train'].select(range(train_sample_size+val_sample_size))}
    #weights = get_train_weight(dataset['train'],train_weights)
    dataset['train'] = dataset['train'].map(lambda x: get_train_weight(x, category,test_data=read_json(train_dataset),name2genre=name2genre,genre_dict=genre_dict,w_dict=w_dict))
    print("Features:{}".format(dataset["train"].features))
    train_val_split = dataset['train'].train_test_split(train_size=train_sample_size, test_size=val_sample_size)
    train_data = train_val_split['train']
    print("Features:{}".format(train_data.features))
    val_data = train_val_split['test']


    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = LlamaForCausalLM.from_pretrained(base_model, device_map=device_map, \
                                                  quantization_config=bnb_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    if 'Llama-3' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    if resume_from_checkpoint!="base_model":
        model = PeftModel.from_pretrained(model, resume_from_checkpoint, 
                                        is_trainable=True)
    else:
        peft_config = LoraConfig(
        inference_mode=False,
        r=64,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        # bias="none",
        task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        remove_unused_columns= True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="wandb",
        run_name=wandb_name,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        save_only_model=True,
        ddp_find_unused_parameters=False, # should set to False becuase there are no unused parameters in the forward process
    )
    trainer = IFTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=cutoff_len,
        args=training_args
        #data_collator=data_callator
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)