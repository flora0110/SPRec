gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4
lr=0.00002
num_beams=5  # [新增] 設定 Beam Search 數量，建議 3~5
num_return_sequences=10   # 每個 Prompt 要保留幾個負樣本 (例如 3 個)
diverse_beam_search=True # 是否開啟多樣化束搜索 (True/False)
diversity_penalty=1.0    # 多樣性懲罰係數 (通常 1.0)

# Only change the parameters above if needed
for category in "Goodreads"
do
    # 注意：如果是迭代訓練，lora_weights 在後續迴圈應該要指向上一輪訓練出的 checkpoint
    lora_weights="./models/SFT_4096/${category}"
    
    # [新增] 定義 name2genre.json 的路徑，請確認你的檔案確實放在這裡
    name2genre_path="./eval/${category}/name2genre.json"
    
    output_dir="./models/SPRec/${category}_${train_sample_size}_${lr}"
    
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1"
    echo "Iterations: $its"
    echo "Train Sample Size: $train_sample_size"
    echo "Valid Sample Size: $valid_sample_size"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"
    echo "Learning Rate: $lr"
    echo "Num Beams: $num_beams"
    echo "Return Sequences: $num_return_sequences"
    echo "Diverse Beam Search: $diverse_beam_search"
    echo "Diversity Penalty: $diversity_penalty"
    echo "Seed: $seed"
    echo "Output Dir: $output_dir"


    for ((i=0 ;i<$its;i++))
    do
        echo ----------------- Iteration $i starts! -----------------
        it_output_dir="${output_dir}/it${i}/"
        dpo_train_data_path="${it_output_dir}/data/trie_${num_return_sequences}_train.jsonl"
        dpo_valid_data_path="${it_output_dir}/data/trie_${num_return_sequences}_valid.jsonl"
        sft_train_data_path="${it_output_dir}/data/sft_train.jsonl"
        sft_valid_data_path="${it_output_dir}/data/sft_valid.jsonl"
        
        mkdir -p $it_output_dir
        mkdir -p "${it_output_dir}/data"
        touch "${dpo_train_data_path}"
        touch "${dpo_valid_data_path}"
        touch "${sft_train_data_path}"
        touch "${sft_valid_data_path}"
        
        # Data Generation
        # [修改重點]
        # 1. 加入 --name2genre_path
        # 2. 加入 --num_beams
        # 3. 將 batch_size 改為變數 $batch_size (避免 Beam Search 導致 OOM)
        CUDA_VISIBLE_DEVICES=$gpu1 python ./train/data_generate_trie.py \
            --train_json_file ./data/${category}/train.json \
            --valid_json_file ./data/${category}/valid.json \
            --result_json_dpo_data_train $dpo_train_data_path \
            --result_json_dpo_data_valid $dpo_valid_data_path \
            --result_json_sft_data_train $sft_train_data_path \
            --result_json_sft_data_valid $sft_valid_data_path \
            --base_model $base_model \
            --lora_weights $lora_weights \
            --batch_size $batch_size \
            --train_sample_size $train_sample_size \
            --valid_sample_size $valid_sample_size \
            --name2genre_path $name2genre_path \
            --num_beams $num_beams \
            --num_return_sequences $num_return_sequences \
            --diverse_beam_search $diverse_beam_search \
            --diversity_penalty $diversity_penalty \
            --sh_file_path "./shell/trie_data_generate.sh"
        
    done
    echo SPRec for category ${category} has successfully completed!
done