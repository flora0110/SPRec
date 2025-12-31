gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
neg_num=$3
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4
lr=0.00002
seed=42
# Only change the parameters above if needed
for category in "Goodreads"
do
    lora_weights="./models/SFT_4096/${category}"
    output_dir="./models/SPRec/${category}_${train_sample_size}_${lr}"
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1"
    echo "Iterations Neg: $its_neg"
    echo "Train Sample Size: $train_sample_size"
    echo "Valid Sample Size: $valid_sample_size"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"
    echo "Learning Rate: $lr"


    for ((i=0;i<$its;i++))
    do
        echo ----------------- Iteration$i starts! -----------------
        it_output_dir="${output_dir}/it${i}/"
        sdpo_train_data_path="${it_output_dir}/data/dpo_train_RN.jsonl"
        sdpo_valid_data_path="${it_output_dir}/data/dpo_valid_RN.jsonl"
        sft_train_data_path="${it_output_dir}/data/sft_train.jsonl"
        sft_valid_data_path="${it_output_dir}/data/sft_valid.jsonl"
        mkdir -p $it_output_dir
        mkdir -p "${it_output_dir}/data"
        touch "${sdpo_train_data_path}"
        touch "${sdpo_valid_data_path}"
        # Data Generation
        # CUDA_VISIBLE_DEVICES=$gpu1 python ./train/data_generate_sdpo.py \
        #     --n_negatives $neg_num \
        #     --result_json_sft_data_train $sft_train_data_path \
        #     --result_json_sft_data_valid $sft_valid_data_path \
        #     --result_json_sdpo_data_train $sdpo_train_data_path \
        #     --result_json_sdpo_data_valid $sdpo_valid_data_path \
        #     --seed $seed \
        #     --train_sample_size $train_sample_size \
        #     --valid_sample_size $valid_sample_size \
        # # SFT
        # wandb_name="iteration${i}_SFT"
        # SFT_path="${it_output_dir}SFT"
        # mkdir -p $SFT_path
        # CUDA_VISIBLE_DEVICES=$gpu1 python ./train/sft.py \
        #     --resume_from_checkpoint $lora_weights \
        #     --output_dir $SFT_path \
        #     --base_model $base_model \
        #     --train_dataset $sft_train_data_path \
        #     --valid_dataset $sft_valid_data_path \
        #     --train_sample_size $train_sample_size \
        #     --wandb_project $wandb_project \
        #     --wandb_name $wandb_name \
        #     --gradient_accumulation_steps 16 \
        #     --batch_size $batch_size \
        #     --num_train_epochs 1 \
        #     --learning_rate $lr \
        #     --cutoff_len 512 \
        # # Evaluate SFT model
        # lora_weights=$SFT_path
        # bash ./shell/eval_single_file.sh  $gpu1 \
        #                                 $base_model \
        #                                 $lora_weights \
        #                                 $category
        # SDPO
        DPO_path="${it_output_dir}DPO_RN1/"
        mkdir -p $DPO_path
        CUDA_VISIBLE_DEVICES=$gpu1 python ./train/dpo.py \
            --train_dataset $sdpo_train_data_path \
            --val_dataset $sdpo_valid_data_path \
            --output_dir $DPO_path \
            --base_model $base_model \
            --resume_from_checkpoint $lora_weights \
            --batch_size 2 \
            --gradient_accumulation_steps 16 \
            --learning_rate $lr \
            --cutoff_len 512 \
            --num_epochs 1 
        # Evaluate DPO model
        lora_weights=$DPO_path
        bash ./shell/eval_single_file.sh  $gpu1 \
                                        $base_model \
                                        $lora_weights \
                                        $category 
        
    done
    echo SDPO for category ${category} has successfully completed!
done