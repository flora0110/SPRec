base_model="meta-llama/Llama-3.2-1B-Instruct" # Specify your base model here
gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4 
sample=4096
# for category in "MovieLens"  "Goodreads" "CDs_and_Vinyl" "Steam"
for category in "Goodreads"
do
    echo ---------------------- SFT for category $category starting! ---------------------- 
    train_dataset="./data/${category}/train.json"
    valid_dataset="./data/${category}/valid.json"
    output_dir="./models/SFT_${sample}/${category}"
    mkdir -p $output_dir

    # Match gpu 4 to 1, gradient_accumulation_steps 16*4=64 effective batch size
    CUDA_VISIBLE_DEVICES=$gpu1 python ./train/sft.py \
        --output_dir $output_dir\
        --base_model $base_model \
        --train_dataset $train_dataset \
        --valid_dataset $valid_dataset \
        --train_sample_size $sample \
        --wandb_project SFT_${category}_${sample} \
        --wandb_name SFT_${category}_${sample} \
        --gradient_accumulation_steps 16 \
        --batch_size 4 \
        --num_train_epochs 4 \
        --learning_rate 0.0003 \
        --cutoff_len 512 

    bash ./shell/eval_single_file.sh  $gpu1 \
                                            $base_model \
                                            $output_dir \
                                            $category \
                                            $topk
done

