for category in "Goodreads"
do
    cudalist="7"
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python ./evaluate.py \
            --base_model ./output_dir/${category}/ \
            --train_file ${train_file} \
            --info_file ${info_file} \
            --category ${category} \
            --test_data_path ./temp/${category}_base/${i}.csv \
            --result_json_data ./temp/${category}_base/${i}.json \
            --length_penalty 0.0 \
            --logits_file YOUR_LOGITS_FILE_PATH
    done
    wait
    python ./code/merge.py --input_path ./temp/${category}_base --output_path ./output_dir/${category}/final_result.json
    python ./code/calc.py --path ./output_dir/${category}/final_result.json --item_path ${info_file}
done
