for TASK_NAME in mnli
do
    for config in parallel scaled_parallel
    do
        for reduction_factor in 4 16 64 256
        do
            CUDA_VISIBLE_DEVICES=0 python run_glue.py \
            --model_name_or_path bert-base-uncased \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 1e-4 \
            --num_train_epochs 10.0 \
            --output_dir dev/shm/tmp/$TASK_NAME/baseline/$config/$reduction_factor \
            --overwrite_output_dir \
            --train_adapter \
            --adapter_config $config \
            --adapter_reduction_factor $reduction_factor
        done
    done
done