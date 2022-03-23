#!/bin/bash


DATA_DIR=$1
CHECK_POINT=$2
OUTPUT_DIR=$3
GPUS=$4
TEST_NAME=$5
NUM_SEQ=$6
NUM_SHARDS=$7
SHARD_ID=$8

for ((i=0; i<$GPUS; i++));
do 
    CUDA_VISIBLE_DEVICES=$i python finetune.py \
    --data_dir=${DATA_DIR} \
    --task graph2text \
    --model_name_or_path=t5-large \
    --eval_batch_size=16 \
    --gpus=1 \
    --output_dir=$OUTPUT_DIR \
    --checkpoint=$CHECK_POINT \
    --max_source_length=384 \
    --max_target_length=384 \
    --val_max_target_length=384 \
    --test_max_target_length=384 \
    --eval_max_gen_length=384 \
    --do_predict \
    --eval_beams=$NUM_SEQ \
    --progress_bar_refresh_rate 1 \
    --test_name=$TEST_NAME \
    --num_returned_sequence=$NUM_SEQ \
    --num_shards=$NUM_SHARDS \
    --shard_id=$SHARD_ID --gpu_id=$i --total_num_gpus=$GPUS 
done