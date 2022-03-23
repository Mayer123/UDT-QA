#!/bin/bash

DATA_DIR=$1
MODEL=$2
OUTNAME=$3
GPUS=$4
EPOCHS=$5

export OUTPUT_DIR_NAME=outputs/${MODEL}_${OUTNAME}
export OUTPUT_DIR=${DATA_DIR}/${OUTPUT_DIR_NAME}

mkdir -p $OUTPUT_DIR
export OMP_NUM_THREADS=10

python finetune.py \
--data_dir=${DATA_DIR} \
--learning_rate=3e-5 \
--num_train_epochs=$EPOCHS \
--task graph2text \
--model_name_or_path=${MODEL} \
--train_batch_size=8 \
--eval_batch_size=16 \
--early_stopping_patience 15 \
--gpus=$GPUS \
--output_dir=$OUTPUT_DIR \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_train --do_predict \
--eval_beams 3 \
--progress_bar_refresh_rate 100 --save_top_k=$EPOCHS
