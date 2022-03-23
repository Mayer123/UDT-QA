#!/bin/bash

MODEL_FILE=$1
CTX_SRC=$2
OUT_FILE=$3
BATCH_SIZE=$4
SHARD_ID=$5
NUM_SHARDS=$6
GPUS=$7

mkdir -p ${OUT_FILE}/retriever_results

for ((i=0; i<$GPUS; i++));
do 
    CUDA_VISIBLE_DEVICES=$i MKL_THREADING_LAYER=GNU python generate_dense_embeddings.py  \
    model_file=${MODEL_FILE} ctx_src=${CTX_SRC} out_file=${OUT_FILE} \
    batch_size=${BATCH_SIZE} shard_id=${SHARD_ID} num_shards=${NUM_SHARDS} \
    gpu_id=$i num_gpus=$GPUS &
done

num_file=$(ls ${OUT_FILE}_shard${SHARD_ID}_gpu* | wc -l)

while [ ${num_file} -lt $GPUS ];
do
echo "Waiting for the prediction to be done"
echo "${num_file} out of $GPUS is done"
sleep 10m
num_file=$(ls ${OUT_FILE}_shard${SHARD_ID}_gpu* | wc -l)
done