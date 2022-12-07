#!/bin/bash

# Compute chainer score for OTT-QA hop1 evidences 
CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results dev_hop1_retrieved_results.json \
--passage_path ott_table_chunks_original.json --output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10 --do_tables

# Compute chainer score for OTT-QA hop2 evidences
CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results dev_hop1_retrieved_results.json \
--table_pasg_links_path path_to_ott_linker_results/table_chunks_to_passages_shard* \
--passage_path ott_wiki_passages.json --output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10

# Compute chainer score for NQ hop1 evidences 
CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results nq_full_dev.json \
--output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10 --nq

# Compute chainer score for NQ hop2 evidences
CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results nq_full_dev.json \
--output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10 --nq \
--table_pasg_links_path path_to_nq_linker_results/nq_table_chunks_to_passages_shard* --passage_path psgs_w100.tsv --nq_link


# Run chainer for OTT-QA 
python run_chainer.py --mode ott --retriever_results dev_hop1_retrieved_results.json \
--table_pasg_links_path path_to_ott_linker_results/table_chunks_to_passages_shard* \
--passage_path ott_wiki_passages.json --table_path ott_table_chunks_original.json \
--previous_cache aggregated_ott_score_cache.json --output_path your_output_dir_for_reader_data --split dev

# Run chainer for NQ 
# note that for output_path, it should be different from the input path (retriever_results path)
python run_chainer.py --mode nq --retriever_results nq_retriever_results_test.json \
--table_pasg_links_path path_to_nq_linker_results/nq_table_chunks_to_passages_shard* \
--passage_path psgs_w100.tsv --previous_cache aggregated_NQ_score_cache.json \
--output_path your_output_dir_for_reader_data 