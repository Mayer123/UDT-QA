# Update 08/2023

**\*\*\*\*\* Migrating our models and data to HuggingFace \*\*\*\*\***

Due to the restricted access of the Azure blob storage, we are migrating our models and data to HuggingFace. Please use the following command to download our models and data. The resource keys are the same as in download_data.py.

```
python download_data_hf.py --resource resource_key --output_dir your_output_dir 
```

# Update 06/2023

**\*\*\*\*\* Adding code for [COS](https://github.com/Mayer123/UDT-QA/blob/main/README_COS.md) \*\*\*\*\***

The code and model for the paper "Chain-of-Skills: A Configurable Model for Open-domain Question Answering" (ACL 2023) is added. See full paper [here](https://arxiv.org/abs/2305.03130)

# Update

**\*\*\*\*\* Adding code for [CORE](https://github.com/Mayer123/UDT-QA/blob/main/README_CORE.md) \*\*\*\*\***

The code and data for the paper "Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge" (Findings of EMNLP 2022) is added. See full paper [here](https://arxiv.org/abs/2210.12338)

# Open Domain Question Answering with A Unified Knowledge Interface
This repository contains the code and data for the paper "Open Domain Question Answering with A Unified Knowledge Interface" (ACL 2022). See full paper [here](https://arxiv.org/abs/2110.08417)

Note that our code is adpated from [DPR repo](https://github.com/facebookresearch/DPR) and [plms-graph2text](https://github.com/UKPLab/plms-graph2text)

## Knowledge Sources 
We provide the following knowledge resources, you can copy the link below and do "wget link" to download them, whole: each table is a unit, chunked: each table is splited into chunks of approximately 100 tokens
Source | Format | Whole | Chunked 
---|---|---|---
Table | Raw | [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/tables/all_raw_tables.json) | [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/tables/all_raw_table_chunks_for_index.json)
Table | Verbalized | [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/tables/all_verbalized_tables.json) | [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/tables/all_verbalized_table_chunks_for_index.json)
KB | Verbalized | [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/kb/grouped_WD_graphs.jsonl)| [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/kb/verbalized_WD_graphs_for_index.tsv)

## Enviroments
Our verbalizer and retriever require different environments so you should build two separate environments, for DPR part
```
cd DPR
pip install .
```
For verbalizer, the code has been tested on Python 3.8, Pytorch 1.7.1 and Transformers 3.3.1, pytorch-lightning 0.9.0, you can install the required packages by 
```
cd Verbalizer
pip install -r requirements.txt
```

## Verbalizing your own knowledge resource 
You can first download our trained verbalizer using this [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/models/verbalizer/t5_large_verbalizer_T-F_ID-T.ckpt). 
Then you would need to prepare your data to the format similar to Verbalizer/data/test.source
Then run verbalizer with 
```
bash generate.sh your_data_folder verbalizer_ckpt output_dir 1 your_data_filename 10 1 0 
```
See generate.sh for more information on the arguments. In short, our verbalizer uses data parallel, i.e. a process is spawned to use 1 GPU to work on 1 shard of data independently.
After generation is done, run following for beam selection 
```
python post_processing.py --verbalizer_output output_file_from_previous_step --verbalizer_input your_data_filename 
```
If you would like to re-train a new verbalizer, run 
```
bash train.sh data t5-large T-F_ID-T 1 5
```

## Retriever data and models 
To download trained retriever models and the encoded knowledge sources, use the download_data.py 
```
python download_data.py --resource model.retriever.nq_v_tables --output_dir your_output_dir 
```
Check the resource_map in download_data.py for more information. We provide the following 3 models on NQ (metrics on NQ test set)
Knowledge Sources | Format | Resource key | R20 | R100 | EM 
---|---|---|---|---|---
Text+Table | Raw | nq_raw_tables | 86.9 | 91.9 | 54.7
Text+Table | Verbalized | nq_v_tables | 87.0 | 91.7 | 55.2
Text+Table+KB | Verbalized | nq_v_all | 85.6 | 91.2 | 55.1

If you do not already have the chunked wikipedia passages, you can download it using this [link](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/psgs_w100.tsv). It's the same copy provided by DPR repo.
To run inference on the retriever 
```
python dense_retriever.py model_file=downloaded_model_file qa_dataset=[nq_test] \
    ctx_datatsets=[dpr_wiki,verbalized_table] encoded_ctx_files=[download_path/wiki*,download_path/verbalized*] \
    out_file=[output_location] 
```
Note that you will need to update the file paths in the /DPR/conf/ctx_sources/default_sources.yaml and /DPR/conf/datasets/retriever_default.yaml
If you have small CPU RAM, you can add the validation_workers=1 argument. 

## Training your own retriever 
If you would like to train your own retriever model, you can download the retriever training data using download_data.py,
we provide the training data for the above 3 settings on NQ and for WebQ with all verbalized knowledge. 
To run retriever training,
```
python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py train=biencoder_nq  
    train_datasets=[nq_train_bm25,nq_train_dpr_v_tab_index,nq_train_v_tab_bm25,nq_train_v_tab_dpr] dev_datasets=[nq_dev_v_tab_index]
    output_dir=your_output_dir
    checkpoint_file_name=dpr_biencoder
```
For more information on the train datasets keys, check the /DPR/conf/datasets/encoder_train_default.yaml, note that you will also need to update the corresponding file paths.
Task | Knowledge Sources | Format | Resource key | Train set keys | Dev set keys 
---|---|---|---|---|---
NQ | Text+Table | Raw | nq_raw_tables | nq_train_bm25,nq_train_dpr_raw_tab_index,nq_train_raw_tab_bm25,nq_train_raw_tab_dpr | nq_dev_raw_tab_index 
NQ | Text+Table | Verbalized | nq_v_tables | nq_train_bm25,nq_train_dpr_v_tab_index,nq_train_v_tab_bm25,nq_train_v_tab_dpr | nq_dev_v_tab_index
NQ | Text+Table+KB | Verbalized | nq_v_all | nq_train_bm25,nq_train_dpr_v_all_index,nq_train_v_tab_bm25,nq_train_v_kb_bm25 | nq_dev_v_all_index
WebQ | Text+Table+KB | Verbalized | webq_v_all | webq_train,webq_train_v_kb_bm25,webq_train_v_tab_bm25,webq_train_dpr_v_all_index | webq_dev_v_all_index

Training on NQ takes about 1.5 days to finish. After training is done, you can generate the encoded embeddings with following,
```
python generate_dense_embeddings.py model_file=your_best_model_checkpoint ctx_src=verbalized_table \
    out_file=your_output_location batch_size=2048 shard_id=0 num_shards=1 gpu_id=0 num_gpus=1 
```
This script runs with data parallel, i.e. each processing will work on 1 shard of data independently, 
thus you can run multiple processes to work on different pieces of data, if you have multiple GPUs. 

## Training your own reader 
If you are only interested in training a reader model, we also provide the reader training data (i.e. retriever results) for NQ in above 3 settings. 
Again, you can download them using download_data.py. Note that we did not use the reader model implemented under the DPR folder and we used [UnitedQA](https://github.com/hao-cheng/UnitedQA-E) instead. 

## Cite 
```
@inproceedings{ma-etal-2022-open,
    title = "Open Domain Question Answering with A Unified Knowledge Interface",
    author = "Ma, Kaixin  and
      Cheng, Hao  and
      Liu, Xiaodong  and
      Nyberg, Eric  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.113",
    doi = "10.18653/v1/2022.acl-long.113",
    pages = "1605--1620",
}

@inproceedings{ma-etal-2022-open-domain,
    title = "Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge",
    author = "Ma, Kaixin  and
      Cheng, Hao  and
      Liu, Xiaodong  and
      Nyberg, Eric  and
      Gao, Jianfeng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.392",
    pages = "5360--5374",
}

@inproceedings{ma-etal-2023-chain,
    title = "Chain-of-Skills: A Configurable Model for Open-Domain Question Answering",
    author = "Ma, Kaixin  and
      Cheng, Hao  and
      Zhang, Yu  and
      Liu, Xiaodong  and
      Nyberg, Eric  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.89",
    doi = "10.18653/v1/2023.acl-long.89",
    pages = "1599--1618",
}

```
