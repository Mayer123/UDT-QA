# Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge
The code and data for the paper "Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge" (Findings of EMNLP 2022) is added. See full paper [here](https://arxiv.org/abs/2210.12338)

## Enviroments
Our retriever and linker uses the same environment as DPR, except that we also require faiss-gpu
```
cd DPR
pip install .
conda install -c conda-forge faiss-gpu
```

## Retriever data and models 
We also provide the retriever results for both OTT-QA and NQ, where all questions are retrieved from the corresponding knowledge sources. You can download them with resource key: core.data.retriever_ott_results for OTT-QA and core.data.retriever_nq_results for NQ. Thus you can skip the retriever step with the provided results.

To download trained retriever models and the encoded knowledge sources, use the download_data.py, we provide the joint retriever model trained on NQ and OTT-QA
```
python download_data.py --resource core.model.retriever --output_dir your_output_dir 
```
For NQ, we use the same knowledge sources as in UDT-QA raw setting ([text](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/psgs_w100.tsv), [tables](https://msrdeeplearning.blob.core.windows.net/udq-qa/data/tables/all_raw_table_chunks_for_index.json)), for OTT-QA, we use the original table sets released by Chen et al. [ott-qa-tables](https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/knowledge/ott_table_chunks_original.json) (These can be downloaded with wget)

To run retriever inference on OTT-QA (the qa_dataset files are the same ones as used to train the retriever or test.blind.json)
```
CUDA_VISIBLE_DEVICES=0 python dense_retrieve_link.py model_file=core_joint_retriever.ckpt encoded_ctx_files=[download_dir/ott_tables_original*] \
qa_dataset=/home/kaixinm/kaixinm/UDT-QA/DPR/data/CORE/ott_dev_q_to_tables_with_bm25neg.json do_retrieve=True
```
Because OTT-QA index is mucher smaller than NQ, thus we can it on one GPU. To run retriever inference on NQ (cpu-only), please follow the instructions of UDT-QA. 

## Training your own retriever 
If you would like to train your own retriever model, you can download the retriever training data using download_data.py (resource key: core.data.retriever). Additionaly, you will also need the NQ training file (nq-train.json from resource key: data.retriever.nq_raw_tables). Note that you will also need to update the corresponding file paths in /DPR/conf/datasets/encoder_train_default.yaml.

To train the OTT-QA and NQ joint retriever,
```
python -m torch.distributed.launch --nproc_per_node=2 train_dense_encoder_core.py train=biencoder_nq \
train_datasets=[nq_train_bm25,  nq_retriever_table_train,ott_retriever_train] dev_datasets=[ott_retriever_dev] \
train.batch_size=64 global_loss_buf_sz=1184000 output_dir=your_output_dir checkpoint_file_name=dpr_biencoder
```
After training is done, you can generate embeddings for the knowledge sources using the following command (for ctx_src, use ott_table for OTT-QA and dpr_wiki+raw_table for NQ)
```
python generate_dense_embeddings.py model_file=your_best_model_checkpoint ctx_src=ott_table \
    out_file=your_output_location batch_size=2048 shard_id=0 num_shards=1 gpu_id=0 num_gpus=1 
```

## Linker data and models
We provide the linker results for both OTT-QA and NQ, where all entities in every table chunk are linked to their corresponding wiki passages. You can download them with resource key: core.data.linker_ott_results for OTT-QA and core.data.linker_nq_results for NQ.

Note that we have generated links for all tables in the OTT-QA dataset, but for NQ we only generated links for tables that are retrieved by the retriever. This is because the NQ table sets are much larger and we want to save time and space. Hence that you need a preprocessing step to extract the retrieved tables from NQ results, if you want to run linker inference on your own. 

If you would like to run linker inference yourself, we also provide the trained span proposal model and linking model, you can download them with the resource key: core.model.linker
To run linker inference on OTT-QA, first run span proposal model then run linking model 
```
CUDA_VISIBLE_DEVICES=0 python dense_retrieve_link.py model_file=core_span_proposal.ckpt \
qa_dataset=download_dir/ott_table_chunks_original.json do_span=True label_question=True 

CUDA_VISIBLE_DEVICES=0 python dense_retrieve_link.py model_file=core_table_linker.ckpt encoded_ctx_files=[download_dir/ott_wiki*] \
qa_dataset=download_dir/all_table_chunks_span_prediction.json do_link=True
```

## Training your own linker
If you would like to train your own linker, we also release the linker training data (resource key: core.data.linker). To train the linker, you will need to seperately train the span proposal model and the linking model. 
```
CUDA_VISIBLE_DEVICES=0 python train_table_linker.py train=biencoder_nq train_datasets=[ott_linker_train] dev_datasets=[ott_linker_dev] \
train.batch_size=64 global_loss_buf_sz=1184000 output_dir=your_output_dir_span_proposal checkpoint_file_name=dpr_biencoder \
encoder.encoder_model_type=hf_table_link label_question=True

CUDA_VISIBLE_DEVICES=0 python train_table_linker.py train=biencoder_nq train_datasets=[ott_linker_train] dev_datasets=[ott_linker_dev] \
train.batch_size=64 global_loss_buf_sz=1184000 output_dir=your_output_dir_linking checkpoint_file_name=dpr_biencoder \
train.num_train_epochs=100 encoder.encoder_model_type=hf_table_link
```
After the training is finished, you can generate embeddings in the same way as retriever. 

## Chainer inference 
The chainer model requires transformers 4.x, so you probably need to run it in a new environment. We used transformers 4.12.5 for our experiments.
The chainer consists of 2 steps, a) computing the rerank scores using T0-3B model b) building chains and reorder them to form reader data. Please refer to Chainer/run_rerank.sh for the detailed commands.

Note that since the T0-3B model is very computationally heavy, we also provide the precomputed score caches for OTT-QA and NQ dataset, you can download them with resource key: core.data.chainer_ott_cache and core.data.chainer_nq_cache. During inference, you can provide these cache files to the argument --previous_cache so that the model will not recompute the scores for the same question-document pairs. 

## Reader 
We provide the joint reader model trained on OTT-QA and NQ dataset (resource key: core.model.reader) and the reader training data (resource key: core.data.reader_ott and core.data.reader_nq). Please see the instructions under fid_reader/ folder to evaluate the trained model or train your own reader. You should be able to get 53.9 EM on NQ test set and 49.0 EM on OTT-QA dev set with our provided model.

## Cite 
```
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
```
