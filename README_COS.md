# Chain-of-Skills: A Configurable Model for Open-domain Question Answering
The code and model for the paper "Chain-of-Skills: A Configurable Model for Open-domain Question Answering" (ACL 2023) is added. See full paper [here](https://arxiv.org/abs/2305.03130)

## Enviroments
Our environment is based on transformers 4.12.5
```
conda env create -f cos.yml
```

## Using Chain-of-Skills (COS)
Our multi-task finetuned COS model and pretrained COS model can be downloaded. Note that the pretrained model (resource key: cos.model.pretrained) only has 4 experts. 
```
python download_data.py --resource cos.model --output_dir your_output_dir 
```

To use COS on your own questions and documents, first clone this repo and set up the environment, then cd to DPR directory: 
```
import torch
from transformers import BertTokenizer
from dpr.models.hf_models_cos import HFEncoder, BertTensorizer
from dpr.models.biencoder_joint import MoEBiEncoderJoint
model_ckpt = torch.load('path/to/downloaded/checkpoint', map_location="cpu")
encoder = HFEncoder('bert-base-uncased', use_moe=True, moe_type='mod2:attn', num_expert=6)     
model = MoEBiEncoderJoint(encoder, encoder, num_expert=6, do_rerank=True, do_span=True)
model_ckpt['model_dict']['question_model.encoder.embeddings.position_ids'] = model.state_dict()['question_model.encoder.embeddings.position_ids']
model_ckpt['model_dict']['ctx_model.encoder.embeddings.position_ids'] = model.state_dict()['ctx_model.encoder.embeddings.position_ids']
model.load_state_dict(model_ckpt['model_dict'])
model.eval()
model.to('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tensorizer = BertTensorizer(tokenizer, max_length=256)

# For simple query
tensor = tensorizer.text_to_tensor('this is a question').unsqueeze(0).cuda()
segments = torch.zeros_like(tensor).cuda()
att_mask = tensorizer.get_attn_mask(tensor).cuda()
# expert_id 0 for simple query
outputs = model.question_model(tensor, segments, att_mask, expert_id=0)
q_vec = outputs[1]  # a vector of size 768

# For retrieval index
tensor = tensorizer.text_to_tensor('this is a document', title='title').unsqueeze(0).cuda()
segments = torch.zeros_like(tensor).cuda()
att_mask = tensorizer.get_attn_mask(tensor).cuda()
# expert_id 1 for retrieval index
outputs = model.ctx_model(tensor, segments, att_mask, expert_id=1)
d_vec = outputs[1]  # a vector of size 768

# For entity span proposal, it uses expert_id 5
from run_chain_of_skills_hotpot import contrastive_generate_grounding
# If you have a title, concatenate it with [SEP] token
entities = contrastive_generate_grounding(model, tensorizer, ['title [SEP] text of the document'], None, None, 128)[0]
# If not 
entities = contrastive_generate_grounding(model, tensorizer, ['this is a question with an Entity'], None, None, 128, sep_id=0)[0]
# To get indices and spans
indices = [[e[0], e[1]] for e in entities]
spans = [e[2] for e in entities]

# For entity linking
tensor = tensorizer.text_to_tensor('this is a question with an Entity').unsqueeze(0).cuda()
segments = torch.zeros_like(tensor).cuda()
att_mask = tensorizer.get_attn_mask(tensor).cuda()
# expert_id 2 for entity representation
outputs = model.question_model(tensor, segments, att_mask, expert_id=2)
# to get the first entity
rep_pos = torch.zeros_like(tensor)
rep_pos[0][indices[0][0]] = 1
rep_pos[0][indices[0][1]] = 1
ent_vec = torch.sum(outputs[0]*rep_pos.unsqueeze(-1), dim=1)/rep_pos.sum(dim=1).unsqueeze(-1)

# Entity index can be used in the same way as retrieval index, except that you need expert_id 3

# For reranking 
tensor = tensorizer.text_to_tensor('this is a question [SEP] this is a document').unsqueeze(0).cuda()
segments = torch.zeros_like(tensor).cuda()
att_mask = tensorizer.get_attn_mask(tensor).cuda()
# expert_id 4 for reranking (also for expanded retrieval)
outputs = encoder(tensor, segments, att_mask, expert_id=4)
sep_position = (tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]
rerank_score = (outputs[1][0]*outputs[0][0][sep_position]).sum(dim=-1)
```

## Evaluating (COS)
If you would like to evaluate COS on HotpotQA, OTT-QA, or Natural Questions, you can download the corresponding data, corpus and embeddings and then run inference.

For each dataset, we used their corresponding knowledge sources, and they can be downloaded (along with their embeddings) using the following commands:
```
python download_data.py --resource cos.embeds.hotpot --output_dir your_output_dir 
python download_data.py --resource cos.results.hotpot --output_dir your_output_dir 
```

Note that current code requires GPU to run, for Hotpot and OTT-QA, a GPU with >=40GB memory should be enough. For NQ, a total of >128GB memory is required, so you probably need multiple GPUs. 

To run COS inference on HotpotQA, first do question entity linking then run COS
```
python run_chain_of_skills_hotpot.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=/path/to/hotpot/data do_span=True out_file=span_output_path
python run_chain_of_skills_hotpot.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoded_ctx_files=[/path/to/hotpot_wiki_linker*] encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=span_output_path out_file=link_output_path ctx_datatsets=[/path/to/hotpot_corpus.jsonl] do_link=True 
python run_chain_of_skills_hotpot.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoded_ctx_files=[/path/to/hotpot_wiki_retriever*] encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=/path/to/hotpot/data ctx_datatsets=[/path/to/hotpot_corpus.jsonl,link_output_path] num_shards=10 batch_size=10 hop1_limit=200 hop1_keep=30 do_cos=True 
```

To run COS inference on OTT-QA
```
python run_chain_of_skills_ott.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos encoded_ctx_files=[/path/to/ott_table_original*] qa_dataset=/path/to/ott_dev_q_to_tables_with_bm25neg.json do_cos=True ctx_datatsets=[/path/to/ott_table_chunks_original.json,/path/to/ott_wiki_passages.json,[/path/to/table_chunks_to_passages*]] hop1_limit=100 hop1_keep=200 
```

To run COS inference on NQ, first do question entity linking then run COS
```
python run_chain_of_skills_hotpot.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=/path/to/nq-dev.csv do_span=True out_file=nq_dev_span
python run_chain_of_skills_nq.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=[/path/to/nq_dev_span] do_link=True encoded_ctx_files=[/path/to/nq_wiki_linker*] ctx_datatsets=[dpr_wiki] out_file=[nq_dev_links] hop1_limit=1000
python run_chain_of_skills_nq.py model_file=/path/to/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos qa_dataset=[/path/to/nq-dev.csv] do_cos=True encoded_ctx_files=[/path/to/nq_wiki_retriever*] ctx_datatsets=[dpr_wiki,/path/to/nq_dev_links.json] out_file=[nq_dev_cos_output] hop1_limit=1000
```

## Training your own COS
If you would like to train your own COS model, we provide the training data for COS, you can download them using download_data.py (resource key: cos.data). Note that you will also need to update the corresponding file paths in /DPR/conf/datasets/encoder_train_default.yaml.

To train the COS model,
```
python -m torch.distributed.launch --nproc_per_node=16 python train_dense_encoder_COS.py train=biencoder_nq  train_datasets=[hotpot_train_single,hotpot_train_expanded,hotpot_question_link_train,hotpot_link_train,hotpot_train_rerank,ott_single_train,ott_expanded_train,ott_link_train,ott_rerank_train,NQ_train,NQ_train_table,NQ_train_rerank,NQ_train_table_rerank] dev_datasets=[hotpot_dev_single,hotpot_dev_expanded,hotpot_question_link_dev,hotpot_link_dev,hotpot_dev_rerank] val_av_rank_start_epoch=35 output_dir=/output/dir train.batch_size=12 global_loss_buf_sz=1184000 train.val_av_rank_max_qs=20000 train.warmup_steps=8000 encoder.pretrained_file=/path/to/cos_pretrained_4_experts.ckpt checkpoint_file_name=dpr_biencoder encoder.use_moe=True encoder.num_expert=6 encoder.moe_type=mod2:attn train.use_layer_lr=False train.learning_rate=2e-5 encoder.encoder_model_type=hf_cos
```

After training is done, you can generate embeddings for the knowledge sources using the following command (for ctx_src, use (ott_table+ott_wiki_passage) for OTT-QA, wiki_hotpot for HotpotQA and dpr_wiki for NQ). Note that retrieval index use expert_id 1 and linking index uses expert_id 3, so they would produce different embeddings. Thus you would need to provide different target_expert argument. 
```
python generate_dense_embeddings.py model_file=your_best_model_checkpoint ctx_src=wiki_hotpot \
    out_file=your_output_location batch_size=2048 shard_id=0 num_shards=1 gpu_id=0 num_gpus=1 encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos target_expert=1 
```

## Reader Evaluation 
We provide the trained FiE reader model for NQ and OTT-QA respectively, as well as the reader data, they can be downloaded using download_data.py (resource key: cos.model.reader, cos.reader.data)

For NQ, run the following command to evaluate 
```
python train_qa_fie.py --do_predict --model_name google/electra-large-discriminator --train_batch_size 2 --gradient_accumulation_steps 2 --predict_batch_size 1 --output_dir your/output/dir --num_train_steps 5000 --use_layer_lr --layer_decay 0.9 --eval-period 500 --learning_rate 5e-5 --max_ans_len 15 --gradient_checkpointing --max_q_len 28 --max_seq_len 250 --max_grad_norm 1.0 --train_file /path/to/nq_train_reader.json --predict_file /path/to/nq_dev_reader.json --init_checkpoint /path/to/nq_fie_checkpoint_best.pt --save_prediction dev_results
```

For OTT-QA
```
python train_qa_fie.py --do_predict --model_name google/electra-large-discriminator --train_batch_size 2 --gradient_accumulation_steps 2 --predict_batch_size 1 --output_dir your/output/dir --num_train_steps 5000 --use_layer_lr --layer_decay 0.9 --eval-period 500 --learning_rate 5e-5 --max_ans_len 15 --gradient_checkpointing --num_ctx 50 --max_grad_norm 1.0 --train_file /path/to/ott_train_reader.json --predict_file /path/to/ott_dev_reader.json --init_checkpoint /path/to/ott_fie_checkpoint_best.pt --save_prediction dev_results
```

If you would like to train your own FiE reader model, just launch with the following command and add the --do_train flag (in addition to the above inference command)
```
python -m torch.distributed.launch --nproc_per_node=16 train_qa_fie.py 
```

We also provide the large path reranker and reader model for HotpotQA 
For path reranker, run the following command to evaluate 
```
python train_qa_hotpot.py --do_predict --model_name google/electra-large-discriminator --train_batch_size 8 --eval-period 1000 --predict_batch_size 2 --output_dir your/output/dir --num_train_steps 20000 --use_layer_lr --layer_decay 0.9 --learning_rate 5e-5 --para-weight 1.0 --sp-weight 0.5 --num_ctx 50 --listmle --sentlistmle --max_ans_len 30 --train_file /path/to/hotpot_train_reader.json --predict_file /path/to/hotpot_dev_reader.json --init_checkpoint /path/to/hotpot_path_reranker_checkpoint_best.pt --save_prediction dev_path_rerank_results --verbose
python train_qa_hotpot.py --do_predict --model_name google/electra-large-discriminator --train_batch_size 2 --eval-period 1000 --predict_batch_size 2  --output_dir your/output/dir --num_train_steps 20000 --use_layer_lr --layer_decay 0.9 --learning_rate 5e-5 --para-weight 0.0 --sp-weight 0.5 --num_ctx 10 --listmle --sentlistmle --max_ans_len 30 --train_file /path/to/hotpot_train_reader.json --predict_file /path/to/hotpot_dev_reader_2hops.json --init_checkpoint /path/to/hotpot_reader_checkpoint_best.pt --save_prediction dev_reader_results --verbose
```

Again, if you would like to train your own models, just lanuch with DDP and add the --do_train flag
```
python -m torch.distributed.launch --nproc_per_node=16 train_qa_hotpot.py
```

## Cite 
```
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
