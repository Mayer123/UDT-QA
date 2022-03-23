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

## Cite 
```
@misc{ma2021open,
    title={Open Domain Question Answering with A Unified Knowledge Interface},
    author={Kaixin Ma and Hao Cheng and Xiaodong Liu and Eric Nyberg and Jianfeng Gao},
    year={2021},
    eprint={2110.08417},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
