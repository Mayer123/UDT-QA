import argparse
import os
import pathlib
import wget

RESOURCES_MAP = {
    "data.retriever.nq_raw_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train_raw_table_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train_raw_table_pos_dpr_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train_text_pos_dpr_neg_raw_index.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-dev_text_or_raw_table_pos_dpr_neg.json"],
        "desc": "Retriever training data for NQ with text+raw tables",
    },
    "data.retriever.nq_v_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_tables/nq-train_v_table_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_tables/nq-train_v_table_pos_dpr_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_tables/nq-train_text_pos_dpr_neg_v_index.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_tables/nq-dev_text_or_v_table_pos_dpr_neg.json"],
        "desc": "Retriever training data for NQ with text+verbalized tables",
    },
    "data.retriever.nq_v_all": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_raw_tables/nq-train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_tables/nq-train_v_table_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_all/nq-train_kb_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_all/nq-train_text_table_kb_pos_dpr_neg_v_index_all.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/nq_v_all/nq-dev_text_table_kb_pos_dpr_neg_v_index_all.json"],
        "desc": "Retriever training data for NQ with text+verbalized tables+verbalized kb",
    },
    "data.retriever.webq_v_all": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/webq_v_all/webq-train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/webq_v_all/webq-train_v_kb_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/webq_v_all/webq-train_v_table_pos_bm25_neg.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/webq_v_all/webq-train_text_table_kb_pos_dpr_neg_v_index_all.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/retriever/webq_v_all/webq-dev_text_table_kb_pos_dpr_neg_v_index_all.json"],
        "desc": "Retriever training data for WebQ with text+verbalized tables+verbalized kb",
    },
    "data.retriever.qas.nq": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq-train.csv",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq-dev.csv",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq-test.csv",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq_table_answerable_train.jsonl",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq_table_answerable_dev.jsonl",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/nq_table_answerable_test.jsonl"],
        "desc": "Retriever inference data for NQ",
    },
    "data.retriever.qas.webq": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/webq-train.jsonl",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/webq-dev.jsonl",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/inference_data/webq-test.csv"],
        "desc": "Retriever inference data for WebQ",
    },
    "data.reader.nq_raw_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_raw_tables_reader/train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_raw_tables_reader/dev.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_raw_tables_reader/test.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_raw_tables_reader/train_gold_info.json"],
        "desc": "Reader training data for NQ with text+raw tables",
    },
    "data.reader.nq_v_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_tables_reader/train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_tables_reader/dev.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_tables_reader/test.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_tables_reader/train_gold_info.json"],
        "desc": "Reader training data for NQ with text+verbalized tables",
    },
    "data.reader.nq_v_all": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_all_reader/train.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_all_reader/dev.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_all_reader/test.json",
                  "https://msrdeeplearning.blob.core.windows.net/udq-qa/data/reader/nq_v_all_reader/train_gold_info.json"],
        "desc": "Reader training data for NQ with text+verbalized tables+verbalized kb",
    },
    "model.retriever.nq_raw_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_raw/dpr_biencoder.ckpt"] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_raw/encoded_index/wiki_passages_shard{i}.pkl" for i in range(16)] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_raw/encoded_index/raw_tables_shard{i}.pkl" for i in range(5)],
        "desc": "Retriever trained on NQ with text+raw tables and encoded knowledge sources",
    },
    "model.retriever.nq_v_tables": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_verbalized/dpr_biencoder.ckpt"] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_verbalized/encoded_index/wiki_passages_shard{i}.pkl" for i in range(16)] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_verbalized/encoded_index/verbalized_tables_shard{i}.pkl" for i in range(5)],
        "desc": "Retriever trained on NQ with text+verbalized tables and encoded knowledge sources",
    },
     "model.retriever.nq_v_all": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_and_kb_verbalized/dpr_biencoder.ckpt"] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_and_kb_verbalized/encoded_index/wiki_passages_shard{i}.pkl" for i in range(16)] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_and_kb_verbalized/encoded_index/verbalized_tables_shard{i}.pkl" for i in range(5)] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/models/retriever/nq_with_all_tables_and_kb_verbalized/encoded_index/verbalized_kb_shard{i}.pkl" for i in range(4)],
        "desc": "Retriever trained on NQ with text+verbalized tables+verbalized kb and encoded knowledge sources",
    },
    "core.model.retriever": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/models/core_joint_retriever.ckpt"] +
                 [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/encoded_ctx/dpr_wiki_shard{i}_gpu0" for i in range(4)] +
                 ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/encoded_ctx/dpr_wiki_shard2_gpu1",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/encoded_ctx/ott_tables_original_shard0_gpu0",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/encoded_ctx/udtqa_tables_shard1_gpu0",],
        "desc": "Joint Retriever trained on NQ and OTT-QA and encoded knowledge sources",
    },
    "core.data.retriever": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/OTT-QA/ott_train_q_to_tables_with_bm25neg.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/OTT-QA/ott_dev_q_to_tables_with_bm25neg.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/NQ/train_NQ_q_to_tables_with_bm25neg_merged1.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever/OTT-QA/test.blind.json"],
        "desc": "Joint Retriever training data and ott test inference data",
    },
    "core.data.retriever_ott_results": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/OTT-QA/train_hop1_retrieved_results.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/OTT-QA/dev_hop1_retrieved_results.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/OTT-QA/test_hop1_retrieved_results.json"],
        "desc": "Joint Retriever ott results",
    },
    "core.data.retriever_nq_results": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/NQ/nq_full_train.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/NQ/nq_full_dev.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/CORE/data/retriever_results/NQ/nq_full_test.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/retriever_results/NQ/nq_table_answerable_train_fullindex.json"],
        "desc": "Joint Retriever nq results",
    },
    "core.model.linker": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/models/core_span_proposal.ckpt",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/models/core_table_linker.ckpt",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/linker/OTT-QA/ott_wiki_text_shard0_gpu0",],
        "desc": "Linker model trained on OTT-QA dataset and encoded OTT wiki passages",
    },
    "core.data.linker": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/linker/ott_table_linker_dev_with_bm25neg.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/linker/ott_table_linker_train_with_bm25neg.json",],
        "desc": "Linker training data",
    },
    "core.data.linker_ott_results": {
        "links": [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/linker/OTT-QA/table_chunks_to_passages_shard{i}_of_10.json"
                 for i in range(10)],
        "desc": "Linking results on OTT-QA table sets",
    },
    "core.data.linker_nq_results": {
        "links": [f"https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/linker/NQ/nq_table_chunks_to_passages_shard{i}.json" for i in range(2)],
        "desc": "Linking results on NQ table sets",
    },
    "core.model.reader": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/models/reader/pytorch_model.bin",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/models/reader/config.json"],
        "desc": "Reader model trained on OTT-QA and NQ dataset",
    },
    "core.data.reader_ott": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/OTT-QA/ott_train_reader.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/OTT-QA/ott_dev_reader.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/OTT-QA/ott_test_reader.json"],
        "desc": "Reader data for OTT-QA dataset",
    },
    "core.data.reader_nq": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/NQ/nq_merged_train_reader.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/NQ/nq_dev_reader.json",
                 "https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/reader/NQ/nq_test_reader.json"],
        "desc": "Reader data for NQ dataset",
    },
    "core.data.chainer_ott_cache": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/chainer/aggregated_ott_score_cache.json"],
        "desc": "Chainer score cache for OTT-QA dataset",
    },
    "core.data.chainer_nq_cache": {
        "links": ["https://msrdeeplearning.blob.core.windows.net/udq-qa/CORE/data/chainer/aggregated_NQ_score_cache.json"],
        "desc": "Chainer score cache for NQ dataset",
    },
}

def download_resource(link: str, resource_key: str, out_dir: str):
    print ("Requested resource from %s", link)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    print ("Download root_dir %s", root_dir)

    save_root = os.path.join(
        root_dir, "downloads", *path_names
    )  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file = os.path.abspath(
        os.path.join(save_root, link.split('/')[-1])
    )
    print ("File to be downloaded as %s", local_file)

    if os.path.exists(local_file):
        print ("File already exist %s", local_file)
        return local_file

    wget.download(link, out=local_file)

    print ("Downloaded to %s", local_file)
    return local_file

def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print ("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    links = download_info["links"]

    data_files = []

    for i, url in enumerate(links):
        local_file = download_resource(
            url,
            resource_key,
            out_dir,
        )
        data_files.append(local_file)
   
    return data_files

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print ("Resource key=%s  :  %s" % (k, v["desc"]))

if __name__ == "__main__":
    main()