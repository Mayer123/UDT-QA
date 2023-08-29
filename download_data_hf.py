import argparse
import os
from huggingface_hub import hf_hub_download
import joblib

RESOURCES_MAP = {
    "cos.model": {'ID': 'kaixinm/COS', 'filenames': ['models/cos_nq_ott_hotpot_finetuned_6_experts.ckpt']},
    "cos.model.pretrained": {'ID': 'kaixinm/COS', 'filenames': ['models/cos_pretrained_4_experts.ckpt']},
    "cos.model.reader.ott": {'ID': 'kaixinm/COS', 'filenames': ['models/ott_fie_checkpoint_best.pt']},
    "cos.model.reader.nq": {'ID': 'kaixinm/COS', 'filenames': ['models/nq_fie_checkpoint_best.pt']},
    "cos.model.reader.hotpot": {'ID': 'kaixinm/COS', 'filenames': ['models/hotpot_path_reranker_checkpoint_best.pt', 'models/hotpot_reader_checkpoint_best.pt']},
    "cos.data.hotpot": {'ID': 'kaixinm/COS', 'filenames': ['data/hotpot_data.zip']},
    "cos.data.nq": {'ID': 'kaixinm/COS', 'filenames': ['data/nq_data.zip']},
    "cos.data.ott": {'ID': 'kaixinm/COS', 'filenames': ['data/ott_data.zip']},
    "udtqa.knowledge.raw_table": {'ID': 'kaixinm/UDT-QA', 'filenames': ['knowledge_sources/tables_raw.zip']},
    "udtqa.knowledge.V_table": {'ID': 'kaixinm/UDT-QA', 'filenames': ['knowledge_sources/tables_V.zip']},
    "udtqa.knowledge.V_kb": {'ID': 'kaixinm/UDT-QA', 'filenames': ['knowledge_sources/WD_graphs_V.zip']},
    "core.knowledge": {'ID': 'kaixinm/CORE', 'filenames': ['data/knowledge.zip'], 'description': 'The OTT-QA table sets and OTT-QA Wiki passages'},
    "core.model.retriever": {'ID': 'kaixinm/CORE', 'filenames': ['models/core_joint_retriever.ckpt']},
    "core.model.linker": {'ID': 'kaixinm/CORE', 'filenames': ['models/core_span_proposal.ckpt', 'models/core_table_linker.ckpt']},
    "core.model.reader": {'ID': 'kaixinm/CORE', 'filenames': ['models/pytorch_model.bin', 'models/config.json']},
    "core.data.retriever": {'ID': 'kaixinm/CORE', 'filenames': ['data/retriever_data.zip']},
    "core.data.retriever_ott_results": {'ID': 'kaixinm/CORE', 'filenames': ['data/retriever_results/ott_results.zip']},
    "core.data.retriever_nq_results": {'ID': 'kaixinm/CORE', 'filenames': ['data/retriever_results/nq_results.zip']},
    "core.data.linker": {'ID': 'kaixinm/CORE', 'filenames': ['data/linker_data.zip']},
    "core.data.reader_ott": {'ID': 'kaixinm/CORE', 'filenames': ['data/reader/ott_reader.zip']},
    "core.data.reader_nq": {'ID': 'kaixinm/CORE', 'filenames': ['data/reader/nq_reader.zip']},
    "cos.reader.data.hotpot": {'ID': 'kaixinm/COS', 'filenames': ['data/results/hotpot_reader.zip']},
    "cos.reader.data.nq": {'ID': 'kaixinm/COS', 'filenames': ['data/results/nq_reader.zip']},
    "cos.reader.data.ott": {'ID': 'kaixinm/COS', 'filenames': ['data/results/ott_reader.zip']},
}

def download(resource, output_dir):
    REPO_ID, files = RESOURCES_MAP[resource]['ID'], RESOURCES_MAP[resource]['filenames']
    for f in files:
        hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=output_dir)

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