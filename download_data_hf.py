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