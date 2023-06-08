import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from collections import Counter
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T

from dpr.models import init_biencoder_components
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
import faiss
from dense_retriever import validate
from tqdm import tqdm
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer
from run_chain_of_skills_hotpot import set_up_encoder, generate_question_vectors, rerank_hop1_results, generate_entity_queries
import csv
logger = logging.getLogger()
setup_logger(logger)

def prepare_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    return merged_data

def chain_of_skills(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg)

    all_passages = {}
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets[:-1]:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        ctx_sources.append(ctx_src)
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)

    for di, ds_key in enumerate(cfg.qa_dataset):
        if 'dev' in ds_key:
            split = 'dev'
        elif 'train' in ds_key:
            split = 'train'
        else:
            split = 'test'
        logger.info("qa_dataset: %s", ds_key)
        data = retrieve(cfg, encoder, tensorizer, gpu_index_flat, doc_ids, all_passages, ds_key)
        #data = json.load(open(ds_key, 'r'))
        shard_id = int(cfg.shard_id)
        shard_size = int(len(data)/int(cfg.num_shards))
        print ('shard size', shard_size)
        if shard_id != int(cfg.num_shards)-1:
            start = shard_id*shard_size
            end = (shard_id+1)*shard_size
        else:
            start = shard_id*shard_size
            end = len(data)
        print ('working on start', start, 'end', end)
        data = data[start:end]
        linker_results = json.load(open(cfg.ctx_datatsets[-1], 'r'))
        linker_d = {}
        for sample in linker_results:
            linker_d[sample['question']] = sample['ctxs']
        r20 = [0]*5
        r100 = [0]*5
        num_unique = 0
        num_ql = 0
        num_r1 = 0
        for si, sample in enumerate(tqdm(data)):
            sample['ctxs'] = sample['ctxs'][:1000]
            for ctx in sample['ctxs']:
                ctx['score'] = float(ctx['score'])
                ctx['source'] = 'r1'
            hop1_ids = {ctx['id']:_ for _, ctx in enumerate(sample['ctxs'])}
            all_ctxs = [ctx for ctx in sample['ctxs']]
            max_r1_score = max([float(ctx['score']) for ctx in sample['ctxs']])
            if sample['question'] in linker_d and len(linker_d[sample['question']]) > 0:
                max_ql_score = max([ctx['score'] for ctx in linker_d[sample['question']]])
                max_ql_score = max(max_ql_score, max_r1_score)
                for ctx in linker_d[sample['question']]:
                    ctx['score'] = ctx['score'] / max_ql_score * max_r1_score
                    ctx['source'] = 'ql'
                    if ctx['id'] not in hop1_ids:
                        hop1_ids[ctx['id']] = len(all_ctxs)
                        if 'text' not in ctx:
                            ctx['text'] = all_passages[ctx['id']][0]
                        all_ctxs.append(ctx)
                    else:
                        idx = hop1_ids[ctx['id']]
                        all_ctxs[idx]['score'] = max(all_ctxs[idx]['score'], ctx['score'])
            num_unique += len(all_ctxs)
            all_ctxs = sorted(all_ctxs, key=lambda x: x['score'], reverse=True)
            all_ctxs = all_ctxs[:1000]
            num_ql += len([ctx for ctx in all_ctxs if ctx['source'] == 'ql'])
            num_r1 += len([ctx for ctx in all_ctxs if ctx['source'] == 'r1'])
            scores = rerank_hop1_results(encoder, tensorizer, [[sample['question'] + ' [SEP] ' + b['title'] + ', ' +  b['text'] for b in all_ctxs]], 1, expert_id=4, silence=True)[0]
            if si == 0:
                print (sample['question'] + ' [SEP] ' + all_ctxs[0]['title'] + ', ' +  all_ctxs[0]['text'])
                print (sample['question'] + ' [SEP] ' + all_ctxs[1]['title'] + ', ' +  all_ctxs[1]['text'])
            for i, ctx in enumerate(all_ctxs):
                ctx['rerank score'] = scores[i]
            if 'nq' in ds_key:
                w = 0.5
            else:
                w = 1
            all_ctxs = sorted(all_ctxs, key=lambda x: x['rerank score']*w + x['score'], reverse=True)
            if any([ctx['has_answer'] for ctx in all_ctxs[:20]]):
                r20[0] += 1
            if any([ctx['has_answer'] for ctx in all_ctxs[:100]]):
                r100[0] += 1
            if split == 'train':
                all_ctxs = all_ctxs[:100]
            sample['ctxs'] = all_ctxs
        print (f'num_unique={num_unique/len(data)}')
        print (f'num_ql={num_ql/len(data)}')
        print (f'num_r1={num_r1/len(data)}')
        print (f'w={w}, r20={r20[0]/len(data)}, r100={r100[0]/len(data)}')
        with open(cfg.out_file[di]+ f'_shard{shard_id}_of_{cfg.num_shards}.json', 'w') as fout:
            json.dump(data, fout, indent=4)

def link(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg)
    expert_id = None
    if cfg.encoder.use_moe:
        logger.info("Setting expert_id=2")
        expert_id = 2
        logger.info(f"mean pool {cfg.mean_pool}")

    all_passages = {}
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        ctx_sources.append(ctx_src)
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)

    tokenizer = SimpleTokenizer()
    for di, ds_key in enumerate(cfg.qa_dataset):
        logger.info("qa_dataset: %s", ds_key)
        data = json.load(open(ds_key, 'r'))
        questions = []
        spans = []
        indices = []
        question_answers = []
        for sample in data:
            if len(sample['grounding']) == 0:
                continue
            questions.append(sample['question'])
            spans.append([ent[2] for ent in sample['grounding']])
            indices.append([[ent[0], ent[1]] for ent in sample['grounding']])
            question_answers.append(sample['answers'])
        logger.info('questions: %s', len(questions))
        questions_tensor, spans = generate_entity_queries(encoder, tensorizer, questions, spans, indices, cfg.batch_size, expert_id=expert_id, mean_pool=cfg.mean_pool)

        k = cfg.hop1_limit                         
        b_size = 16384
        all_retrieved = []
        for i in tqdm(range(0, len(questions_tensor), b_size)):
            D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].cpu().numpy(), k)  # actual search
            for j, ind in enumerate(I):
                retrieved_chunks = [doc_ids[idx] for idx in ind]
                retrieved_scores = D[j].tolist()
                all_retrieved.append((retrieved_chunks, retrieved_scores))
        print ('all retrieved', len(all_retrieved))

        results = []
        curr = 0
        max_len = 0
        for i, sp in enumerate(spans):
            sample_results = {'question': questions[i], 'answers': question_answers[i], 'ctxs': []}
            this_retrieved = all_retrieved[curr:curr+len(sp)]
            max_len = max(max_len, len(sp))
            for j, psg in enumerate(this_retrieved):
                for k, p in enumerate(psg[0]):
                    content = all_passages[p]
                    p_has_answer = has_answer(question_answers[i], content[0], tokenizer, 'string')
                    sample_results['ctxs'].append({'title': content[1], 'id': p, 'score': psg[1][k], 'span': sp[j], 'has_answer': p_has_answer})
            
            curr += len(sp)
            results.append(sample_results)
        limits = [1, 2, 5, 10, 20, 100, 200, 500, 1000]
        hits = [0]*len(limits)
        for i, limit in enumerate(limits):
            for sample in results:
                if any([ctx['has_answer'] for ctx in sample['ctxs'][:limit]]):
                    hits[i] += 1
        for i, limit in enumerate(limits):
            print (f'hit@{limit}', hits[i]/len(results))    
        with open(cfg.out_file[di], 'w') as fout:
            json.dump(results, fout, indent=4)
        logger.info("max len: %s", max_len)

def retrieve(cfg: DictConfig, encoder, tensorizer, gpu_index_flat, doc_ids, all_passages, ds_key):
    expert_id = None
    if cfg.encoder.use_moe:
        logger.info("Setting expert_id=0")
        expert_id = 0
        logger.info(f"mean pool {cfg.mean_pool}")

    if ds_key.endswith('.csv'):
        data = []
        with open(ds_key, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                data.append({'question': question, 'answers': answers})
    else:
        data = json.load(open(ds_key, 'r'))

    questions = []
    question_answers = []
    for sample in data:
        questions.append(sample['question'])
        question_answers.append(sample['answers'])
    logger.info('questions: %s', len(questions))
    questions_tensor = generate_question_vectors(encoder, tensorizer, questions, cfg.batch_size, expert_id=expert_id, mean_pool=cfg.mean_pool)

    k = cfg.hop1_limit               
    b_size = 16384
    all_retrieved = []
    for i in tqdm(range(0, len(questions_tensor), b_size)):
        D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].cpu().numpy(), k)  # actual search
        for j, ind in enumerate(I):
            retrieved_chunks = [doc_ids[idx] for idx in ind]
            retrieved_scores = D[j].tolist()
            all_retrieved.append((retrieved_chunks, retrieved_scores))
    print ('all retrieved', len(all_retrieved))

    questions_doc_hits = validate(
                all_passages,
                question_answers,
                all_retrieved,
                cfg.validation_workers,
                cfg.match,
            )
    results = prepare_results(
                all_passages,
                questions,
                question_answers,
                all_retrieved,
                questions_doc_hits,
            )
    return results

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    if cfg.do_link:
        link(cfg)
    elif cfg.do_cos:
        chain_of_skills(cfg)
    else:
        raise ValueError('Please specify the task to do')

if __name__ == '__main__':
    main()