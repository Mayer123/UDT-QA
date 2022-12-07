import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

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
from dense_retriever import generate_question_vectors
import faiss
from tqdm import tqdm

logger = logging.getLogger()
setup_logger(logger)

def set_up_encoder(cfg, sequence_length=None):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    if sequence_length is not None:
        cfg.encoder.sequence_length = sequence_length
    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")

    encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
    prefix_len = len(encoder_prefix)

    logger.info("Encoder state prefix %s", encoder_prefix)
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith(encoder_prefix)
    }
    # TODO: long term HF state compatibility fix
    if 'encoder.embeddings.position_ids' in question_encoder_state:
        if 'encoder.embeddings.position_ids' not in model_to_load.state_dict():
            del question_encoder_state['encoder.embeddings.position_ids']
    model_to_load.load_state_dict(question_encoder_state, strict=True)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    ctx_files_patterns = cfg.encoded_ctx_files
    all_context_vecs = []
    for i, pattern in enumerate(ctx_files_patterns):
        print (pattern)
        pattern_files = glob.glob(pattern)
        for f in pattern_files:
            print (f)
            all_context_vecs.extend(pickle.load(open(f, 'rb')))
    all_context_embeds = np.array([line[1] for line in all_context_vecs]).astype('float32')
    doc_ids = [line[0] for line in all_context_vecs]

    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)
    index_flat = faiss.IndexFlatIP(all_context_embeds.shape[1]) 
    gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
    index_flat
    )
    gpu_index_flat.add(all_context_embeds)
    return encoder, tensorizer, gpu_index_flat, doc_ids

def generate_grounding(
    encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    row_start: List[int],
    row_indices: List[List[int]],
    bsz: int,
) -> T:
    n = len(questions)
    found_cells = []
    breaking = 0
    with torch.no_grad():
        for batch_start in tqdm(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            batch_row_start = row_start[batch_start : batch_start + bsz]
            if row_indices is not None:
                batch_row_indices = row_indices[batch_start : batch_start + bsz]
            else:
                batch_row_indices = None

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            q_seq, q_pooled, _, cells_need_link = encoder(question_ids=q_ids_batch, question_segments=q_seg_batch, question_attn_mask=q_attn_mask, 
            question_rep_pos=None, pid_tensors=None, context_ids=None, ctx_segments=None, ctx_attn_mask=None)
            cell_grounding = torch.argmax(cells_need_link, dim=-1).cpu().numpy()
            for i in range(len(cell_grounding)):
                cell_grounding[i][:batch_row_start[i]] = 0
                cell_grounding[i][q_ids_batch[i].cpu().numpy() == tensorizer.tokenizer.pad_token_id] = 0
                cells = []
                start, end = -1, -1 
                for j in range(len(cell_grounding[i])):
                    if batch_row_indices and j in batch_row_indices[i] and cell_grounding[i][j] != 0:
                        if start != -1:
                            breaking += 1
                            full_word_start, full_word_end = extend_span_to_full_words(tensorizer, batch_token_tensors[i], (start, j-1))
                            span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                            cells.append((full_word_start, full_word_end, span))
                            start, end = -1, -1
                    if cell_grounding[i][j] != 0:
                        if start == -1:
                            start = j
                    else:
                        if start != -1:
                            end = j
                            full_word_start, full_word_end = extend_span_to_full_words(tensorizer, batch_token_tensors[i], (start, end-1))
                            span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                            cells.append((full_word_start, full_word_end, span))
                            start, end = -1, -1
                found_cells.append(cells)
    print ('breaking', breaking)
    return found_cells

def generate_entity_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    cells: List[str],
    indices, 
    bsz: int,
    query_token: str = None,
) -> T:
    n = len(questions)
    query_vectors = []
    skip = 0
    remaining_cells = []
    with torch.no_grad():
        for j, batch_start in enumerate(tqdm(range(0, n, bsz))):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            batch_cells = cells[batch_start : batch_start + bsz]

            batch_indices = indices[batch_start : batch_start + bsz]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            seq_out, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            batch_cell_ids = [[torch.tensor(tensorizer.tokenizer.encode(cell, add_special_tokens=False),dtype=torch.long, device=q_ids_batch.device) for cell in sample_cell] for sample_cell in batch_cells]
            question_rep_pos = []
            for b, sample_cell_ids in enumerate(batch_cell_ids):
                q_tensor = q_ids_batch[b]
                this_question = []
                sample_indices = batch_indices[b]
                sample_cell_remain = []
                for _, cell_ids in enumerate(sample_cell_ids):
                    rep_pos = torch.zeros(len(q_tensor), device=q_ids_batch.device)
                    if sample_indices[_][0] == 0:
                        start = 0
                        for i in range(len(q_tensor)-len(cell_ids)):
                            if torch.equal(q_tensor[i:i+len(cell_ids)],cell_ids):
                                start = i
                                break
                        if start == 0:
                            print (q_tensor)
                            print (cell_ids)
                            print ('cannot find cell')
                            print (questions[b])
                            print (batch_cells[b][_])
                            print (j, b)
                            exit(0)
                        rep_pos[start] = 1
                        rep_pos[start+len(cell_ids)-1] = 1
                    else:
                        start = sample_indices[_][0]
                        end = sample_indices[_][1]
                        rep_pos[start] = 1
                        rep_pos[end] = 1
                        if end != start+len(cell_ids)-1:
                            skip += 1
                            continue
                        assert end == start+len(cell_ids)-1
                    sample_cell_remain.append(batch_cells[b][_])
                    this_question.append(rep_pos)
                if len(this_question) == 0:
                    print ('big skip')
                    print (batch_questions[b])
                    question_rep_pos.append(None)
                else:
                    question_rep_pos.append(torch.stack(this_question, dim=0))
                remaining_cells.append(sample_cell_remain)
            for b in range(len(batch_cell_ids)):
                if question_rep_pos[b] is not None:
                    cell_reps = torch.sum(seq_out[b].unsqueeze(0)*question_rep_pos[b].unsqueeze(-1), dim=1)/question_rep_pos[b].sum(dim=1).unsqueeze(-1)
                    query_vectors.append(cell_reps.cpu()) 

    logger.info("Total encoded queries tensor %s", len(query_vectors))
    print ("skip", skip)
    flat_query_vectors = []
    for q in query_vectors:
        flat_query_vectors.extend(q)
    return torch.stack(flat_query_vectors, dim=0), remaining_cells

def get_row_indices(question, tokenizer):
    original_input = tokenizer.tokenize(question)
    rows = question.split('\n')
    indices = []
    tokens = []
    for row in rows:
        tokens.extend(tokenizer.tokenize(row))
        indices.append(len(tokens)+1)
    assert tokens == original_input
    return indices

def prepare_all_table_chunks(filename, tokenizer):
    data = json.load(open(filename, 'r'))
    table_chunks = []
    row_start = []
    row_indices = []
    for chunk in tqdm(data):
        table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
        table_row_indices = get_row_indices(chunk['title'] + ' [SEP] ' + chunk['text'], tokenizer)
        row_start.append(table_row_indices[0])
        row_indices.append(table_row_indices[1:])
    return data, table_chunks, row_start, row_indices

def prepare_all_table_chunks_step2(filename, num_shards, shard_id):
    data = json.load(open(filename, 'r'))
    shard_size = int(len(data)/int(num_shards))
    print ('shard size', shard_size)
    if shard_id != int(num_shards)-1:
        start = shard_id*shard_size
        end = (shard_id+1)*shard_size
    else:
        start = shard_id*shard_size
        end = len(data)
    data = data[start:end]
    print ('working on examples', start, end)
    table_chunks = []
    table_chunk_ids = []
    cells = []
    indices = []
    total_skipped = 0
    for chunk in data:
        if len(chunk['grounding']) == 0:
            total_skipped += 1
            continue
        table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
        table_chunk_ids.append(chunk['chunk_id'])
        cells.append([pos[2] for pos in chunk['grounding']])
        indices.append([(pos[0], pos[1]) for pos in chunk['grounding']])
    print ('total skipped', total_skipped)
    return data, table_chunks, table_chunk_ids, cells, indices

def extend_span_to_full_words(
    tensorizer: Tensorizer, tokens: List[int], span: Tuple[int, int]
) -> Tuple[int, int]:
    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and tensorizer.is_sub_word_id(tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and tensorizer.is_sub_word_id(tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index

def build_query(filename):
    data = json.load(open(filename))
    for sample in data:
        if 'hard_negative_ctxs' in sample:
            del sample['hard_negative_ctxs']
    return data 

def span_proposal(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    print (saved_state.encoder_params)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )
    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    model_to_load.load_state_dict(saved_state.model_dict, strict=False)
    data, table_chunks, table_chunk_ids, row_start, row_indices = prepare_all_table_chunks(cfg.qa_dataset, tensorizer.tokenizer)
    found_cells = generate_grounding(encoder, tensorizer, table_chunks, row_start, row_indices, cfg.batch_size)

    for i in tqdm(range(len(found_cells))):
        data[i]['grounding'] = found_cells[i]
    output_name = '/'.join(cfg.model_file.split('/')[:-1]) + '/all_table_chunks_span_prediction.json'
    json.dump(data, open(output_name, 'w'), indent=4)

def retrieve_hop1_evidence(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg)
    data = build_query(cfg.qa_dataset)
    questions_tensor = generate_question_vectors(encoder, tensorizer,
        [s['question'] for s in data], cfg.batch_size
    )
    assert questions_tensor.shape[0] == len(data)

    k = 100                         
    b_size = 2048
    all_retrieved = []
    for i in tqdm(range(0, len(questions_tensor), b_size)):
        D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].cpu().numpy(), k)  # actual search
        for j, ind in enumerate(I):
            retrieved_chunks = [doc_ids[idx].replace('ott-original:_', '').strip() for idx in ind]
            retrieved_scores = D[j].tolist()
            all_retrieved.append((retrieved_chunks, retrieved_scores))
    print ('all retrieved', len(all_retrieved))

    limits = [1, 5, 20, 50, 100]
    topk = [0]*len(limits)

    for i, sample in enumerate(data):
        if 'positive_ctxs' in sample:
            gold = [pos['chunk_id'] for pos in sample['positive_ctxs']]
            sample['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_], 'gold': ctx in gold} for _, ctx in enumerate(all_retrieved[i][0])]
            for j, limit in enumerate(limits):
                retrieved = all_retrieved[i][0][:limit]
                if any([g in retrieved for g in gold]):
                    topk[j] += 1
        else:
            sample['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_]} for _, ctx in enumerate(all_retrieved[i][0])]
    for i, limit in enumerate(limits):
        print ('topk', topk[i]/len(data), 'limit', limit)
    if 'train' in cfg.qa_dataset:
        split = 'train'
    elif 'dev' in cfg.qa_dataset:
        split = 'dev'
    elif 'test' in cfg.qa_dataset:
        split = 'test'
    else:
        print ('split not found')
        exit(0)
    output_name = '/'.join(cfg.model_file.split('/')[:-1]) + f'/{split}_hop1_retrieved_results.json'
    with open(output_name, 'w') as f:
        json.dump(data, f, indent=4)


def link_all_table_chunks(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg)
    # get questions & answers
    for shard_id in range(int(cfg.num_shards)):
        data, table_chunks, table_chunk_ids, cells, indices = prepare_all_table_chunks_step2(cfg.qa_dataset, cfg.num_shards, shard_id)
        questions_tensor, cells = generate_entity_vectors(encoder, tensorizer,
            table_chunks, cells, indices, cfg.batch_size
        )
        k = 10                        
        b_size = 2048
        all_retrieved = []
        for i in tqdm(range(0, len(questions_tensor), b_size)):
            D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].numpy(), k)  # actual search
            for j, ind in enumerate(I):
                retrieved_titles = [doc_ids[idx].replace('ott-wiki:_', '').split('_')[0].strip() for idx in ind]
                retrieved_scores = D[j].tolist()
                all_retrieved.append((retrieved_titles, retrieved_scores))
        print ('all retrieved', len(all_retrieved))
        curr = 0
        results_data = []
        for i, sample in enumerate(cells):
            sample_res = {'table_chunk_id': table_chunk_ids[i], 'question': table_chunks[i], 'results': []}
            retrieved = all_retrieved[curr:curr+len(sample)]
            curr += len(sample)
            for j, cell in enumerate(sample):
                sample_res['results'].append({'original_cell': cell, 'retrieved': retrieved[j][0], 'scores': retrieved[j][1]})
            results_data.append(sample_res)
        output_name = '/'.join(cfg.model_file.split('/')[:-1]) + f'/table_chunks_to_passages_shard{shard_id}_of_{cfg.num_shards}.json'
        with open(output_name, 'w') as f:
            json.dump(results_data, f, indent=4)

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    if cfg.do_retrieve:
        retrieve_hop1_evidence(cfg)
    elif cfg.do_link:
        link_all_table_chunks(cfg)
    elif cfg.do_span:
        span_proposal(cfg)
    else:
        raise ValueError('Please specify the task to do')

if __name__ == '__main__':
    main()