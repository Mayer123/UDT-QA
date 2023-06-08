import json
import logging
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
import faiss
from tqdm import tqdm
import os
import glob
import pickle
import csv
logger = logging.getLogger()
setup_logger(logger)

def set_up_encoder(cfg, sequence_length=None, no_index=False):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    if saved_state.encoder_params['pretrained_file'] is not None:
        print ('the pretrained file is not None and set to None', saved_state.encoder_params['pretrained_file'])
        saved_state.encoder_params['pretrained_file'] = None
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    print ('because of loading model file, setting pretrained model cfg to bert', cfg.encoder.pretrained_model_cfg)
    cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
    cfg.encoder.encoder_model_type = 'hf_cos'
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
    else:
        if 'encoder.embeddings.position_ids' in model_to_load.state_dict():
            question_encoder_state['encoder.embeddings.position_ids'] = model_to_load.state_dict()['encoder.embeddings.position_ids']     
    model_to_load.load_state_dict(question_encoder_state, strict=True)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    if not no_index:
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
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
        index_flat, co, ngpu=ngpus
        )
        gpu_index_flat.add(all_context_embeds)
    else:
        gpu_index_flat = None
        doc_ids = None
    
    return encoder, tensorizer, gpu_index_flat, doc_ids

def check_across_row(start, end, row_indices):
    for i in range(len(row_indices)):
        if start < row_indices[i] and end > row_indices[i]:
            return row_indices[i]
    return False

def locate_row(start, end, row_indices):
    for i in range(len(row_indices)):
        if end <= row_indices[i]:
            return i
    return -1

def contrastive_generate_grounding(
    encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    row_start: List[int],
    row_indices: List[List[int]],
    bsz: int, sep_id=1, expert_id=5
) -> T:
    n = len(questions)
    found_cells = []
    breaking = 0
    accepted = 0
    rejected = 0
    thresholds = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            if row_start is not None:
                batch_row_start = row_start[batch_start : batch_start + bsz]
            else:
                batch_row_start = []
                for i, token_tensor in enumerate(batch_token_tensors):
                    if '[SEP]' in batch_questions[i]:
                        batch_row_start.append((token_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]+1)
                    else:
                        batch_row_start.append(1)
            if row_indices is not None:
                batch_row_indices = row_indices[batch_start : batch_start + bsz]
            else:
                batch_row_indices = None

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            start_positions = []
            end_positions = []
            num_spans = []
            for i in range(len(batch_row_start)):
                padding_start = (batch_token_tensors[i] == tensorizer.tokenizer.sep_token_id).nonzero()[sep_id][0]+1
                spans = [(i, j) for i in range(batch_row_start[i], padding_start) for j in range(i, min(i+10, padding_start))]
                start_positions.append([s[0] for s in spans])
                end_positions.append([s[1] for s in spans])
                num_spans.append(len(spans))
            batch_start_positions = torch.zeros((len(start_positions), max(num_spans)), dtype=torch.long, device=q_ids_batch.device)
            batch_end_positions = torch.zeros((len(end_positions), max(num_spans)), dtype=torch.long, device=q_ids_batch.device)
            for i in range(len(start_positions)):
                batch_start_positions[i][:len(start_positions[i])] = torch.tensor(start_positions[i])
                batch_end_positions[i][:len(end_positions[i])] = torch.tensor(end_positions[i])

            q_outputs = encoder.question_model(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
            start_vecs = []
            end_vecs = []
            for i in range(q_outputs[0].shape[0]):
                start_vecs.append(torch.index_select(q_outputs[0][i], 0, batch_start_positions[i]))
                end_vecs.append(torch.index_select(q_outputs[0][i], 0, batch_end_positions[i]))
            start_vecs = torch.stack(start_vecs, dim=0)
            end_vecs = torch.stack(end_vecs, dim=0)
            span_vecs = torch.cat([start_vecs, end_vecs], dim=-1)
            span_vecs = torch.tanh(encoder.span_proj(span_vecs))
            cells_need_link = encoder.span_query(span_vecs).squeeze(-1)
            q_pooled = encoder.span_query(q_outputs[1])
            invalid_spans = batch_start_positions == 0
            cells_need_link[invalid_spans] = -1e10
            cells_need_link = cells_need_link - q_pooled 
            sorted_score, indices = torch.sort(cells_need_link, dim=-1, descending=True)
            for i in range(len(batch_row_start)):
                accepted_index = (sorted_score[i] < 0).nonzero()[0][0]
                sorted_start = batch_start_positions[i][indices[i]][:accepted_index]
                sorted_end = batch_end_positions[i][indices[i]][:accepted_index]
                cut_off = 0
                patient = 0
                cells = []
                for j in range(accepted_index):
                    start = sorted_start[j].item()
                    end = sorted_end[j].item()
                    if any([start <= cell[1] and end >= cell[0] for cell in cells]):
                        continue
                    if batch_row_indices is not None:
                        mid = check_across_row(start, end, batch_row_indices[i])
                    else:
                        mid = False
                    if mid:
                        breaking += 1
                        full_word_start = start
                        full_word_end = end
                        span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                        row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                        if span.isnumeric():
                            continue 
                        cells.append((full_word_start, full_word_end, span, row_id))
                        full_word_start = start
                        full_word_end = end
                        span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                        row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                        if span.isnumeric():
                            continue 
                        cells.append((full_word_start, full_word_end, span, row_id))
                    else:
                        full_word_start = start
                        full_word_end = end
                        span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                        if batch_row_indices is not None:
                            row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                        else:
                            row_id = None
                        if span.isnumeric():
                            continue 
                        cells.append((full_word_start, full_word_end, span, row_id))
                thresholds.append(cut_off)
                found_cells.append(cells)
    print ('breaking', breaking)
    print ('accepted', accepted, 'rejected', rejected, 'threshold', np.mean(thresholds))
    return found_cells

def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    expert_id=None, silence=False, mean_pool=False
) -> T:
    n = len(questions)
    query_vectors = []
    with torch.no_grad():
        if not silence:
            iterator = tqdm(range(0, n, bsz))
        else:
            iterator = range(0, n, bsz)
        for batch_start in iterator:
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            seq_out, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
            if mean_pool:
                out = mean_pooling(seq_out, q_attn_mask)
            query_vectors.append(out.cpu())

    return torch.cat(query_vectors, dim=0)

def rerank_hop1_results(
    encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    expert_id: None, silence=False
) -> T:
    all_scores = []
    with torch.no_grad():
        if silence:
            iterator = range(0, len(questions), bsz)
        else:
            iterator = tqdm(range(0, len(questions), bsz))
        for i in iterator:
            batch_questions = questions[i : i + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for sample in batch_questions for q in sample]
            batch_sep_positions = [(q_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0] for q_tensor in batch_token_tensors]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            outputs = encoder(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
            ctx_sep_rep = []
            for i in range(len(batch_sep_positions)):
                ctx_sep_rep.append(outputs[0][i][batch_sep_positions[i]])
            ctx_sep_rep = torch.stack(ctx_sep_rep, dim=0)
            rerank_scores = (ctx_sep_rep * outputs[1]).sum(dim=-1).tolist()
            curr = 0
            for sample in batch_questions:
                all_scores.append(rerank_scores[curr:curr+len(sample)])
                curr += len(sample)
    return all_scores

def generate_entity_queries(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    cells: List[str],
    indices, 
    bsz: int,
    expert_id=None, mean_pool=False
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

            outputs = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
            seq_out = outputs[0]

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
                            print (batch_questions[b])
                            print (batch_cells[b][_])
                            print (j, b)
                            continue
                            exit(0)
                        if mean_pool:
                            rep_pos[start:start+len(cell_ids)] = 1
                        else:
                            rep_pos[start] = 1
                            rep_pos[start+len(cell_ids)-1] = 1
                    else:
                        start = sample_indices[_][0]
                        end = sample_indices[_][1]
                        if mean_pool:
                            rep_pos[start:end+1] = 1
                        else:
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

def prepare_hotpot_questions(filename):
    if filename.endswith('.csv'):
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                data.append({'question': question, 'answers': answers})
    else:
        data = json.load(open(filename, 'r'))
    passages = []
    for sample in data:
        if 'question' not in sample:
            passages.append(sample['title'] + ' [SEP] ' + sample['text'])
        else:
            passages.append(sample['question'])
        if 'hard_negative_ctxs' in sample:
            del sample['hard_negative_ctxs']
        if 'ctxs' in sample:
            del sample['ctxs']
        if 'results' in sample:
            del sample['results']
    print ('total passages', len(passages))
    return data, passages  

def prepare_hotpot_grounded_corpus(filename):
    data = json.load(open(filename, 'r'))
    passages = []
    passage_ids = []
    cells = []
    indices = []
    total_skipped = 0
    for _, sample in tqdm(enumerate(data)):
        grounding = sample['grounding'] 
        if len(grounding) == 0:
            total_skipped += 1
            continue
        if 'chunk_id' in sample:
            passages.append(sample['title'] + ' [SEP] ' + sample['text'])
            passage_ids.append(sample['chunk_id'])
        else:
            passages.append(sample['question'])
            if '_id' in sample:
                passage_ids.append(sample['_id'])
            else:
                passage_ids.append(_)
        cells.append([pos[2] for pos in grounding])
        indices.append([(pos[0], pos[1]) for pos in grounding])
    print ('total passages', len(passages), 'total skipped', total_skipped)
    return passages, passage_ids, cells, indices

def span_proposal(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    cfg.encoder.encoder_model_type = 'hf_cos'

    cfg.encoder.pretrained_file=None
    cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'

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
    if 'question_model.encoder.embeddings.position_ids' not in saved_state.model_dict:
        if 'question_model.encoder.embeddings.position_ids' in model_to_load.state_dict():
            saved_state.model_dict['question_model.encoder.embeddings.position_ids'] = model_to_load.state_dict()['question_model.encoder.embeddings.position_ids']
            saved_state.model_dict['ctx_model.encoder.embeddings.position_ids'] = model_to_load.state_dict()['ctx_model.encoder.embeddings.position_ids']
    model_to_load.load_state_dict(saved_state.model_dict, strict=True)

    data, table_chunks = prepare_hotpot_questions(cfg.qa_dataset)
    shard_size = int(len(table_chunks)/int(cfg.num_shards))
    print ('shard size', shard_size)
    shard_id = int(cfg.shard_id)
    if shard_id != int(cfg.num_shards)-1:
        start = shard_id*shard_size
        end = (shard_id+1)*shard_size
    else:
        start = shard_id*shard_size
        end = len(table_chunks)
    print ('working on start', start, 'end', end)
    table_chunks = table_chunks[start:end]
    data = data[start:end]
    expert_id = 5
    if cfg.encoder.num_expert < 6:
        logger.info("Setting expert_id=0 since num_expert < 6")
        expert_id = 0
    else:
        logger.info("Setting expert_id=5")
    if 'question' in data[0] and '[SEP]' not in table_chunks[0]:
        print ('setting sep id to 0')
        found_cells = contrastive_generate_grounding(encoder, tensorizer, table_chunks, None, None, cfg.batch_size, sep_id=0, expert_id=expert_id)
    else:
        found_cells = contrastive_generate_grounding(encoder, tensorizer, table_chunks, None, None, cfg.batch_size, expert_id=expert_id)

    for i in tqdm(range(len(found_cells))):
        data[i]['grounding'] = found_cells[i]
    print ('writing to', '/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}_shard{shard_id}_of_{cfg.num_shards}.json')
    json.dump(data, open('/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}_shard{shard_id}_of_{cfg.num_shards}.json', 'w'), indent=4)

def main_link_all_passage_corpus(cfg, encoder, tensorizer, gpu_index_flat, doc_ids):    
    expert_id = None
    if cfg.encoder.use_moe:
        logger.info("Setting expert_id=2")
        expert_id = 2
        if cfg.encoder.num_expert < 6:
            logger.info("Setting expert_id=0 since num_expert < 6")
            expert_id = 0

    # get questions & answers
    table_chunks_all, table_chunk_ids_all, cells_all, indices_all = prepare_hotpot_grounded_corpus(cfg.qa_dataset)
    shard_size = int(len(table_chunks_all)/int(cfg.num_shards))
    print ('shard size', shard_size)
    hotpot_index_to_title, _ = get_hotpot_index_to_title(filename=cfg.ctx_datatsets[0])
    shard_id = int(cfg.shard_id)
    if shard_id != int(cfg.num_shards)-1:
        start = shard_id*shard_size
        end = (shard_id+1)*shard_size
    else:
        start = shard_id*shard_size
        end = len(table_chunks_all)
    print ('working on start', start, 'end', end)
    table_chunks = table_chunks_all[start:end]
    table_chunk_ids = table_chunk_ids_all[start:end]
    cells = cells_all[start:end]
    indices = indices_all[start:end]
    questions_tensor, cells = generate_entity_queries(encoder, tensorizer,
        table_chunks, cells, indices, cfg.batch_size, expert_id=expert_id
    )

    k = 5                        
    b_size = cfg.batch_size*8
    all_retrieved = []
    for i in tqdm(range(0, len(questions_tensor), b_size)):
        D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].numpy(), k)  # actual search
        for j, ind in enumerate(I):
            retrieved_titles = [hotpot_index_to_title[doc_ids[idx].replace('hotpot-wiki:_', '')] for idx in ind]
            retrieved_scores = D[j].tolist()
            all_retrieved.append((retrieved_titles, retrieved_scores))
    print ('all retrieved', len(all_retrieved))
    
    curr = 0
    results_data = []
    for i, sample in enumerate(cells):
        sample_res = {'passage_id': table_chunk_ids[i], 'question': table_chunks[i], 'results': []}
        retrieved = all_retrieved[curr:curr+len(sample)]
        curr += len(sample)
        for j, cell in enumerate(sample):
            sample_res['results'].append({'grounded': cell, 'retrieved': retrieved[j][0], 'scores': retrieved[j][1]})
        results_data.append(sample_res)
    output_name = '/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}_shard{shard_id}_of_{cfg.num_shards}.json'
    print ('writing to', output_name)
    with open(output_name, 'w') as f:
        json.dump(results_data, f, indent=4)

def get_hotpot_index_to_title(include_text=False, filename='HotpotQA/hotpot_corpus.jsonl'):
    hotpot_index_to_title = {}
    title_to_text = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            para = json.loads(line)
            if include_text:
                hotpot_index_to_title[str(i)] = para['title']
                title_to_text[para['title']] = para['text']
            else:
                hotpot_index_to_title[str(i)] = para['title']
    return hotpot_index_to_title, title_to_text

def load_links(links_path, linked_size, hotpot_index_to_title):
    table_to_pasg_links = []
    for path in links_path:
        for f in glob.glob(path):
            print (f)
            table_to_pasg_links += json.load(open(f, 'r'))
    all_links = {}
    for sample in tqdm(table_to_pasg_links):
        links = []
        for ci, res in enumerate(sample['results']):
            for i in range(linked_size):
                links.append((res['retrieved'][i], res['scores'][i], ci, res['grounded'], i))
        unique_links = {}
        for l in links:
            if l[0] not in unique_links:
                unique_links[l[0]] = (l[1], l[2], l[3], l[4])
            else:
                if unique_links[l[0]][0] < l[1]:
                    unique_links[l[0]] = (l[1], l[2], l[3], l[4])
        all_links[hotpot_index_to_title[sample['passage_id']]] = sorted(unique_links.items(), key=lambda x: x[1][0], reverse=True)
    print ('all links', len(all_links))
    return all_links

def build_query(filename):
    data = json.load(open(filename, 'r'))
    questions = []
    question_ids = []
    gold_titles = []
    answers = []
    types = []
    for i, sample in enumerate(data):
        questions.append(sample['question'])
        question_ids.append(sample['_id'])
        if 'positive_ctxs' in sample:
            pos_titles = [pos['title'] for pos in sample['positive_ctxs']]
            if 'other_positive_ctxs' in sample:
                pos_titles.extend([pos['title'] for pos in sample['other_positive_ctxs']])
        else:
            pos_titles = []
        gold_titles.append(pos_titles)
        if 'answers' in sample:
            answers.append(sample['answers'])
            types.append(sample['type'])
        else:
            answers.append('random')
            types.append('unknown')
    return questions, gold_titles, question_ids, answers, types

def q_to_passage_retrieval(cfg, encoder, tensorizer, gpu_index_flat, doc_ids, hotpot_index_to_title):
    # get questions & answers
    expert_id = None
    if cfg.encoder.use_moe:
        expert_id = 0
        logger.info(f"Setting expert_id={expert_id}")
    logger.info(f"mean pool {cfg.mean_pool}")

    questions, gold_titles, question_ids, answers, types = build_query(cfg.qa_dataset)
   
    questions_tensor = generate_question_vectors(encoder, tensorizer,
        questions, cfg.batch_size, expert_id=expert_id, mean_pool=cfg.mean_pool
    )
    assert questions_tensor.shape[0] == len(questions)

    k = cfg.hop1_limit                     
    b_size = 2048
    all_retrieved = []
    for i in tqdm(range(0, len(questions_tensor), b_size)):
        D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].numpy(), k)  # actual search
        for j, ind in enumerate(I):
            retrieved_chunks = [hotpot_index_to_title[doc_ids[idx].replace('hotpot-wiki:_', '').strip()] for idx in ind]
            retrieved_scores = D[j].tolist()
            all_retrieved.append((retrieved_chunks, retrieved_scores))
    print ('all retrieved', len(all_retrieved))

    limits = [1, 5, 10, 20, 50, 100]
    topk = [0]*len(limits)
    topk_single = [0]*len(limits)

    results_data = []
    for i, gold in enumerate(gold_titles):
        sample_res = {'_id': question_ids[i], 'question': questions[i], 'type': types[i], 'answers': answers[i], 'pos titles': gold}
        if gold is not None:
            sample_res['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_], 'gold': ctx in gold} for _, ctx in enumerate(all_retrieved[i][0])]
            for j, limit in enumerate(limits):
                retrieved = all_retrieved[i][0][:limit]
                if all([g in retrieved for g in gold]):
                    topk[j] += 1
                if any([g in retrieved for g in gold]):
                    topk_single[j] += 1
        else:
            sample_res['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_]} for _, ctx in enumerate(all_retrieved[i][0])]
        results_data.append(sample_res)
    for i, limit in enumerate(limits):
        print ('topk', topk[i]/len(questions), 'topk single', topk_single[i]/len(questions), 'limit', limit)
    return results_data

def build_hop1_results(data, linker_d, title_to_text, hop1_limit=100, num_shards=1, shard_id=0):
    shard_size = int(len(data)/int(num_shards))
    print ('shard size', shard_size)
    if shard_id != int(num_shards)-1:
        start = shard_id*shard_size
        end = (shard_id+1)*shard_size
    else:
        start = shard_id*shard_size
        end = len(data)
    print ('working on start', start, 'end', end)
    data = data[start:end]
    questions = []
    hop1_info = []
    rerank_scores = []
    for i, sample in enumerate(data):
        this_question = []
        this_titles = []
        this_scores = []
        this_sources = []
        for res in sample['results'][:hop1_limit]:
            this_question.append(sample['question']+' [SEP] '+''.join(title_to_text[res['title']]))
            this_titles.append(res['title'])
            this_scores.append(res['score'])
            this_sources.append('r1')
        if sample['question'] in linker_d:
            this_ql_results = linker_d[sample['question']]
            max_retriever_score = max(this_scores)
            max_linker_score = max([res['scores'][0] for res in this_ql_results['results']])
            max_linker_score = max(max_linker_score, max_retriever_score)
            for res in this_ql_results['results']:
                for ti, title in enumerate(res['retrieved'][:5]):
                    scaled_score = res['scores'][ti]*max_retriever_score/max_linker_score
                    if title not in this_titles:
                        this_question.append(sample['question']+' [SEP] '+''.join(title_to_text[title]))
                        this_titles.append(title)
                        this_scores.append(scaled_score)
                        this_sources.append('ql')
                    else:
                        idx = this_titles.index(title)
                        this_scores[idx] = max(this_scores[idx], scaled_score)
                
        questions.append(this_question)
        this_info = [{'title': title, 'score': score, 'source': source, 'q': question} for title, score, source, question in zip(this_titles, this_scores, this_sources, this_question)]
        hop1_info.append(this_info)
    return data, questions, hop1_info

def chain_of_skills(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg)

    if 'train' in cfg.qa_dataset:
        split = 'train'
    elif 'dev' in cfg.qa_dataset:
        split = 'dev'
    elif 'test' in cfg.qa_dataset:
        split = 'test'
    else:
        print ('split not found')
        exit(0)

    print ('hop1 limit', cfg.hop1_limit, 'hop1 keep', cfg.hop1_keep, 'hop2 limit', cfg.hop2_limit)
    
    hotpot_index_to_title, title_to_text = get_hotpot_index_to_title(True, cfg.ctx_datatsets[0])
    hop1_retrieved = q_to_passage_retrieval(cfg, encoder, tensorizer, gpu_index_flat, doc_ids, hotpot_index_to_title)

    linker_results = json.load(open(cfg.ctx_datatsets[1], 'r'))
    linker_d = {}
    for sample in linker_results:
        linker_d[sample['question']] = sample
    
    data, questions, hop1_info = build_hop1_results(hop1_retrieved, linker_d, title_to_text, hop1_limit=cfg.hop1_limit, num_shards=cfg.num_shards, shard_id=cfg.shard_id)

    output_name = cfg.qa_dataset.replace('.json', f'_hop1_rerank_results_shard{cfg.shard_id}_of_{cfg.num_shards}.json')
    rank_expert=4
    if cfg.encoder.num_expert < 6:
        logger.info("setting rank_expert to 0 because num_expert < 6")
        rank_expert = 0
    if os.path.exists(output_name):
        hop1_info = json.load(open(output_name, 'r'))
        logger.info(f'loading hop1 rerank results from {output_name}')
        rerank_scores = [[] for _ in range(len(hop1_info))]
    else:
        rerank_scores = rerank_hop1_results(encoder, tensorizer, questions, cfg.batch_size, expert_id=rank_expert)
        for i in range(len(hop1_info)):
            for j in range(len(hop1_info[i])):
                hop1_info[i][j]['rerank_score'] = rerank_scores[i][j]
        with open(output_name, 'w') as f:
            json.dump(hop1_info, f, indent=4)

    for i in range(len(hop1_info)):
        rerank_scores[i] = {res['title']: res['rerank_score'] for res in hop1_info[i]}
        hop1_info[i] = sorted(hop1_info[i], key=lambda x: x['rerank_score']*1.5+x['score'], reverse=True)
        hop1_info[i] = hop1_info[i][:cfg.hop1_keep]
    
    torch.cuda.empty_cache()
    # get questions & answers
    logger.info(f"Setting expert_id={cfg.hop2_expert}")
    
    all_links = load_links(['/'.join(cfg.model_file.split('/')[:-1]) +f'/hotpot_pasg_to_pasg_links*'], 2, hotpot_index_to_title)
    print ('all links', len(all_links))

    k = cfg.hop2_limit                         
    b_size = 2048
    results_data = []
    limits = [1, 5, 10, 20, 50, 100]
    topk = [0]*len(limits)

    for i in tqdm(range(len(hop1_info))):
        questions_tensor = generate_question_vectors(encoder, tensorizer, [exp['q'] for exp in hop1_info[i]], b_size, expert_id=cfg.hop2_expert, silence=True)
        D, I = gpu_index_flat.search(questions_tensor.numpy(), k)  # actual search
        relevant_graph = {}
        hop1_titles = [exp['title'] for exp in hop1_info[i]]
        for hop1 in hop1_titles:
            if hop1 in all_links:
                relevant_graph[hop1] = {link[0]:link[1][0] for link in all_links[hop1] if link[0] != hop1}
                if len(relevant_graph[hop1]) == 0:
                    relevant_graph.pop(hop1)
                    print ('no links', hop1)
        this_question = []
        rerank_d = rerank_scores[i]
        hop2_for_rerank = []
        hop2_title_d = {}

        for j, ind in enumerate(I):
            hop2_scores = D[j].tolist()
            hop2_titles = [hotpot_index_to_title[doc_ids[idx].replace('hotpot-wiki:_', '').strip()] for idx in ind]
            pl_scores = []
            pl_titles = []
            if hop1_titles[j] in relevant_graph:
                max_hop2_score = max(hop2_scores)
                this_max_pl = max(max(relevant_graph[hop1_titles[j]].values()), max_hop2_score)
                for hop2 in relevant_graph[hop1_titles[j]]:
                    if hop2 not in hop2_titles:
                        pl_scores.append(relevant_graph[hop1_titles[j]][hop2]*max_hop2_score/this_max_pl)
                        pl_titles.append(hop2)
                    else:
                        idx = hop2_titles.index(hop2)
                        hop2_scores[idx] = max(hop2_scores[idx], relevant_graph[hop1_titles[j]][hop2]*max_hop2_score/this_max_pl)*1.06
            hop2_sources = ['r2'] * len(hop2_titles) + ['pl'] * len(pl_titles)
            hop2_titles = hop2_titles + pl_titles
            hop2_scores = hop2_scores + pl_scores
            for title in hop2_titles:
                if title not in rerank_d and title not in hop2_title_d:
                    hop2_for_rerank.append(data[i]['question'] + ' [SEP] ' + ''.join(title_to_text[title]))
                    hop2_title_d[title] = len(hop2_for_rerank) - 1
            this_question.append((hop2_titles, hop2_scores, hop2_sources))
        hop2_rerank_scores = rerank_hop1_results(encoder, tensorizer, [hop2_for_rerank[b:b+cfg.hop1_limit] for b in range(0, len(hop2_for_rerank), cfg.hop1_limit)], cfg.batch_size, expert_id=rank_expert, silence=True)
        hop2_rerank_scores = [x for sub in hop2_rerank_scores for x in sub]
        for h2, h2_idx in hop2_title_d.items():
            rerank_d[h2] = hop2_rerank_scores[h2_idx]
        beams = []
        for j, (hop2_titles, hop2_scores, hop2_sources) in enumerate(this_question):
            for _, title in enumerate(hop2_titles):
                beams.append([hop1_titles[j], title, hop1_info[i][j]['score']+hop1_info[i][j]['rerank_score']*1.5+hop2_scores[_]+rerank_d[title], hop1_info[i][j]['score'],hop1_info[i][j]['rerank_score'], hop2_scores[_], rerank_d[title], hop1_info[i][j]['source'], hop2_sources[_]])
               
        beams.sort(key=lambda x: x[2], reverse=True)
        sample_res = data[i]
        sample_res['ctxs'] = []
        exist = set()
        for j, p in enumerate(beams):
            if p[0] == p[1]:
                continue
            if any([len(set(prev).intersection([p[0], p[1]]))==2  for prev in exist]):
                continue 
            exist.add((p[0], p[1]))
            if len(set([p[0], p[1]]).intersection(set(data[i]['pos titles']))) == 2:
                sample_res['ctxs'].append({'hop1 title': p[0], 'hop2 title':p[1], 'path score': p[2], 'hop1 score': p[3], 'hop1 rerank':p[4], 'hop2 score': p[5], 'hop2 rerank':p[6], 'hop1 source': p[7], 'hop2 source': p[8], 'is gold':True})
            else:
                sample_res['ctxs'].append({'hop1 title': p[0], 'hop2 title':p[1], 'path score': p[2], 'hop1 score': p[3], 'hop1 rerank':p[4], 'hop2 score': p[5], 'hop2 rerank':p[6], 'hop1 source': p[7], 'hop2 source': p[8], 'is gold':False})
        for j, limit in enumerate(limits):
            retrieved = sample_res['ctxs'][:limit]
            if any([r['is gold'] for r in retrieved]):
                topk[j] += 1
        results_data.append(sample_res)
    for i, limit in enumerate(limits):
        print ('topk', topk[i]/len(data), 'limit', limit)
   
    output_name = cfg.qa_dataset.replace('.json', f'_chain_of_skills_results_shard{cfg.shard_id}_of_{cfg.num_shards}.json')
    with open(output_name, 'w') as f:
        json.dump(results_data, f, indent=4)

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    if cfg.do_link:
        encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg, sequence_length=512)
        main_link_all_passage_corpus(cfg, encoder, tensorizer, gpu_index_flat, doc_ids)
    elif cfg.do_span:
        span_proposal(cfg)
    elif cfg.do_cos:
        chain_of_skills(cfg)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
    
    