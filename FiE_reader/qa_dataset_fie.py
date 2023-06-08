# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from collections import Counter
import json
import random

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import logging 
import sys 
sys.path.append("../DPR/")
from dpr.utils.tokenizers import SimpleTokenizer
logger = logging.getLogger(__name__)

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def prepare_path(path, q_type=None, pasg_sep='[SEP]'):
    """
    tokenize the passages chains, add sentence start markers for SP sentence identification
    """
    def _process_p(title, sents, curr):
        para = title + ' '
        curr += len(title) + 1
        sent_starts = []
        for idx, sent in enumerate(sents):
            sent_starts.append(curr)
            sent = sent.strip()
            para += sent + ' '
            curr += len(sent) + 1
        return para, sent_starts

    # mark passage boundary
    contexts = f'yes no [unused1] '
    sent_starts = []

    for i in range(2):
        contexts += f"{pasg_sep} "
        curr = len(contexts)
        p_title = path[f'hop{i+1} title']
        p_text = path[f'hop{i+1} text']
        text, starts = _process_p(p_title, p_text, curr)
        contexts += text
        sent_starts += starts
    return contexts, sent_starts

def match_answer_span_new(tokens, answer_tokens):
    spans = []
    for i in range(0, len(tokens)-len(answer_tokens)+1):
        if tokens[i:i+len(answer_tokens)] == answer_tokens:
            spans.append((i, i+len(answer_tokens)-1))
    return spans

def get_answer_indices(answers, para_offset, tokens, ans_tokens):
    if answers == "yes":
        starts, ends= [para_offset], [para_offset]
    elif answers == "no":
        starts, ends= [para_offset + 1], [para_offset + 1]
    else:
        matched_spans = match_answer_span_new(tokens, ans_tokens)
        starts, ends = [], []
        for span in matched_spans:
            starts.append(span[0] + para_offset)
            ends.append(span[1] + para_offset)
        if len(starts) == 0:
            starts, ends = [-1], [-1]       
    return starts, ends  

def get_answer_indices_general(para_offset, tokens, ans_tokens):
    starts, ends = [], []
    for ans in ans_tokens:
        matched_spans = match_answer_span_new(tokens, ans)
        for span in matched_spans:
            starts.append(span[0] + para_offset)
            ends.append(span[1] + para_offset)
    if len(starts) == 0:
        starts, ends = [-1], [-1]       
    return starts, ends  

class QADataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_seq_len,
        max_q_len,
        max_ans_len,
        train=False,
        no_sent_label=False, neg_num=5, debug=False, num_ctx=101,
        world_size=1, global_rank=0
        ):
        if debug:
            retriever_outputs = json.load(open(data_path, 'r'))[:50]
        else:
            retriever_outputs = json.load(open(data_path, 'r'))
        answer_len_d = Counter()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len
        self.train = train
        self.no_sent_label = no_sent_label
        self.simple_tok = SimpleTokenizer()
        self.neg_num = neg_num
        logger.info(f"World size {world_size} Global rank {global_rank} Data size {len(retriever_outputs)} Neg num {self.neg_num}")
        self.data = []
        avg_ctx = 0
        avg_sp = 0
        for si, item in enumerate(retriever_outputs):
            if si % world_size != global_rank:
                continue
            if item["question"].endswith("?"):
                item["question"] = item["question"][:-1]
            q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
            ans_tokens = self.tokenizer.tokenize(item["answers"][0])[:self.max_ans_len]
            answer_len_d[len(ans_tokens)] += 1
            sp_gold = item["sp"]   
            sp_titles = item['pos titles']
            candidate_chains = []
            sp_sent_labels = {}
            for chain in item["ctxs"][:num_ctx]:
                chain['hop1 text'] = [x for x in chain['hop1 text'] if x.strip() != '']
                chain['hop2 text'] = [x for x in chain['hop2 text'] if x.strip() != '']
                chain_ctx = prepare_path(chain, pasg_sep='[SEP]')
                encoded = self.tokenizer(chain_ctx[0], add_special_tokens=False, return_offsets_mapping=True, max_length=self.max_seq_len - len(q_toks) - 3, truncation=True)
                chain_sent_st = [encoded.char_to_token(cid)+len(q_toks)+2 for cid in chain_ctx[1] if encoded.char_to_token(cid) is not None]
                chain_sent_st.append(len(encoded['input_ids'])+len(q_toks)+2)
                chain['sent_st'] = chain_sent_st
                chain['original_text'] = chain_ctx[0]
                chain['encoded'] = encoded
                if self.train:
                    chain['sp_sent_labels'] = []
                    if chain['hop1 title'] in sp_titles:
                        if chain['hop1 title'] not in sp_sent_labels:
                            sp_sent_labels[chain['hop1 title']] = []
                            this_sp_sents = [sent[1] for sent in sp_gold if sent[0] == chain['hop1 title']]
                            for i in range(len(chain['hop1 text'])):
                                if i in this_sp_sents:
                                    sp_sent_labels[chain['hop1 title']].append(1)
                                else:
                                    sp_sent_labels[chain['hop1 title']].append(0)
                        chain['sp_sent_labels'].extend(sp_sent_labels[chain['hop1 title']])
                    else:
                        chain['sp_sent_labels'].extend([0]*len(chain['hop1 text']))
                    if chain['hop2 title'] in sp_titles:
                        if chain['hop2 title'] not in sp_sent_labels:
                            sp_sent_labels[chain['hop2 title']] = []
                            this_sp_sents = [sent[1] for sent in sp_gold if sent[0] == chain['hop2 title']]
                            for i in range(len(chain['hop2 text'])):
                                if i in this_sp_sents:
                                    sp_sent_labels[chain['hop2 title']].append(1)
                                else:
                                    sp_sent_labels[chain['hop2 title']].append(0)
                        chain['sp_sent_labels'].extend(sp_sent_labels[chain['hop2 title']])
                    else:
                        chain['sp_sent_labels'].extend([0]*len(chain['hop2 text']))
                    chain['sp_sent_labels'] = chain['sp_sent_labels'][:len(chain_sent_st)-1]
                    if item['type'] == 'comparison':
                        if chain['hop1 title'] in sp_titles and chain['hop2 title'] in sp_titles:
                            starts, ends = get_answer_indices(item["answers"][0], len(q_toks) + 2, self.tokenizer.convert_ids_to_tokens(encoded['input_ids']), ans_tokens)
                        else:
                            starts, ends = [-1], [-1]
                    else:
                        starts, ends = get_answer_indices(item["answers"][0], len(q_toks) + 2, self.tokenizer.convert_ids_to_tokens(encoded['input_ids']), ans_tokens)
                        if item["answers"][0].lower() in chain_ctx[0].lower():
                            if starts[0] == -1:
                                print ('answer matching problem')
                                print (item["answers"][0], chain_ctx[0])
                                print (ans_tokens)
                                print (self.tokenizer.convert_ids_to_tokens(encoded['input_ids']))
                    chain['ans_starts'] = starts
                    chain['ans_ends'] = ends
                candidate_chains.append(chain)
            avg_ctx += len(candidate_chains)
            if self.train:
                if all([chain['ans_starts'][0] == -1 for chain in candidate_chains]):
                    continue
            self.data.append({
                "question": item["question"],
                "q_toks": q_toks,
                "type": item["type"],
                "qid": item["_id"],
                "gold_answer": item["answers"],
                "ans_tokens": ans_tokens,
                "candidate_chains": candidate_chains,
                "sp_gold": sp_gold,}
            )
        print (answer_len_d)
        logger.info(f"Global rank {global_rank} Data size {len(self.data)} Average context {avg_ctx/len(self.data)} Average sp {avg_sp/len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        q_toks = item['q_toks']
        para_offset = len(q_toks) + 2 # cls and seq

        input_ids = []
        token_type_ids = []
        answer_starts = []
        answer_ends = []
        sent_starts = []
        sent_labels = []
        chain_titles = []
        original_text = []
        offset_mapping = []
        if self.train:
            for i, ctx in enumerate(item['candidate_chains']):
                sent_starts.append(torch.LongTensor(ctx['sent_st']))
                sent_labels.append(torch.LongTensor(ctx['sp_sent_labels']))
                answer_starts.append(torch.LongTensor(ctx['ans_starts']))
                answer_ends.append(torch.LongTensor(ctx['ans_ends']))
                input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token]) + ctx['encoded']['input_ids'] + [self.tokenizer.sep_token_id]))
                token_type_ids.append(torch.LongTensor([0]*(len(q_toks)+2) + [1]*len(ctx['encoded']['input_ids']) + [0]))
        else:
            for i, chain in enumerate(item['candidate_chains']):
                chain_titles.append([chain['hop1 title'], chain['hop2 title'], len(chain['hop1 text']), len(chain['hop2 text'])])
                encoded = chain['encoded'] 
                original_text.append(chain['original_text'])
                offset_mapping.append(encoded['offset_mapping'])
                sent_starts.append(torch.LongTensor(chain['sent_st']))
                input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token]) + encoded['input_ids'] + [self.tokenizer.sep_token_id]))
                token_type_ids.append(torch.LongTensor([0] * (len(q_toks) + 2) + [1] * (len(encoded['input_ids']) + 1)))
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'answer_starts': answer_starts, 'answer_ends': answer_ends, 'sp_sent_labels': sent_labels, 'sent_starts': sent_starts, 'sp_gold': item['sp_gold'], 'qid': item['qid'], 'type': item['type'], 'gold_answer': item['gold_answer'], 'para_offset': para_offset, 'chain_titles': chain_titles, 'original_text': original_text, 'offset_mapping': offset_mapping}

def qa_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([p for s in samples for p in s['input_ids']], pad_id),
        'token_type_ids': collate_tokens([p for s in samples for p in s['token_type_ids']], 0),
        }
    batch['attention_mask'] = batch['input_ids'].ne(pad_id).long()
    batch['num_ctx'] = len(samples[0]['input_ids'])

    # training labels
    if len(samples[0]['answer_starts']) > 0:
        batch["sent_starts"] = collate_tokens([p for s in samples for p in s['sent_starts']], -1)
        batch['sp_sent_labels'] = collate_tokens([p for s in samples for p in s["sp_sent_labels"]], -1)
        batch["starts"] = collate_tokens([p for s in samples for p in s['answer_starts']], -1)
        batch["ends"] = collate_tokens([p for s in samples for p in s['answer_ends']], -1)
    else:
        batch["sent_starts"] = collate_tokens([p for s in samples for p in s['sent_starts']], -1)

    batched = {
        "qids": [s["qid"] for s in samples],
        "gold_answer": [s["gold_answer"] for s in samples],
        "sp_gold": [s["sp_gold"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "chain_titles": [s["chain_titles"] for s in samples],
        "original_text": [s["original_text"] for s in samples],
        "offset_mapping": [s["offset_mapping"] for s in samples],
        "net_inputs": batch,
    }

    return batched


class QADatasetNoSP(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_seq_len,
        max_q_len,
        max_ans_len,
        train=False,
        neg_num=5, debug=False, num_ctx=100,
        world_size=1, global_rank=0
        ):
        if debug:
            retriever_outputs = json.load(open(data_path, 'r'))[:50]
        else:
            retriever_outputs = json.load(open(data_path, 'r'))
        answer_len_d = Counter()
        answer_num_d = Counter()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len
        self.train = train
        self.simple_tok = SimpleTokenizer()
        self.neg_num = neg_num
        logger.info(f"World size {world_size} Global rank {global_rank} Data size {len(retriever_outputs)} Neg num {self.neg_num}")
        self.data = []
        avg_ctx = 0
        for si, item in enumerate(retriever_outputs):
            if si % world_size != global_rank:
                continue
            if item["question"].endswith("?"):
                item["question"] = item["question"][:-1]
            q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
            if 'answers' not in item:
                item['answers'] = 'random'
            ans_tokens = [self.tokenizer.tokenize(ans)[:self.max_ans_len] for ans in item["answers"]]
            answer_len_d.update([len(ans) for ans in ans_tokens])
            answer_num_d.update([len(ans_tokens)])
            
            if self.train and all([not ctx['has_answer'] for ctx in item['ctxs'][:num_ctx]]):
                continue        
            # top ranked negative chains
            candidate_chains = []
            for chain in item["ctxs"][:num_ctx]:
                chain_ctx = chain['title'] + ' [SEP] ' + chain['text']
                encoded = self.tokenizer(chain_ctx, add_special_tokens=False, return_offsets_mapping=True, max_length=self.max_seq_len - len(q_toks) - 3, truncation=True)
                chain['original_text'] = chain_ctx
                chain['encoded'] = encoded
                if self.train:
                    if chain['has_answer']:
                        starts, ends = get_answer_indices_general(len(q_toks) + 2, self.tokenizer.convert_ids_to_tokens(encoded['input_ids']), ans_tokens)
                        chain['answer_starts'] = starts
                        chain['answer_ends'] = ends
                    else:
                        chain['answer_starts'] = [-1]
                        chain['answer_ends'] = [-1]
                candidate_chains.append(chain)
            if self.train and all([chain['answer_starts'][0] == -1 for chain in candidate_chains]):
                print ('no answer matched, probably due to truncation')
                continue
            avg_ctx += len(candidate_chains)
            self.data.append({
                "question": item["question"],
                "q_toks": q_toks,
                "qid": item['id'] if 'id' in item else si,
                "gold_answer": item["answers"],
                "ans_tokens": ans_tokens,
                "candidate_chains": candidate_chains
            })
        print (answer_len_d)
        print (answer_num_d)
        logger.info(f"Global rank {global_rank} Data size {len(self.data)} Average context {avg_ctx/len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        q_toks = item['q_toks']
        para_offset = len(q_toks) + 2 # cls and seq

        input_ids = []
        token_type_ids = []
        answer_starts = []
        answer_ends = []
        original_text = []
        offset_mapping = []
        for i, chain in enumerate(item['candidate_chains']):
            encoded = chain['encoded'] 
            input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token]) + encoded['input_ids'] + [self.tokenizer.sep_token_id]))     
            token_type_ids.append(torch.LongTensor([0] * (len(q_toks) + 2) + [1] * (len(encoded['input_ids']) + 1)))
            if self.train:
                answer_starts.append(torch.LongTensor(chain['answer_starts']))
                answer_ends.append(torch.LongTensor(chain['answer_ends']))  
            else:
                original_text.append(chain['original_text'])
                offset_mapping.append(encoded['offset_mapping'])
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'answer_starts': answer_starts, 'answer_ends': answer_ends, 'qid': item['qid'], 'gold_answer': item['gold_answer'], 'para_offset': para_offset, 'original_text': original_text, 'offset_mapping': offset_mapping}

def qa_collate_no_sp(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([p for s in samples for p in s['input_ids']], pad_id),
        'token_type_ids': collate_tokens([p for s in samples for p in s['token_type_ids']], 0),
        }
    batch['attention_mask'] = batch['input_ids'].ne(pad_id).long()
    batch['num_ctx'] = len(samples[0]['input_ids'])

    # training labels
    if len(samples[0]['answer_starts']) > 0:
        batch["starts"] = collate_tokens([p for s in samples for p in s['answer_starts']], -1)
        batch["ends"] = collate_tokens([p for s in samples for p in s['answer_ends']], -1)

    batched = {
        "qids": [s["qid"] for s in samples],
        "gold_answer": [s["gold_answer"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "original_text": [s["original_text"] for s in samples],
        "offset_mapping": [s["offset_mapping"] for s in samples],
        "net_inputs": batch,
    }
    return batched