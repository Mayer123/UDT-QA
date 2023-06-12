# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import collections
import json
import random

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

import sys 
sys.path.append("../DPR/")
from dpr.utils.tokenizers import SimpleTokenizer

def para_has_answer(answer, para, tokenizer):
    text = normalize(para)
    tokens = tokenizer.tokenize(text)
    text = tokens.words(uncased=True)
    assert len(text) == len(tokens)
    for single_answer in answer:
        single_answer = normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False

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

def prepare_path(path, tokenizer, budget, pasg_sep='[SEP]'):
    """
    tokenize the passages chains, add sentence start markers for SP sentence identification
    """
    def _process_p(title, sents, curr):
        para = title + ' '
        curr += len(title) + 1
        sent_starts = []
        num_tokens = len(tokenizer.tokenize(title))
        for idx, sent in enumerate(sents):
            num_tokens += len(tokenizer.tokenize(sent))
            sent_starts.append(curr)
            sent = sent.strip()
            para += sent + ' '
            curr += len(sent) + 1
        return para, sent_starts
    # mark passage boundary
    contexts = f'yes no '
    sent_starts = []
    for i in range(int(len(path)/2)):
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

def qa_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([p for s in samples for p in s['input_ids']], pad_id),
        'token_type_ids': collate_tokens([p for s in samples for p in s['token_type_ids']], 0),
        #'position_ids': collate_tokens([p for s in samples for p in s['position_ids']], 0),
        }
    batch['attention_mask'] = batch['input_ids'].ne(pad_id).long()

    # training labels
    if len(samples[0]['answer_starts']) > 0:
        batch["sent_starts"] = collate_tokens([p for s in samples for p in s['sent_starts']], -1)
        batch['sp_sent_labels'] = collate_tokens([p for s in samples for p in s["sp_sent_labels"]], -1)
        batch["starts"] = collate_tokens([p for s in samples for p in s['answer_starts']], -1)
        batch["ends"] = collate_tokens([p for s in samples for p in s['answer_ends']], -1)
        batch['para_starts'] = collate_tokens([p for s in samples for p in s['para_starts']], 0)
        batch['para_labels'] = collate_tokens([p for s in samples for p in s['para_labels']], -1)
        batch['para_has_answer'] = torch.cat([s['para_has_answer'] for s in samples], dim=0)
    else:
        batch["sent_starts"] = collate_tokens([p for s in samples for p in s['sent_starts']], -1)
        batch['para_starts'] = collate_tokens([p for s in samples for p in s['para_starts']], 0)
    # print (batch["sent_starts"])

    batched = {
        "qids": [s["qid"] for s in samples],
        "gold_answer": [s["gold_answer"] for s in samples],
        "sp_gold": [s["sp_gold"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "chain_titles": [s["chain_titles"] for s in samples],
        "title_list": [s["title_list"] for s in samples],
        "original_text": [s["original_text"] for s in samples],
        "offset_mapping": [s["offset_mapping"] for s in samples],
        "net_inputs": batch,
    }

    return batched


class QADatasetV2Eval(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_seq_len,
        max_q_len,
        train=False,
        debug=False, num_ctx=101,
        world_size=1, global_rank=0
        ):
        if debug:
            retriever_outputs = json.load(open(data_path, 'r'))[:50]
        else:
            retriever_outputs = json.load(open(data_path, 'r'))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.train = train
        self.simple_tok = SimpleTokenizer()
        print(f"Data size {len(retriever_outputs)}")
        self.data = []
        empty_sent = 0
        avg_ctx = 0
        avg_sp = 0
        self.para_token_id = self.tokenizer.convert_tokens_to_ids("[unused2]")
        for si, item in enumerate(retriever_outputs):
            if si % world_size != global_rank:
                continue
            if item["question"].endswith("?"):
                item["question"] = item["question"][:-1]
            q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
            ans_tokens = self.tokenizer.tokenize(item["answers"][0])

            candidate_chains = []
            for chain in item["ctxs"][0:100]:
                chain_titles = {}
                curr = 0
                title_list = []
                for i in range(int(len(chain)/2)):
                    chain[f"hop{i+1} text"] = [x for x in chain[f"hop{i+1} text"] if x.strip() != '']
                    for j in range(len(chain[f"hop{i+1} text"])):
                        chain_titles[curr+j] = [chain[f"hop{i+1} title"], j]
                    curr += len(chain[f"hop{i+1} text"])
                    title_list.append(chain[f"hop{i+1} title"])
                chain_ctx = prepare_path(chain, self.tokenizer, 506-len(q_toks), pasg_sep='[unused2]')
                encoded = self.tokenizer(chain_ctx[0], add_special_tokens=False, return_offsets_mapping=True, max_length=self.max_seq_len - len(q_toks) - 3, truncation=True)
                chain_sent_st = [encoded.char_to_token(cid)+len(q_toks)+2 for cid in chain_ctx[1] if encoded.char_to_token(cid) is not None]
                chain_sent_st.append(len(encoded['input_ids'])+len(q_toks)+2)
                chain_para_st = [i+len(q_toks)+2 for i, t in enumerate(encoded['input_ids']) if t == self.para_token_id]
                chain['sent_st'] = chain_sent_st
                chain['para_st'] = chain_para_st
                chain['original_text'] = chain_ctx[0]
                chain['encoded'] = encoded
                chain['titles'] = chain_titles
                chain['title_list'] = title_list
                candidate_chains.append(chain)
            avg_ctx += len(candidate_chains)
            self.data.append({
                "question": item["question"],
                "q_toks": q_toks,
                "type": item["type"],
                "qid": item["_id"],
                "gold_answer": item["answers"],
                "ans_tokens": ans_tokens,
                "sp_gold": item["sp"] if "sp" in item else None,
                "candidate_chains": candidate_chains,
            })

        print(f"Global rank {global_rank} Data size {len(self.data)}", f"Average context {avg_ctx/len(self.data)}", f"Average sp {avg_sp/len(self.data)}", f"Empty sent {empty_sent}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        q_toks = item['q_toks']
        para_offset = len(q_toks) + 2 # cls and seq
        input_ids = []
        token_type_ids = []
        sent_starts = []
        chain_titles = []
        original_text = []
        offset_mapping = []
        para_starts = []
        title_list = []

        for i, chain in enumerate(item['candidate_chains']):
            chain_titles.append(chain['titles'])
            title_list.append(chain['title_list'])
            encoded = chain['encoded'] 
            original_text.append(chain['original_text'])
            offset_mapping.append(encoded['offset_mapping'])
            sent_starts.append(torch.LongTensor(chain['sent_st']))
            para_starts.append(torch.LongTensor(chain['para_st']))
            input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token]) + encoded['input_ids'] + [self.tokenizer.sep_token_id]))
            token_type_ids.append(torch.LongTensor([0] * (len(q_toks) + 2) + [1] * (len(encoded['input_ids']) + 1)))
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'sent_starts': sent_starts, 'sp_gold': item['sp_gold'], 'qid': item['qid'], 'type': item['type'], 'gold_answer': item['gold_answer'], 'para_offset': para_offset, 'chain_titles': chain_titles, 'original_text': original_text, 'offset_mapping': offset_mapping, 'para_starts': para_starts, 'answer_starts': [], 'title_list': title_list}


class QADatasetV2(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_seq_len,
        max_q_len,
        max_ans_len,
        train=False,
        debug=False, num_ctx=101,
        global_rank=0, world_size=1
        ):
        if debug:
            retriever_outputs = json.load(open(data_path, 'r'))[:50]
        else:
            retriever_outputs = json.load(open(data_path, 'r'))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len
        self.train = train
        self.simple_tok = SimpleTokenizer()
        print(f"Data size {len(retriever_outputs)}")
        self.data = []
        empty_sent = 0
        avg_ctx = 0
        avg_sp = 0
        self.para_token_id = self.tokenizer.convert_tokens_to_ids("[unused2]")
        for si, item in enumerate(retriever_outputs):
            if si % world_size != global_rank:
                continue
            if item["question"].endswith("?"):
                item["question"] = item["question"][:-1]
            q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
            ans_tokens = self.tokenizer.tokenize(item["answers"][0])[:self.max_ans_len]
            gold_chain = item['ctxs'][0]
            sp_sent_labels = []
            sp_gold = item["sp"]
            hop1_sid = [sent[1] for sent in sp_gold if sent[0] == gold_chain['hop1 title']]
            hop2_sid = [sent[1] for sent in sp_gold if sent[0] == gold_chain['hop2 title']]
            gold_chain['hop1 text'] = [x for x in gold_chain['hop1 text'] if x.strip() != '']
            gold_chain['hop2 text'] = [x for x in gold_chain['hop2 text'] if x.strip() != '']
            for i in range(len(gold_chain['hop1 text'])):
                if i in hop1_sid:
                    sp_sent_labels.append(1)
                else:
                    sp_sent_labels.append(0)
            for i in range(len(gold_chain['hop2 text'])):
                if i in hop2_sid:
                    sp_sent_labels.append(1)
                else:
                    sp_sent_labels.append(0)  
            hop1_sp_labels = [x for x in sp_sent_labels[:len(gold_chain['hop1 text'])]]
            hop2_sp_labels = [x for x in sp_sent_labels[len(gold_chain['hop1 text']):]]
            avg_sp += sum(sp_sent_labels)

            sp_titles = item['pos titles']
            ans_titles = set()
            if item["type"] == "bridge":
                if para_has_answer(item["answers"], "".join(gold_chain["hop1 text"]), self.simple_tok):
                    ans_titles.add(gold_chain["hop1 title"])
                if para_has_answer(item["answers"], "".join(gold_chain["hop2 text"]), self.simple_tok):
                    ans_titles.add(gold_chain["hop2 title"])
            # top ranked negative chains
            neg_paras = []
            for chain in item["ctxs"][1:num_ctx]:
                chain_titles = [chain['hop1 title'], chain['hop2 title']]
                if self.train and len(set(chain_titles).intersection(sp_titles)) == 2:
                    continue
                if chain['hop1 title'] not in sp_titles and chain['hop1 title'] not in neg_paras:
                    neg_paras.append((chain['hop1 title'], [x for x in chain['hop1 text'] if x.strip() != ''], 0))
                if chain['hop2 title'] not in sp_titles and chain['hop2 title'] not in neg_paras:
                    neg_paras.append((chain['hop2 title'], [x for x in chain['hop2 text'] if x.strip() != ''], 0))

            avg_ctx += len(neg_paras)
            self.data.append({
                "question": item["question"],
                "q_toks": q_toks,
                "type": item["type"],
                "qid": item["_id"],
                "gold_answer": item["answers"],
                "ans_tokens": ans_tokens,
                "gold_chain": gold_chain,
                "hop_sp_labels": {gold_chain['hop1 title']: hop1_sp_labels, gold_chain['hop2 title']: hop2_sp_labels},
                "neg_paras": neg_paras,
                "ans_titles": ans_titles,
                "sp_gold": sp_gold,
                "sp_sent_dict": {gold_chain['hop1 title']: hop1_sid, gold_chain['hop2 title']: hop2_sid}
            })

        print(f"Global rank {global_rank} Data size {len(self.data)}", f"Average neg paras {avg_ctx/len(self.data)}", f"Average sp {avg_sp/len(self.data)}", f"Empty sent {empty_sent}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        q_toks = item['q_toks']
        para_offset = len(q_toks) + 2 # cls and seq
        ans_tokens = item['ans_tokens']

        input_ids = []
        token_type_ids = []
        answer_starts = []
        answer_ends = []
        sent_starts = []
        sent_labels = []
        chain_titles = []
        original_text = []
        offset_mapping = []
        para_starts = []
        para_labels = []
        para_has_answer = []
        gold_chain = item['gold_chain']
        pa_label = None
        sp_label = None
        if random.random() < 0.8:
            # use 2 paras
            if random.random() < 0.5:
                # use positive chain
                gold_path = [(gold_chain['hop1 title'], gold_chain['hop1 text']), (gold_chain['hop2 title'], gold_chain['hop2 text'])]
                random.shuffle(gold_path)
                gold_chain = {'hop1 title': gold_path[0][0], 'hop1 text': gold_path[0][1], 'hop2 title': gold_path[1][0], 'hop2 text': gold_path[1][1]}
                path = prepare_path(gold_chain, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
                pa_label = [1, 1]
                sp_label = item['hop_sp_labels'][gold_chain['hop1 title']] + item['hop_sp_labels'][gold_chain['hop2 title']]
            else:
                if random.random() < 0.5:
                    # mix in one neg 
                    neg_para = random.choice(item['neg_paras'])
                    noisy_path = [neg_para]
                    pos_para = random.choice([(gold_chain['hop1 title'], gold_chain['hop1 text'], 1), (gold_chain['hop2 title'], gold_chain['hop2 text'], 1)])
                    noisy_path.append(pos_para)
                    random.shuffle(noisy_path)
                    pa_label = [noisy_path[0][2], noisy_path[1][2]]
                    noisy_path = {'hop1 title': noisy_path[0][0], 'hop1 text': noisy_path[0][1], 'hop2 title': noisy_path[1][0], 'hop2 text': noisy_path[1][1]}
                    path = prepare_path(noisy_path, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
                    if noisy_path['hop1 title'] in item['hop_sp_labels']:
                        sp_label = item['hop_sp_labels'][noisy_path['hop1 title']] + [0]*len(noisy_path['hop2 text'])
                    else:
                        sp_label = [0]*len(noisy_path['hop1 text']) + item['hop_sp_labels'][noisy_path['hop2 title']]
                else:
                    # mix in two neg
                    neg_para = random.sample(item['neg_paras'], 2)
                    noisy_path = [neg_para[0], neg_para[1]]
                    pa_label = [0, 0]
                    noisy_path = {'hop1 title': noisy_path[0][0], 'hop1 text': noisy_path[0][1], 'hop2 title': noisy_path[1][0], 'hop2 text': noisy_path[1][1]}
                    path = prepare_path(noisy_path, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
                    sp_label = [0]*len(noisy_path['hop1 text']) + [0]*len(noisy_path['hop2 text'])
        else:
            neg_para = random.choice(item['neg_paras'])
            noisy_path = [neg_para, (gold_chain['hop1 title'], gold_chain['hop1 text'], 1), (gold_chain['hop2 title'], gold_chain['hop2 text'], 1)]
            random.shuffle(noisy_path)
            pa_label = [noisy_path[0][2], noisy_path[1][2], noisy_path[2][2]]
            noisy_path = {'hop1 title': noisy_path[0][0], 'hop1 text': noisy_path[0][1], 'hop2 title': noisy_path[1][0], 'hop2 text': noisy_path[1][1], 'hop3 title': noisy_path[2][0], 'hop3 text': noisy_path[2][1]}
            path = prepare_path(noisy_path, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
            sp_label = []
            for i in range(3):
                if noisy_path[f'hop{i+1} title'] in item['hop_sp_labels']:
                    sp_label += item['hop_sp_labels'][noisy_path[f'hop{i+1} title']]
                else:
                    sp_label += [0]*len(noisy_path[f'hop{i+1} text'])

        encoded = self.tokenizer(path[0], add_special_tokens=False, return_offsets_mapping=True, max_length=self.max_seq_len - len(q_toks) - 3, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        sent_st = [encoded.char_to_token(cid)+len(q_toks)+2 for cid in path[1] if encoded.char_to_token(cid) is not None]
        sent_st.append(len(encoded['input_ids'])+len(q_toks)+2)
        if len(sent_st)-1 != len(sp_label):
            sp_label = sp_label[:len(sent_st)-1]
        if len(sent_st)-1 != len(sp_label):
            print (sent_st)
            print (sp_label)
            print (item['question'])
            print (item['qid'])
            print (noisy_path)
            exit()
        para_st = [i+len(q_toks)+2 for i, t in enumerate(tokens) if t == "[unused2]"]
        if len(para_st) != len(pa_label):
            pa_label = pa_label[:len(para_st)]
    
        ans_starts, ans_ends = get_answer_indices(item["gold_answer"][0], len(q_toks)+2, tokens, ans_tokens)
        if sum(pa_label) != 2:
            ans_starts = [-1]
            ans_ends = [-1]
        sent_starts.append(torch.LongTensor(sent_st))
        para_starts.append(torch.LongTensor(para_st))
        
        sent_labels.append(torch.LongTensor(sp_label))
        para_labels.append(torch.LongTensor(pa_label))
        input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token] + tokens + [self.tokenizer.sep_token])))
        token_type_ids.append(torch.LongTensor([0] * (len(q_toks) + 2) + [1] * (len(tokens) + 1)))
        answer_starts.append(torch.LongTensor(ans_starts))
        answer_ends.append(torch.LongTensor(ans_ends))
        para_has_answer.append(1)

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'answer_starts': answer_starts, 'answer_ends': answer_ends, 'sp_sent_labels': sent_labels, 'sent_starts': sent_starts, 'sp_gold': item['sp_gold'], 'qid': item['qid'], 'type': item['type'], 'gold_answer': item['gold_answer'], 'para_offset': para_offset, 'chain_titles': chain_titles, 'original_text': original_text, 'offset_mapping': offset_mapping, 'para_starts': para_starts, 'para_labels': para_labels, 'para_has_answer': torch.LongTensor(para_has_answer), 'title_list': []}


class QADatasetV2Reader(QADatasetV2):

    def __getitem__(self, index):
        item = self.data[index]
        q_toks = item['q_toks']
        para_offset = len(q_toks) + 2 # cls and seq
        ans_tokens = item['ans_tokens']

        input_ids = []
        token_type_ids = []
        answer_starts = []
        answer_ends = []
        sent_starts = []
        sent_labels = []
        chain_titles = []
        original_text = []
        offset_mapping = []
        para_starts = []
        para_labels = []
        para_has_answer = []
        gold_chain = item['gold_chain']
        pa_label = None
        sp_label = None
        # use 2 paras
        if random.random() < 0.8:
            # use positive chain
            gold_path = [(gold_chain['hop1 title'], gold_chain['hop1 text']), (gold_chain['hop2 title'], gold_chain['hop2 text'])]
            random.shuffle(gold_path)
            gold_chain = {'hop1 title': gold_path[0][0], 'hop1 text': gold_path[0][1], 'hop2 title': gold_path[1][0], 'hop2 text': gold_path[1][1]}
            path = prepare_path(gold_chain, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
            pa_label = [1, 1]
            sp_label = item['hop_sp_labels'][gold_chain['hop1 title']] + item['hop_sp_labels'][gold_chain['hop2 title']]
        else:
            # mix in one neg 
            neg_para = random.choice(item['neg_paras'])
            noisy_path = [neg_para]
            if item['type'] == 'bridge':
                if gold_chain['hop2 title'] in item['ans_titles']:
                    noisy_path.append((gold_chain['hop2 title'], gold_chain['hop2 text'], 1))
                else:
                    noisy_path.append((gold_chain['hop1 title'], gold_chain['hop1 text'], 1))
            else:
                if random.random() < 0.5:
                    noisy_path.append((gold_chain['hop2 title'], gold_chain['hop2 text'], 1))
                else:
                    noisy_path.append((gold_chain['hop1 title'], gold_chain['hop1 text'], 1))
            random.shuffle(noisy_path)
            pa_label = [noisy_path[0][2], noisy_path[1][2]]
            noisy_path = {'hop1 title': noisy_path[0][0], 'hop1 text': noisy_path[0][1], 'hop2 title': noisy_path[1][0], 'hop2 text': noisy_path[1][1]}
            path = prepare_path(noisy_path, self.tokenizer, 508-para_offset, pasg_sep='[unused2]')
            if noisy_path['hop1 title'] in item['hop_sp_labels']:
                sp_label = item['hop_sp_labels'][noisy_path['hop1 title']] + [0]*len(noisy_path['hop2 text'])
            else:
                sp_label = [0]*len(noisy_path['hop1 text']) + item['hop_sp_labels'][noisy_path['hop2 title']]

        encoded = self.tokenizer(path[0], add_special_tokens=False, return_offsets_mapping=True, max_length=self.max_seq_len - len(q_toks) - 3, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        sent_st = [encoded.char_to_token(cid)+len(q_toks)+2 for cid in path[1] if encoded.char_to_token(cid) is not None]
        sent_st.append(len(encoded['input_ids'])+len(q_toks)+2)
        if len(sent_st)-1 != len(sp_label):
            sp_label = sp_label[:len(sent_st)-1]
        if len(sent_st)-1 != len(sp_label):
            print (sent_st)
            print (sp_label)
            print (item['question'])
            print (item['qid'])
            print (noisy_path)
            exit()
        para_st = [i+len(q_toks)+2 for i, t in enumerate(tokens) if t == "[unused2]"]
        if len(para_st) != len(pa_label):
            pa_label = pa_label[:len(para_st)]
    
        ans_starts, ans_ends = get_answer_indices(item["gold_answer"][0], len(q_toks)+2, tokens, ans_tokens)

        sent_starts.append(torch.LongTensor(sent_st))
        para_starts.append(torch.LongTensor(para_st))
        
        sent_labels.append(torch.LongTensor(sp_label))
        para_labels.append(torch.LongTensor(pa_label))
        input_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + q_toks + [self.tokenizer.sep_token] + tokens + [self.tokenizer.sep_token])))
        token_type_ids.append(torch.LongTensor([0] * (len(q_toks) + 2) + [1] * (len(tokens) + 1)))
        answer_starts.append(torch.LongTensor(ans_starts))
        answer_ends.append(torch.LongTensor(ans_ends))
        para_has_answer.append(1)

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'answer_starts': answer_starts, 'answer_ends': answer_ends, 'sp_sent_labels': sent_labels, 'sent_starts': sent_starts, 'sp_gold': item['sp_gold'], 'qid': item['qid'], 'type': item['type'], 'gold_answer': item['gold_answer'], 'para_offset': para_offset, 'chain_titles': chain_titles, 'original_text': original_text, 'offset_mapping': offset_mapping, 'para_starts': para_starts, 'para_labels': para_labels, 'para_has_answer': torch.LongTensor(para_has_answer), 'title_list': []}