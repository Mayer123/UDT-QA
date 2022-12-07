from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json 
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict
import time
import os
import glob
from torch.cuda.amp import autocast

def load_model():
    model_name = "bigscience/T0_3B"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model.parallelize()
    model.to("cuda:0")
    print("Moved model to GPUs")
    return model, tokenizer

def compute_scores(model, tokenizer, triples, output_path, b_size, num_shards, shard_id, task):
    shard_id = int(shard_id)
    shard_size = int(len(triples)/int(num_shards))
    print ('shard size', shard_size)
    if shard_id != int(num_shards)-1:
        start = shard_id*shard_size
        end = (shard_id+1)*shard_size
    else:
        start = shard_id*shard_size
        end = len(triples)
    print ('working on shard from', start, 'to', end)
    triples = triples[start:end]
    triples = sorted(triples, key=lambda x: len(x[1]), reverse=True)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    suffix = tokenizer(' Please write a question based on this passage.')['input_ids']
    score_cache = defaultdict(dict)
    for i in tqdm(range(0, len(triples), b_size)):
        batch = triples[i:i+b_size]    
        ctxs = []
        questions = []
        for sample in batch:
            ctxs.append('Passage: '+ sample[0] +': '+ sample[1])
            questions.append(sample[2])
        
        inputs = tokenizer(ctxs, truncation=True, max_length=512-len(suffix))
        targets = tokenizer(questions)

        full_inputs = []
        for inp in inputs['input_ids']:
            full_inputs.append(inp[:-1]+suffix)
        max_len = max([len(inp) for inp in full_inputs])
        inputs_ids = torch.full((len(full_inputs), max_len), tokenizer.pad_token_id, dtype=torch.long).to("cuda:0")
        attention_mask = torch.zeros((len(full_inputs), max_len), dtype=torch.long).to("cuda:0")
        max_q_len = max([len(inp) for inp in targets['input_ids']])
        target_ids = torch.full((len(targets['input_ids']), max_q_len), -100, dtype=torch.long).to("cuda:0")

        for i, inp in enumerate(full_inputs):
            inputs_ids[i, :len(inp)] = torch.tensor(inp, dtype=torch.long, device="cuda:0")
            attention_mask[i, :len(inp)] = 1
            target_ids[i, :len(targets['input_ids'][i])] = torch.tensor(targets['input_ids'][i], dtype=torch.long, device="cuda:0")

        decoder_inputs = model._shift_right(target_ids)
        decoder_inputs[decoder_inputs == -100] = tokenizer.pad_token_id
        outputs = model(input_ids=inputs_ids, attention_mask=attention_mask, decoder_input_ids=decoder_inputs)
        q_len = target_ids.shape[1]
        loss = loss_fct(outputs[0].view(-1, outputs[0].size(-1)), target_ids.view(-1))
        loss = loss.view(-1, q_len)
        loss = loss.sum(dim=1)/((target_ids != -100).sum(dim=1))
        for _, sample in enumerate(batch):
            if sample[2] not in score_cache:
                score_cache[sample[2]] = {}
            score_cache[sample[2]][sample[3]] = loss[_].item()

    if os.path.exists(f'{output_path}/{task}_score_cache_shard{shard_id}.json'):
        print ('overwritting score cache')
        old_cache = json.load(open(f'{output_path}/{task}_score_cache_shard{shard_id}.json', 'r'))
        for k, v in old_cache.items():
            score_cache[k].update(v)
    with open(f'{output_path}/{task}_score_cache_shard{shard_id}.json', 'w') as f:
        json.dump(score_cache, f)

def rerank_passages_nq(model, tokenizer, retriever_results, previous_cache_path, output_path, b_size, num_shards, shard_id):
    if type(retriever_results) == str:
        data = json.load(open(retriever_results, 'r'))
    else:
        data = []
        for f in retriever_results:
            print (f)
            data += json.load(open(f, 'r'))
    if previous_cache_path:
        score_cache = defaultdict(dict)
        all_cache_files = []
        for cache_path in previous_cache_path:
            all_cache_files.extend(glob.glob(cache_path))
        for f in all_cache_files:
            print ('loading', f)
            shard = json.load(open(f, 'r'))
            for k, v in shard.items():
                score_cache[k].update(v)
        print ('score cache', len(score_cache))
    else:
        score_cache = {}
    triples = []
    covered = 0
    for sample in data:
        question = sample['question'] 
        for ctx in sample['ctxs']:
            if question not in score_cache or ctx['id'] not in score_cache[question]:
                triples.append([ctx['title'], ctx['text'], question, ctx['id']])
            else:
                covered += 1
    print ('passages covered', covered)
    print ('triples to rerank', len(triples))
    compute_scores(model, tokenizer, triples, output_path, b_size, num_shards, shard_id, 'nq')

def collect_passages_for_rerank(retriever_results, table_pasg_links_path, previous_cache_path, passage_path=None):
    if type(retriever_results) == str:
        data = json.load(open(retriever_results, 'r'))
    else:
        data = []
        for f in retriever_results:
            print (f)
            data += json.load(open(f, 'r'))
    table_to_pasg = []
    for prev_path in table_pasg_links_path:
        print (prev_path)
        chunk_to_pasg_files = glob.glob(prev_path)
        for f in chunk_to_pasg_files:
            print (f)
            table_to_pasg.extend(json.load(open(f, 'r')))
        print ('table to pasg', len(table_to_pasg))
    
    all_links = {}
    for sample in tqdm(table_to_pasg):
        if len(sample['results']) == 0:
            continue
        max_scores = max([res['scores'][0] for res in sample['results']])
        links = [(res['retrieved'][i], res['scores'][i]/max_scores, ci) for ci, res in enumerate(sample['results']) for i in range(len(res['retrieved'][:1]))]
        unique_links = {}
        for l in links:
            if l[0] not in unique_links:
                unique_links[l[0]] = (l[1], l[2])
            else:
                unique_links[l[0]] = (max(unique_links[l[0]][0], l[1]), l[2])
        all_links[sample['table_chunk_id']] = sorted(unique_links.items(), key=lambda x: x[1][0], reverse=True)
    print ('all links', len(all_links))

    if previous_cache_path:
        score_cache = defaultdict(dict)
        all_cache_files = []
        for cache_path in previous_cache_path:
            all_cache_files.extend(glob.glob(cache_path))
        for f in all_cache_files:
            print ('loading', f)
            shard = json.load(open(f, 'r'))
            for k, v in shard.items():
                score_cache[k].update(v)
        print ('score cache', len(score_cache))
    else:
        score_cache = {} 

    if passage_path:
        all_passages = json.load(open(passage_path, 'r'))
        new_passages = {}
        for sample in tqdm(all_passages):
            new_passages[sample['title']] = sample['text']
        all_passages = new_passages
        print ('all_passages', len(all_passages))
    else:
        all_passages = {}
    included_passages = defaultdict(list)
    covered = 0
    for sample in data:
        question = sample['question']
        results = sample['results'] if 'results' in sample else sample['ctxs']
        for ctx in results:
            try:
                linked_passages = all_links[ctx['title']] if 'id' not in ctx else all_links[ctx['id']]
            except:
                linked_passages = []
            for lp in linked_passages:
                if question not in score_cache or lp[0] not in score_cache[question]:
                    if question not in included_passages[lp[0]]:
                        included_passages[lp[0]].append(question)
                else:
                    covered += 1
            if 'title' in ctx and ctx['title'] in all_passages:
                if question not in score_cache or ctx['title'] not in score_cache[question]:
                    if question not in included_passages[ctx['title']]:
                        included_passages[ctx['title']].append(question)
                else:
                    covered += 1

    print ('covered:', covered)
    print ('num unique passages:', len(included_passages))
    print ('num passages', sum([len(v) for v in included_passages.values()]))
    return included_passages

def collect_tables_for_rerank(retriever_results, previous_cache_path):
    if type(retriever_results) == str:
        data = json.load(open(retriever_results, 'r'))
    else:
        data = []
        for f in retriever_results:
            print (f)
            data += json.load(open(f, 'r'))

    if previous_cache_path:
        score_cache = defaultdict(dict)
        all_cache_files = []
        for cache_path in previous_cache_path:
            all_cache_files.extend(glob.glob(cache_path))
        for f in all_cache_files:
            print ('loading', f)
            shard = json.load(open(f, 'r'))
            for k, v in shard.items():
                score_cache[k].update(v)
        print ('score cache', len(score_cache)) 
    else:
        score_cache = {}
    
    included_tables = defaultdict(list)
    covered = 0
    for sample in data:
        question = sample['question']
        for ctx in sample['results']:
            if question not in score_cache or ctx['title'] not in score_cache[question]:
                if question not in included_tables[ctx['title']]:
                    included_tables[ctx['title']].append(question)
            else:
                covered += 1
    print ('covered:', covered)
    print ('num unique tables:', len(included_tables))
    print ('num tables', sum([len(v) for v in included_tables.values()]))
    return included_tables

def rerank_passages(model, tokenizer, included_passages, passage_path, output_path, b_size, num_shards, shard_id):
    all_passages = json.load(open(passage_path, 'r'))
    new_passages = {}
    for sample in tqdm(all_passages):
        new_passages[sample['title'].lower()] = sample['text']
    all_passages = new_passages
    print ('all_passages', len(all_passages))

    triples = []
    for k, v in tqdm(included_passages.items()):
        for q in v:
            triples.append((k, all_passages[k.lower()], q, k))
    print ('triples', len(triples))
    compute_scores(model, tokenizer, triples, output_path, b_size, num_shards, shard_id, 'ott')

import csv
def rerank_passages_nq_link(model, tokenizer, included_passages, passage_path, output_path, b_size, num_shards, shard_id):
    all_passages = {}
    with open(passage_path) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            if row[0] == "id":
                continue
            sample_id = 'wiki:' + str(row[0])
            passage = row[1]
            all_passages[sample_id] = (passage, row[2])
    print ('all_passages', len(all_passages))

    triples = []
    for k, v in tqdm(included_passages.items()):
        for q in v:
            content = all_passages[k]
            triples.append((content[1], content[0], q, k))
    print ('triples', len(triples))
    compute_scores(model, tokenizer, triples, output_path, b_size, num_shards, shard_id, 'nq')    

def rerank_tables(model, tokenizer, included_tables, table_path, output_path, b_size, num_shards, shard_id):
    table_chunks = json.load(open(table_path, 'r'))
    all_table_chunks = {}
    for chunk in table_chunks:
        all_table_chunks[chunk['chunk_id']] = chunk
    triples = []
    for k, v in tqdm(included_tables.items()):
        for q in v:
            triples.append((all_table_chunks[k]['title'], all_table_chunks[k]['text'], q, k))
    print ('triples', len(triples))
    compute_scores(model, tokenizer, triples, output_path, b_size, num_shards, shard_id, 'ott')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", default=1, type=int, required=False)
    parser.add_argument("--shard_id", default=0, type=int, required=False)
    parser.add_argument("--retriever_results", nargs='+', required=True)
    parser.add_argument("--table_pasg_links_path", nargs='+', required=False)
    parser.add_argument("--passage_path", default=None, type=str, required=False)
    parser.add_argument("--previous_cache", nargs='+', required=False)
    parser.add_argument("--output_path", default=None, type=str, required=True)
    parser.add_argument("--b_size", default=25, type=int, required=False)
    parser.add_argument('--do_tables', default=False, action="store_true")
    parser.add_argument('--nq', default=False, action="store_true")
    parser.add_argument('--nq_link', default=False, action="store_true")
    args = parser.parse_args()
    if len(args.retriever_results) == 1:
        args.retriever_results = args.retriever_results[0]
    model, tokenizer = load_model()
    print("Model and tokenizer loaded")
    with torch.no_grad():
        with autocast():
            if not args.do_tables:
                if args.nq:
                    if not args.nq_link:
                        rerank_passages_nq(model, tokenizer, args.retriever_results, args.previous_cache, args.output_path, args.b_size, args.num_shards, args.shard_id)
                    else:
                        included_passages = collect_passages_for_rerank(args.retriever_results, args.table_pasg_links_path, args.previous_cache)  
                        rerank_passages_nq_link(model, tokenizer, included_passages, args.passage_path, args.output_path, args.b_size, args.num_shards, args.shard_id)
                else:
                    included_passages = collect_passages_for_rerank(args.retriever_results, args.table_pasg_links_path, args.previous_cache, args.passage_path)  
                    rerank_passages(model, tokenizer, included_passages, args.passage_path, args.output_path, args.b_size, args.num_shards, args.shard_id)
            else:
                included_tables = collect_tables_for_rerank(args.retriever_results, args.previous_cache)
                rerank_tables(model, tokenizer, included_tables, args.passage_path, args.output_path, args.b_size, args.num_shards, args.shard_id)