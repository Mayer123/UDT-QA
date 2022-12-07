import json
import torch
import csv 
import sys
sys.path.append('../DPR/')
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer
from collections import defaultdict, Counter
import glob
import numpy as np
from tqdm import tqdm
import argparse

def find_row(curr_rows, grounded_cell):
    tgt = -1
    for i, row in enumerate(curr_rows):
        if grounded_cell in row.lower():
            tgt = i
            break
    if tgt == -1:
        grounding_tokens = grounded_cell.split()
        tgt_count = Counter()
        for token in grounding_tokens:
            for i, row in enumerate(curr_rows):
                if token in row.lower():
                    tgt_count[i] += 1
        if len(tgt_count) == 0:
            return -1
        else:
            tgt = max(tgt_count, key=tgt_count.get)
    return tgt

def process_beams(beams, all_passages, original_sample, tokenizer, limit):
    all_included = []
    exist = {}
    failed = 0
    chained = 0
    for beam in beams:
        if beam[0]['id'] not in exist:
            exist[beam[0]['id']] = 1
            all_included.append(beam[0])
            if len(all_included) >= limit:
                break 
        if type(beam[1]) == str:
            if beam[1] in exist:
                continue
            exist[beam[1]] = 1
            content = all_passages[beam[1]]
            title = content[1]
            text = content[0]
            grounded_cell = beam[3]
            curr_rows = beam[0]['text'].split('\n')
            header = curr_rows[0]
            curr_rows = curr_rows[1:]
            tgt = find_row(curr_rows, grounded_cell)
            if tgt != -1:
                full_text = header + '\n' + curr_rows[tgt] + '\n' + title + ' ' + text
                pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                all_included.append({'id':beam[0]['id'], 'title': beam[0]['title'], 'text': full_text, 'has_answer': pasg_has_answer, 'is_chained': True})
                chained += 1
                if len(all_included) == limit:
                    break
            else:
                failed += 1
    return all_included, failed, chained

def rerank_nq_chains(input_path, all_passages, all_links, score_cache):
    data = json.load(open(input_path, 'r'))
    w1 = 10
    w2 = 12
    old_r20 = []
    old_r50 = []
    old_r100 = []
    new_r20 = []
    new_r50 = []
    new_r100 = []
    total_failed = 0
    beam_len = []
    tokenizer = SimpleTokenizer()
    total_chained = 0
    for si, sample in enumerate(tqdm(data)):
        if any([ctx['has_answer'] for ctx in sample['ctxs'][:20]]):
            old_r20.append(1)
        else:
            old_r20.append(0)
        if any([ctx['has_answer'] for ctx in sample['ctxs'][:50]]):
            old_r50.append(1)
        else:
            old_r50.append(0)
        if any([ctx['has_answer'] for ctx in sample['ctxs']]):
            old_r100.append(1)
        else:
            old_r100.append(0)
        step1_results = sample['ctxs']
        step1_scores = [float(ctx['score']) for ctx in step1_results]
        step1_scores = -torch.log_softmax(torch.tensor(step1_scores), dim=0)
        this_scores = score_cache[sample['question']]
        step1_reranker_scores = [this_scores[ctx['id']] for ctx in step1_results]
        beams = []
        for ci, ctx in enumerate(step1_results):
            if ctx['id'] in all_links:
                for link in all_links[ctx['id']]:
                    beams.append((ctx, link[0],step1_scores[ci]+w1*step1_reranker_scores[ci]+w2*this_scores[link[0]], link[1][2]))
            else:
                beams.append((ctx, ctx, step1_scores[ci]+w1*step1_reranker_scores[ci]*2, None))
        beams = sorted(beams, key=lambda x: x[2])
        all_included, failed, chained = process_beams(beams, all_passages, sample, tokenizer, 100)
        assert len(all_included) == 100
        sample['ctxs'] = all_included
        total_failed += failed
        total_chained += chained

        beam_len.append(len(beams))
        if any([ctx['has_answer'] for ctx in all_included[:20]]):
            new_r20.append(1)
        else:
            new_r20.append(0)
        if any([ctx['has_answer'] for ctx in all_included[:50]]):
            new_r50.append(1)
        else:
            new_r50.append(0)
        if any([ctx['has_answer'] for ctx in all_included]):
            new_r100.append(1)
        else:
            new_r100.append(0)

    print ('old r20', np.mean(old_r20), 'old r50', np.mean(old_r50), 'old r100', np.mean(old_r100))
    print ('w1', w1, 'w2', w2, 'new r20', np.mean(new_r20), 'new r50', np.mean(new_r50), 'new r100', np.mean(new_r100))
    print ('total failed', total_failed, 'avg beam len', np.mean(beam_len), 'total chained', total_chained)
    input_dir = '/'.join(input_path.split('/')[:-1])
    if args.output_path == input_dir:
        print ('output dir is the same as input dir, please change output dir otherwise the input file will be overwritten')
        exit()
    output_path = args.output_path + '/' + input_path.split('/')[-1]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_nq_passages(passage_path):
    all_passages = {}
    with open(passage_path) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in tqdm(reader):
            if row[0] == "id":
                continue
            sample_id = 'wiki:' + str(row[0])
            passage = row[1]
            all_passages[sample_id] = (passage, row[2])
    return all_passages

def load_links(links_path):
    table_to_pasg_links = []
    for path in links_path:
        for f in glob.glob(path):
            print (f)
            table_to_pasg_links += json.load(open(f, 'r'))
    all_links = {}
    for sample in tqdm(table_to_pasg_links):
        links = [(res['retrieved'][i], res['scores'][i], ci, res['original_cell']) for ci, res in enumerate(sample['results']) for i in range(len(res['retrieved'][:1]))]
        unique_links = {}
        for l in links:
            if l[0] not in unique_links:
                unique_links[l[0]] = (l[1], l[2], l[3])
            else:
                unique_links[l[0]] = (max(unique_links[l[0]][0], l[1]), l[2], l[3])
        all_links[sample['table_chunk_id']] = sorted(unique_links.items(), key=lambda x: x[1][0], reverse=True)
    print (len(all_links))
    return all_links

def load_score_cache(previous_cache):
    score_cache = defaultdict(dict)
    for cache_path in previous_cache:
        cache_files = glob.glob(cache_path)
        for f in cache_files:
            print (f)
            shard = json.load(open(f, 'r'))
            for k, v in shard.items():
                score_cache[k].update(v)
    print ('score cache', len(score_cache))
    return score_cache

def chain_nq(args):
    all_passages = load_nq_passages(args.passage_path)
    all_links = load_links(args.table_pasg_links_path)
    score_cache = load_score_cache(args.previous_cache)
    rerank_nq_chains(args.retriever_results, all_passages, all_links, score_cache)

def load_ott_passages(passage_path):
    all_passages = json.load(open(passage_path, 'r'))
    new_passages = {}
    for sample in tqdm(all_passages):
        new_passages[sample['title'].lower()] = sample['text']
    all_passages = new_passages
    return all_passages

def process_ott_beams(beams, original_sample, all_table_chunks, all_passages, tokenizer, limit):
    exist = {}
    all_included = []
    em = 0
    recall = 0
    failed = 0
    single_table = 0
    single_pasg = 0
    for beam in beams:
        ctx = beam[0]
        if ctx['title'] not in exist:
            exist[ctx['title']] = 1
            if original_sample and ctx['gold']:
                recall = 1
            if ctx['title'] in all_table_chunks:
                single_table += 1
                content = all_table_chunks[ctx['title']]
                content_text = content['text']
                content_title = content['title']
            else:
                single_pasg += 1
                content_title = ctx['title']
                content_text = all_passages[ctx['title'].lower()]
            if original_sample:
                ctx_has_answer = has_answer(original_sample['answers'], content_text, tokenizer, 'string')
            else:
                ctx_has_answer = False
            if ctx_has_answer:
                em = 1
            all_included.append({'id':ctx['title'], 'title': content_title, 'text': content_text, 'is_gold': ctx['gold'] if original_sample else False, 'has_answer': ctx_has_answer})
            if len(all_included) == limit:
                break
        if type(beam[1]) == str:
            if beam[1] in exist:
                continue
            exist[beam[1]] = 1
            text = all_passages[beam[1].lower()]
            grounded_cell = beam[3]
            curr_rows = all_table_chunks[ctx['title']]['text'].split('\n')
            header = curr_rows[0]
            curr_rows = curr_rows[1:]
            tgt = find_row(curr_rows, grounded_cell)
            if tgt != -1:
                full_text = header + '\n' + curr_rows[tgt] + '\n' + beam[1] + ' ' + text
                if original_sample:
                    pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                else:
                    pasg_has_answer = False
                if pasg_has_answer:
                    em = 1
                all_included.append({'id':ctx['title'], 'title': all_table_chunks[ctx['title']]['title'], 'text': full_text, 'has_answer': pasg_has_answer})
                if len(all_included) == limit:
                    break
            else:
                failed += 1
    return em, recall, all_included, failed, single_table, single_pasg

def rerank_ott_chains(args, all_passages, all_links, score_cache, all_table_chunks):
    retriever_results = json.load(open(args.retriever_results, 'r'))
    step1_tables = 0
    step1_pasgs = 0
    final_tables = 0
    final_pasgs = 0
    w1 = 9
    w2 = 16
    ems = 0
    recalls = 0
    total_failed = 0
    reader_data = []
    beam_sizes = []
    tokenizer = SimpleTokenizer()
    linkable = 0
    unlinked_table = 0
    for si, sample in enumerate(tqdm(retriever_results)):
        step1_results = sample['results']
        for ctx in step1_results:
            if ctx['title'] in all_table_chunks:
                step1_tables += 1
            else:
                step1_pasgs += 1
        q_to_table_scores = [ctx['score'] for ctx in step1_results]
        q_to_table_scores = -torch.log_softmax(torch.tensor(q_to_table_scores), dim=0)
        this_scores = score_cache[sample['question']]
        beams = []
        for ci, ctx in enumerate(step1_results):
            try:
                linked_passages = all_links[ctx['title']]
            except:
                linked_passages = []
            if len(linked_passages) > 0:
                for _, link in enumerate(linked_passages):
                    beams.append([ctx, link[0], q_to_table_scores[ci]+this_scores[link[0]]*w1+this_scores[ctx['title']]*w2, link[1][2]])
                linkable += 1
            else:
                unlinked_table += 1
                beams.append([ctx, ctx, q_to_table_scores[ci]+this_scores[ctx['title']]*w2*2, None])
        beams = sorted(beams, key=lambda x: x[2])
        beam_sizes.append(len(beams))
        em, recall, all_included, failed, single_t, single_p = process_ott_beams(beams, sample if 'answers' in sample else None, all_table_chunks, all_passages, tokenizer, 100)
        ems += em
        recalls += recall
        total_failed += failed
        final_tables += single_t
        final_pasgs += single_p
        assert len(all_included) == 100
        if 'answers' in sample:
            reader_data.append({'question': sample['question'], 'answers': sample['answers'], 'ctxs': all_included})
        else:
            reader_data.append({'question': sample['question'], 'ctxs': all_included})
    print ('linkable', linkable, 'unlinked table', unlinked_table, 'step1_pasgs', step1_pasgs/len(retriever_results), 'step1_tables', step1_tables/len(retriever_results), 'final_pasgs', final_pasgs/len(retriever_results), 'final_tables', final_tables/len(retriever_results))
    print ('Allow all strategy answer EM', ems/len(retriever_results), 'recall', recalls/len(retriever_results), 'total failed', total_failed)
    print ('beam sizes', np.mean(beam_sizes), np.std(beam_sizes))
    with open(args.output_path+f'/ott_{args.split}_reader.json', 'w') as f:
        json.dump(reader_data, f, indent=4)

def chain_ott(args):
    all_passages = load_ott_passages(args.passage_path)
    all_links = load_links(args.table_pasg_links_path)
    score_cache = load_score_cache(args.previous_cache)
    table_chunks = json.load(open(args.table_path, 'r'))
    all_table_chunks = {}
    for chunk in table_chunks:
        all_table_chunks[chunk['chunk_id']] = chunk
    rerank_ott_chains(args, all_passages, all_links, score_cache, all_table_chunks)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, type=str, required=True, choices=['ott', 'nq'])
    parser.add_argument("--retriever_results", default=None, type=str, required=True)
    parser.add_argument("--table_pasg_links_path", nargs='+', required=False)
    parser.add_argument("--passage_path", default=None, type=str, required=False)
    parser.add_argument("--table_path", default=None, type=str, required=False)
    parser.add_argument("--previous_cache", nargs='+', required=False)
    parser.add_argument("--output_path", default=None, type=str, required=False)
    parser.add_argument("--split", default=None, type=str, required=False)
    args = parser.parse_args()
    if args.mode == 'ott':
        chain_ott(args)
    elif args.mode == 'nq':
        chain_nq(args)