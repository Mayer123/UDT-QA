# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import collections
import json
import logging
import os
import random
from datetime import date
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from transformers import ElectraTokenizerFast
import argparse
from qa_dataset_hotpot import qa_collate, QADatasetV2Eval, QADatasetV2, QADatasetV2Reader
from qa_model import QAModelV2
from train_qa_fie import AverageMeter, move_to_cuda, init_distributed_mode, average_main, get_layer_lrs, get_optimizer, load_saved, common_args
import sys
sys.path.append('../DPR')
from dpr.models.optimization import AdamWLayer
from hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")   
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps", default=20000, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=250)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_layer_lr", action="store_true")
    parser.add_argument("--layer_decay", default=0.9, type=float)
    parser.add_argument("--num_ctx", type=int, default=101)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--final-metric", default="joint_f1")
    parser.add_argument("--listmle", action="store_true")
    parser.add_argument("--sentlistmle", action="store_true")
    parser.add_argument("--no_answer", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--hard_em", action="store_true")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--warmup-ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--para-weight", default=1.0, type=float, help="weight of the path loss")
    parser.add_argument("--answer-weight", default=1.0, type=float, help="weight of the answer loss")
    parser.add_argument("--sp-weight", default=1.0, type=float, help="weight of the sp loss")
    return parser.parse_args()

def main():
    args = train_args()
    init_distributed_mode(args)
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-lr{args.learning_rate}-para{args.para_weight}-ans{args.answer_weight}-sp{args.sp_weight}-noans{args.no_answer}-listmle{args.listmle}-sentlistmle{args.sentlistmle}"
    args.output_dir = os.path.join(args.output_dir, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if args.is_main and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.is_distributed:
        torch.distributed.barrier()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if args.is_main:
        logger.info(args)

    logger.info("device %s n_gpu %d distributed training %r local rank %d global rank %d", args.device, args.world_size, bool(args.local_rank != -1), args.local_rank, args.global_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.world_size > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name, additional_special_tokens=['[unused1]', '[unused2]'])
    model = QAModelV2(bert_config, args)

    collate_fc = partial(qa_collate, pad_id=tokenizer.pad_token_id)
    eval_dataset = QADatasetV2Eval(tokenizer, args.predict_file, args.max_seq_len, args.max_q_len, world_size=args.world_size, global_rank=args.global_rank, debug=args.debug, num_ctx=args.num_ctx)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        logger.info(f"Loading model from {args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(args.device)
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        optimizer = get_optimizer(model, args)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_em = 0
        train_loss_meter_rerank = AverageMeter()
        train_loss_meter_span = AverageMeter()
        train_loss_meter_sent = AverageMeter()
        model.train()
        if args.para_weight > 0:
            train_dataset = QADatasetV2(tokenizer, args.train_file, args.max_seq_len, args.max_q_len, args.max_ans_len, train=True, world_size=args.world_size, global_rank=args.global_rank, debug=args.debug, num_ctx=args.num_ctx)
        else:
            logger.info("Using the reader loading strategy")
            train_dataset = QADatasetV2Reader(tokenizer, args.train_file, args.max_seq_len, args.max_q_len, args.max_ans_len, train=True, world_size=args.world_size, global_rank=args.global_rank, debug=args.debug, num_ctx=args.num_ctx)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)

        t_total = args.num_train_steps
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        logger.info(f'Start training.... total number of steps: {t_total}')
        if args.is_main:
            os.system("cp %s %s" % ('train_qa_hotpot.py', os.path.join(args.output_dir, 'train_qa_hotpot.py')))
            os.system("cp %s %s" % ('qa_model.py', os.path.join(args.output_dir, 'qa_model.py')))
            os.system("cp %s %s" % ('qa_dataset_hotpot.py', os.path.join(args.output_dir, 'qa_dataset_hotpot.py')))
        while global_step < t_total:
            if args.verbose:
                train_dataloader = tqdm(train_dataloader)
            for batch in train_dataloader:
                batch_step += 1
                batch_inputs = move_to_cuda(batch["net_inputs"])
                loss, rank_loss, sent_loss = model(batch_inputs)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                rank_loss = average_main(rank_loss, args)
                total_loss = average_main(loss, args)
                sent_loss = average_main(sent_loss, args)
                train_loss_meter_rerank.update(rank_loss.item())
                train_loss_meter_span.update(total_loss.item()-rank_loss.item()-sent_loss.item())
                train_loss_meter_sent.update(sent_loss.item())
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.is_main and global_step % 50 == 0:
                        logger.info(f"Step: {global_step} Rank Loss: {train_loss_meter_rerank.avg} Span Loss: {train_loss_meter_span.avg} Sent Loss: {train_loss_meter_sent.avg}")
                   
                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        metrics = predict(args, model, eval_dataloader, logger, tokenizer)
                        if args.para_weight != 0:
                            em = metrics['path_em']
                        else:
                            em = metrics['joint_f1']
                        logger.info("Step %d em %.2f" % (global_step, em*100))
                        if best_em < em:
                            if args.is_main:
                                logger.info("Saving model with best em %.2f -> em %.2f on step=%d" %
                                            (best_em*100, em*100, global_step))
                                torch.save(model.state_dict(), os.path.join(
                                    args.output_dir, f"checkpoint_best.pt"))
                            with open(os.path.join(args.output_dir, args.save_prediction+f"_rank{args.global_rank}"), "w") as f:
                                json.dump(metrics['results'], f, indent=4)
                            best_em = em
                if global_step >= t_total:
                    break
        logger.info("Training finished!")

    elif args.do_predict:
        metrics = predict(args, model, eval_dataloader, logger, tokenizer, fixed_thresh=None)
        with open(os.path.join(args.output_dir, args.save_prediction+f"_rank{args.global_rank}"), "w") as f:
            json.dump(metrics['results'], f, indent=4)

def predict(args, model, eval_dataloader, logger, tokenizer, fixed_thresh=None):
    model.eval()
    id2answer = collections.defaultdict(list)
    id2gold = {}
    id2goldsp = {}
    acc = 0
    total = 0
    if args.verbose:
        eval_dataloader = tqdm(eval_dataloader)
    qid_count = collections.Counter()
    none_token_id = tokenizer.convert_tokens_to_ids('[unused1]')
    for batch in eval_dataloader:
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        with torch.no_grad():
            outputs = model(batch_to_feed)
            scores = outputs["rank_score"]
            thresholds = outputs["rank_thres"].view(-1)
            scores = scores.sum(dim=1)-thresholds*2
            scores = scores.view(-1)
            sp_scores = outputs["sp_score"]
            if args.sentlistmle:
                sp_scores = sp_scores - outputs['sent_thres']
            sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_starts"][:, 1:].eq(-1), float("-inf")).type_as(sp_scores)
            if not args.sentlistmle:
                sp_scores = sp_scores.sigmoid()
            batch_sp_scores = sp_scores
            start_score = outputs['start_logits']
            end_score = outputs['end_logits']

        span_scores = start_score[:, :, None] + end_score[:, None] 
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), args.max_ans_len)
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), float('-inf')).type_as(span_scores)
        curr = 0
        for qid, ans, sp, offset, chain_titles, text, mappings, t_list in zip(batch['qids'], batch['gold_answer'], batch['sp_gold'], batch['para_offsets'], batch['chain_titles'], batch['original_text'], batch['offset_mapping'], batch['title_list']):
            qid_count[qid] += 1
            this_rank_score = scores[curr:curr+len(chain_titles)]
            this_thres_score = thresholds[curr:curr+len(chain_titles)]
            this_para_score = outputs["rank_score"][curr:curr+len(chain_titles)]
            this_sp_score = batch_sp_scores[curr:curr+len(chain_titles)]
            this_span_scores = span_scores_masked[curr:curr+len(chain_titles)]
            this_input_ids = batch_to_feed['input_ids'][curr:curr+len(chain_titles)]

            curr += len(chain_titles)
            id2gold[qid] = ans
            id2goldsp[qid] = sp
            pred_chain = torch.argmax(this_rank_score).item()

            if sp is not None:
                if len(set([v[0] for v in chain_titles[pred_chain].values()]).intersection(set([s[0] for s in sp]))) == 2:
                    acc += 1
            total += 1
            span_count = 0
            for idx, (chain_rank, chain_thres, para_rank, chain_sp, chain_span_score, chain_ids, titles, chain, tok_map) in enumerate(zip(this_rank_score, this_thres_score, this_para_score, this_sp_score, this_span_scores, this_input_ids, chain_titles, text, mappings)):
                sep_positions = chain_ids.eq(tokenizer.sep_token_id).nonzero().view(-1).tolist()
                para_start_positions = chain_ids.eq(tokenizer.convert_tokens_to_ids('[unused2]')).nonzero().view(-1).tolist()
                sep_positions += para_start_positions
                for pos in sep_positions+para_start_positions:
                    chain_span_score[:pos+1, pos:] = float('-inf')
                sorted_span_score, sorted_span_idx = torch.sort(chain_span_score.view(-1), descending=True)
                topk_answers = []
                for i in range(10):
                    chain_start = sorted_span_idx[i] // chain_span_score.size(1)
                    chain_end = sorted_span_idx[i] % chain_span_score.size(1)
                    chain_start = chain_start - offset 
                    chain_end = chain_end - offset
                    pred_ans = chain[tok_map[chain_start][0]:tok_map[chain_end][1]]
                    topk_answers.append((pred_ans, sorted_span_score[i].item()))

                if args.sentlistmle:
                    cutoff = 0.0
                else:
                    cutoff = 0.55
                pred_sp = []
                for i in range(len(chain_sp)):
                    if chain_sp[i] > cutoff:
                        pred_sp.append(titles[i])
                chain_sp = chain_sp.tolist()
                if len(set([p[0] for p in pred_sp])) != 2:
                    new_pred = []
                    reverse_map = collections.defaultdict(list)
                    max_score_by_d = {}
                    for k, v in titles.items():
                        if int(k) < len(chain_sp):
                            reverse_map[v[0]].append(int(k))
                            max_score_by_d[v[0]] = -999
                    for k, v in reverse_map.items():
                        if all([chain_sp[vv] < cutoff for vv in v]):
                            max_idx = np.argmax(np.array([chain_sp[vv] for vv in v]))
                            new_pred.append((k, int(max_idx)))
                            max_score_by_d[k] = max(max_score_by_d[k], chain_sp[v[max_idx]])
                        else:
                            for vv in v:
                                if chain_sp[vv] > cutoff:
                                    new_pred.append((k, v.index(vv)))
                                    max_score_by_d[k] = max(max_score_by_d[k], chain_sp[vv])
                    max_score_by_d = sorted(max_score_by_d.items(), key=lambda x: x[1], reverse=True)
                    max_score_by_d = [x[0] for x in max_score_by_d[:2]]
                    pred_sp = [x for x in new_pred if x[0] in max_score_by_d]
                if len(topk_answers) == 0:
                    topk_answers.append(('no answer', 0.0))
                id2answer[qid].append({"pred_str": topk_answers[0][0], "pred_sp": pred_sp, "rank_score": chain_rank.item(), "para_score":para_rank.tolist(), "span_score": topk_answers[0][1], 'original_idx': idx, 'threshold': chain_thres.item(), 'chain_title': list(set([v[0] for v in titles.values()])),
                'sp_scores': chain_sp, 'sent_map': titles, 'topk_answers': topk_answers})
    logger.info(f"evaluated {total} questions...")
    results = collections.defaultdict(dict)
    if acc > 0:
        ems, f1s, sp_ems, sp_f1s, joint_ems, joint_f1s = [], [], [], [], [], []
        
        for qid, ans_res in id2answer.items():
            ans_res.sort(key=lambda x: 0.8 * x["rank_score"] + (1 - 0.8) * x["span_score"], reverse=True)
            top_pred = ans_res[0]["pred_str"]
            top_pred_sp = ans_res[0]["pred_sp"]

            results["answer"][qid] = top_pred
            results["sp"][qid] = top_pred_sp
            results['full'][qid] = ans_res
            ems.append(exact_match_score(top_pred, id2gold[qid][0]))
            f1, prec, recall = f1_score(top_pred, id2gold[qid][0])
            f1s.append(f1)

            
            metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
            update_sp(metrics, top_pred_sp, id2goldsp[qid])
            sp_ems.append(metrics['sp_em'])
            sp_f1s.append(metrics['sp_f1'])
            # joint metrics
            joint_prec = prec * metrics["sp_prec"]
            joint_recall = recall * metrics["sp_recall"]
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = ems[-1] * sp_ems[-1]
            joint_ems.append(joint_em)
            joint_f1s.append(joint_f1)

        avg_joint_f1 = sum(joint_f1s)
        avg_joint_em = sum(joint_ems)
        avg_sp_f1 = sum(sp_f1s)
        avg_sp_em = sum(sp_ems)
        avg_f1 = sum(f1s)
        avg_em = sum(ems)
        if args.is_distributed:
            avg_joint_f1 = torch.tensor(avg_joint_f1, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_joint_f1, op=torch.distributed.ReduceOp.SUM)
            avg_joint_f1 = avg_joint_f1.item()
            avg_joint_em = torch.tensor(avg_joint_em, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_joint_em, op=torch.distributed.ReduceOp.SUM)
            avg_joint_em = avg_joint_em.item()
            avg_sp_f1 = torch.tensor(avg_sp_f1, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_sp_f1, op=torch.distributed.ReduceOp.SUM)
            avg_sp_f1 = avg_sp_f1.item()
            avg_sp_em = torch.tensor(avg_sp_em, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_sp_em, op=torch.distributed.ReduceOp.SUM)
            avg_sp_em = avg_sp_em.item()
            avg_f1 = torch.tensor(avg_f1, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_f1, op=torch.distributed.ReduceOp.SUM)
            avg_f1 = avg_f1.item()
            avg_em = torch.tensor(avg_em, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
            avg_em = avg_em.item()
            acc = torch.tensor(acc, dtype=torch.float, device=args.device)
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
            acc = acc.item()
    else:
        for qid, ans_res in id2answer.items():
            ans_res.sort(key=lambda x: 0.8 * x["rank_score"] + (1 - 0.8) * x["span_score"], reverse=True)
            top_pred = ans_res[0]["pred_str"]
            top_pred_sp = ans_res[0]["pred_sp"]

            results["answer"][qid] = top_pred
            results["sp"][qid] = top_pred_sp
            results['full'][qid] = ans_res
        avg_joint_f1 = 0
        avg_joint_em = 0
        avg_sp_f1 = 0
        avg_sp_em = 0
        avg_f1 = 0
        avg_em = 0
        acc = 0
    if args.is_distributed:
        total = torch.tensor(total, dtype=torch.float, device=args.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        total = total.item()
    avg_joint_f1 /= total
    avg_joint_em /= total
    avg_sp_f1 /= total
    avg_sp_em /= total
    avg_f1 /= total
    avg_em /= total
    acc /= total

    if args.is_main:
        logger.info(f"evaluated {total} questions...")
        logger.info(f'answer em: {avg_em}')
        logger.info(f'answer f1: {avg_f1}')
        logger.info(f'sp em: {avg_sp_em}')
        logger.info(f'sp f1: {avg_sp_f1}')
        logger.info(f'joint em: {avg_joint_em}')
        logger.info(f'joint f1: {avg_joint_f1}')
        logger.info(f'path em: {acc}')

    model.train()
    return {"em": avg_em, "f1": avg_f1, "joint_em": avg_joint_em, "joint_f1": avg_joint_f1, "sp_em": avg_sp_em, "sp_f1": avg_sp_f1, "results": results, 'path_em': acc}

if __name__ == "__main__":
    main()
