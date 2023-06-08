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
from qa_dataset_fie import QADatasetNoSP, qa_collate_no_sp
from fie_model import FiEModel
import sys
sys.path.append('../DPR')
from dpr.models.optimization import AdamWLayer
from dpr.utils.dist_utils import all_gather_list
from hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
from torch.cuda.amp import autocast
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_layer_lrs(layer_decay: float, n_layers: int, offset=2):
    """Gets a dict mapping from variable name to layerwise learning rate."""
    # TODO(chenghao): This only supports BERT like models.
    key_to_depths = collections.OrderedDict({
        "embeddings": 0,
        "cls": n_layers + offset,
        "encoder.pooler": n_layers + offset,
        "encode_proj": n_layers + offset,
        "discriminator_predictions": n_layers + offset,
        "encoder.expert_gate.fwd": n_layers // 2 + offset,
    })
    for layer in range(n_layers):
        key_to_depths[f"encoder.layer.{layer}"] = layer + 1

        # TODO(chenghao): Makes this configurable.
        key_to_depths[f"encoder.expert_gate.{layer}"] = layer + 1

    return {
        key: layer_decay ** (n_layers + offset - depth)
        for key, depth in key_to_depths.items()
    }

def get_optimizer(model, args) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    if args.use_layer_lr:
        logging.info("Using layerwise adaptive learning rate")
        name_to_adapt_lr = get_layer_lrs(
            layer_decay=args.layer_decay,
            n_layers=model.encoder.config.num_hidden_layers,
        )
        optimizer_grouped_parameters = []
        logging.info(name_to_adapt_lr)
        for name, param in model.named_parameters():
            update_for_var = False
            for key in name_to_adapt_lr:
                if key in name:
                    update_for_var = True
                    lr_adapt_weight = name_to_adapt_lr[key]
            if not update_for_var:
                #raise ValueError("No adaptive LR for %s" % name)
                logging.info("No adaptive LR for %s" % name)
                lr_adapt_weight = 1.0

            wdecay = args.weight_decay
            if any(nd in name for nd in no_decay):
                # Parameters with no decay.
                wdecay = 0.0
            optimizer_grouped_parameters.append({
                "params": param,
                "weight_decay": wdecay,
                "lr_adapt_weight": lr_adapt_weight,
            })
        if args.use_adam:
            logger.info("adam is not yet supported for layerwise adaptive learning rate")
            exit()
        else:
            optimizer = AdamWLayer(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        logging.info("Using regular learning rate")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if args.use_adam:
            optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def load_saved(model, path):
    state_dict = torch.load(path, map_location='cpu')
    def filter(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def init_distributed_mode(args):
    if args.local_rank != -1:
        args.global_rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.n_gpu_per_node = torch.cuda.device_count() #int(os.environ['NGPU'])
        args.n_nodes = args.world_size // args.n_gpu_per_node
        args.node_id = args.global_rank // args.n_gpu_per_node
        args.is_main = args.node_id == 0 and args.local_rank == 0
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )
        args.is_distributed = True 
    else:
        args.global_rank = 0
        args.world_size = 1
        args.n_gpu_per_node = 1
        args.n_nodes = 1
        args.node_id = 0
        args.is_main = True
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.is_distributed = False 

def average_main(x, args):
    if not args.is_distributed:
        return x
    if args.world_size > 1:
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        #if args.is_main:
        x = x / args.world_size
    return x

def common_args():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--train_file", type=str,
                        default=None)
    parser.add_argument("--predict_file", type=str,
                        default=None)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action="store_true", help="for final test submission")

    # model
    parser.add_argument("--model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=64, type=int)
    parser.add_argument("--max_ans_len", default=30, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--predict_batch_size", default=1,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--save_prediction", default="dev_results.json", type=str)
    return parser

def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")   
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps", default=20000, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=250)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_layer_lr", action="store_true")
    parser.add_argument("--layer_decay", default=0.9, type=float)
    parser.add_argument("--neg-num", type=int, default=5, help="how many neg/distant passage chains to use")
    parser.add_argument("--num_ctx", type=int, default=100)
    parser.add_argument("--num_global_tokens", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--final-metric", default="joint_f1")
    parser.add_argument("--sentlistmle", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--warmup-ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--answer-weight", default=1.0, type=float, help="weight of the answer loss")
    parser.add_argument("--sp-weight", default=1.0, type=float, help="weight of the sp loss")
    return parser.parse_args()

def main():
    args = train_args()
    init_distributed_mode(args)
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-lr{args.learning_rate}-ctx{args.num_ctx}-steps{args.num_train_steps}-ans{args.answer_weight}-sp{args.sp_weight}-sentlistmle{args.sentlistmle}-global_tokens{args.num_global_tokens}"
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

    # define model
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name, additional_special_tokens=['[unused1]', '[unused2]'])
    print (tokenizer.convert_ids_to_tokens([_ for _ in range(500, 500+args.num_global_tokens)]))
    model = FiEModel(bert_config, args)
    
    collate_fc = partial(qa_collate_no_sp, pad_id=tokenizer.pad_token_id)
    eval_dataset = QADatasetNoSP(tokenizer, args.predict_file, args.max_seq_len, args.max_q_len, args.max_ans_len, world_size=args.world_size, global_rank=args.global_rank, neg_num=args.neg_num, debug=args.debug, num_ctx=args.num_ctx)
    pred_func = predict_no_sp

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_em = 0
        train_loss_meter_span = AverageMeter()
        train_loss_meter_sent = AverageMeter()
        model.train()
        train_dataset = QADatasetNoSP(tokenizer, args.train_file, args.max_seq_len, args.max_q_len, args.max_ans_len, train=True, world_size=args.world_size, global_rank=args.global_rank, neg_num=args.neg_num, debug=args.debug, num_ctx=args.num_ctx)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)

        t_total = args.num_train_steps
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        logger.info(f'Start training.... total number of steps: {t_total}')
        if args.is_main:
            os.system("cp %s %s" % ('train_qa_fie.py', os.path.join(args.output_dir, 'train_qa_fie.py')))
            os.system("cp %s %s" % ('fie_model.py', os.path.join(args.output_dir, 'fie_model.py')))
            os.system("cp %s %s" % ('qa_dataset_fie.py', os.path.join(args.output_dir, 'qa_dataset_fie.py')))

        while global_step < t_total:
            if args.verbose:
                train_dataloader = tqdm(train_dataloader)
            for batch in train_dataloader:
                batch_step += 1
                batch_inputs = move_to_cuda(batch["net_inputs"])

                loss, span_loss, sent_loss = model(batch_inputs)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                span_loss = average_main(span_loss, args)
                train_loss_meter_span.update(span_loss.item())
                if sent_loss is not None:
                    sent_loss = average_main(sent_loss, args)
                    train_loss_meter_sent.update(sent_loss.item())
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.is_main and global_step % 50 == 0:
                        logger.info(f"Step: {global_step} Span Loss: {train_loss_meter_span.avg} Sent Loss: {train_loss_meter_sent.avg}")
                   
                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        metrics = pred_func(args, model, eval_dataloader, logger, tokenizer)
                        em = metrics["em"]
                        logger.info("Step %d em %.2f" % (global_step, em*100))
                        if best_em < em:
                            if args.is_main:
                                logger.info("Saving model with best em %.2f -> em %.2f on step=%d" %(best_em*100, em*100, global_step))
                                torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_best.pt"))
                            with open(os.path.join(args.output_dir, args.save_prediction+f"_rank{args.global_rank}"), "w") as f:
                                json.dump(metrics['results'], f, indent=4)
                            best_em = em
                if global_step >= t_total:
                    break
        logger.info("Training finished!")
    elif args.do_predict:
        metrics = pred_func(args, model, eval_dataloader, logger, tokenizer, fixed_thresh=None)
        with open(os.path.join(args.output_dir, args.save_prediction+f"_rank{args.global_rank}"), "w") as f:
            json.dump(metrics['results'], f, indent=4)

def predict_no_sp(args, model, eval_dataloader, logger, tokenizer, fixed_thresh=None):
    model.eval()
    id2answer = {}
    id2gold = {}
    total = 0
    if args.verbose:
        eval_dataloader = tqdm(eval_dataloader)
    qid_count = collections.Counter()
    for batch in eval_dataloader:
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        with torch.no_grad():
            outputs = model(batch_to_feed)
            batch_span_scores = outputs["span_logits"]
            batch_span_positions = outputs["span_positions"]

        curr = 0
        for qid, ans, offset, text, mappings in zip(batch['qids'], batch['gold_answer'], batch['para_offsets'], batch['original_text'], batch['offset_mapping']):
            qid_count[qid] += 1
            this_span_scores = batch_span_scores[curr:curr+len(text)]
            this_input_ids = batch_to_feed['input_ids'][curr:curr+len(text)]
            this_span_positions = batch_span_positions[curr:curr+len(text)]
            seq_len = this_input_ids.size(1)
            this_span_scores = torch.cat(this_span_scores, dim=0)
            this_span_scores = torch.softmax(this_span_scores.view(-1), dim=0)#.view(len(text), seq_len, seq_len)
            curr += len(text)
            id2gold[qid] = ans
            total += 1
            ans_d = collections.defaultdict(float)
            span_count = 0
            for idx, (span_positions, chain_ids, chain, tok_map) in enumerate(zip(this_span_positions, this_input_ids, text, mappings)):
                span_indices = span_positions.eq(2).nonzero()
                chain_span_score = this_span_scores[span_count:span_count+len(span_indices)]
                span_count += len(span_indices)
                sep_positions = chain_ids.eq(tokenizer.sep_token_id).nonzero().view(-1).tolist()

                sorted_span_score, sorted_span_idx = torch.sort(chain_span_score.view(-1), descending=True)
                topk_answers = []
                for i in range(10):
                    chain_start = span_indices[sorted_span_idx[i]][0] 
                    chain_end = span_indices[sorted_span_idx[i]][1]
                    if any([chain_start <= pos and chain_end >= pos for pos in sep_positions]):
                        continue
                    chain_start = chain_start - offset 
                    chain_end = chain_end - offset
                    pred_ans = chain[tok_map[chain_start][0]:tok_map[chain_end][1]]
                    ans_score = sorted_span_score[i].item()
                    ans_d[pred_ans] += ans_score
            final_ans = max(ans_d.items(), key=lambda x: x[1])[0]
            final_score = ans_d[final_ans]
            id2answer[qid] = {"pred_str": final_ans, "span_score": final_score}

    ems, f1s = [], []
    results = collections.defaultdict(dict)
    for qid, ans_res in id2answer.items():
        top_pred = ans_res["pred_str"]
        results["answer"][qid] = top_pred
        results['full'][qid] = ans_res
        ems.append(max([exact_match_score(top_pred, ans) for ans in id2gold[qid]]))
        f1s.append(max([f1_score(top_pred, ans)[0] for ans in id2gold[qid]]))
    avg_f1 = sum(f1s)
    avg_em = sum(ems)
    logger.info(f"Rank {args.global_rank} has evaluated {total} examples, avg em: {avg_em}, avg f1: {avg_f1}")

    if args.is_distributed:
        avg_f1 = torch.tensor(avg_f1, dtype=torch.float, device=args.device)
        logger.info(f"Rank {args.global_rank} avg f1: {avg_f1}")
        torch.distributed.all_reduce(avg_f1, op=torch.distributed.ReduceOp.SUM)
        avg_f1 = avg_f1.item()
        avg_em = torch.tensor(avg_em, dtype=torch.float,  device=args.device)
        torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
        avg_em = avg_em.item()
        total = torch.tensor(total, dtype=torch.float, device=args.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        total = total.item()
    avg_f1 /= total
    avg_em /= total
    if args.is_main:
        logger.info(f"evaluated {total} questions...")
        logger.info(f'answer em: {avg_em}')
        logger.info(f'answer f1: {avg_f1}')
    
    model.train()
    return {"em": avg_em, "f1": avg_f1, "results": results}

if __name__ == "__main__":
    main()
