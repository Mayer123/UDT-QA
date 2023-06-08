#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from dataclasses import dataclass
from re import A
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import normalize_passage, normalize_question
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)


@dataclass
class BiEncoderBatch:
    question_ids: T
    question_segments: T
    question_rep_pos: T
    pid_tensors: T
    context_ids: T
    ctx_segments: T
    is_positive: List
    hard_negatives: List
    full_positive_pids: List
    q_target_expert: int
    ctx_target_expert: int
    ctx_sep_positions: T = None
    span_start_positions: T = None
    span_end_positions: T = None
    def _asdict(self):
        return self.__dict__

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def mean_pooling(token_embeddings, mask):
    if token_embeddings is None:
        return None
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_hard_neg_spans(all_pos_d, start, end, first_row_start, doc_len):
    hard_neg_spans = [(s, e) for s in range(start-1, start+2) for e in range(end-1, end+2)]
    hard_neg_spans = [s for s in hard_neg_spans if s[0] >= first_row_start and s[1] <= doc_len]
    hard_neg_spans = [s for s in hard_neg_spans if s not in all_pos_d]
    hard_neg_spans = [s for s in hard_neg_spans if s[0] <= s[1]]
    all_pos_d = list(sorted(all_pos_d))
    curr_idx = all_pos_d.index((start, end))
    for i in range(curr_idx+1, min(curr_idx+3, len(all_pos_d))):
        if all_pos_d[i][1] - start < 10:
            if (start, all_pos_d[i][1]) not in all_pos_d:
                hard_neg_spans.append((start, all_pos_d[i][1]))
    return hard_neg_spans

class MoEBiEncoderJoint(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        use_cls: bool = False,
        num_expert: int = 1,
        do_rerank: bool = False, 
        do_span: bool = False,
        mean_pool: bool = False
    ):
        super(MoEBiEncoderJoint, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        if self.fix_q_encoder:
            print ('fixing question encoder')
            for param in self.question_model.parameters():
                param.requires_grad = False
        self.use_cls = use_cls
        if do_span:
            self.span_proj = nn.Linear(self.question_model.encoder.config.hidden_size*2, self.question_model.encoder.config.hidden_size)
            self.span_query = nn.Linear(
                self.question_model.encoder.config.hidden_size, 1
            )
        self.num_expert = num_expert
        self.mean_pool = mean_pool
        logger.info("Mean pool: %s", self.mean_pool)
        logger.info("Total number of experts: %d", self.num_expert)

    def compute_reranker_loss(self, logits, labels):
        labels = torch.tensor(labels, device=logits.device)
        logits = logits.view(labels.shape[0], -1)
        loss = F.cross_entropy(logits, labels)
        is_correct = (torch.argmax(logits, dim=1) == labels).sum()
        return loss, is_correct
    
    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
        expert_id=None,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        expert_id=expert_id,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos, 
                    expert_id=expert_id,
                )
        if outputs is not None:
            return outputs
        else:
            return sequence_output, pooled_output, hidden_states, None

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        question_rep_pos: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
        q_target_expert = None, c_target_expert = None, 
        ctx_sep_positions=None, 
        span_start_positions=None, span_end_positions=None
    ) -> Tuple[T, T]:
        if self.num_expert == 1:
            q_target_expert = 0
            c_target_expert = 0
        if question_ids is not None:
            bsz = question_ids.shape[0]
        else:
            bsz = 1
        if q_target_expert is not None: 
            assert not self.question_model.use_infer_expert   # we specify the question target expert
            q_expert_ids = torch.full((bsz,), q_target_expert, dtype=torch.int64)
        else:
            q_expert_ids = None
       
        q_outputs = self.get_representation(
            self.question_model,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=q_expert_ids,
        )
        # All 4 types of input pass through the same encoder
        _q_seq = q_outputs[0]
        if self.mean_pool:
            q_pooled_out = mean_pooling(_q_seq, question_attn_mask)
        else:
            q_pooled_out = q_outputs[1]

        if span_start_positions is not None:
            # Type 4: span prediction 
            # This is span proposal inputs
            start_vecs = []
            end_vecs = []
            for i in range(bsz):
                start_vecs.append(torch.index_select(_q_seq[i], 0, span_start_positions[i]))
                end_vecs.append(torch.index_select(_q_seq[i], 0, span_end_positions[i]))
            start_vecs = torch.stack(start_vecs, dim=0)
            end_vecs = torch.stack(end_vecs, dim=0)
            span_vecs = torch.cat([start_vecs, end_vecs], dim=-1)
            span_vecs = torch.tanh(self.span_proj(span_vecs))
            cells_need_link = self.span_query(span_vecs).squeeze(-1)
            # we overwrite the q_pooled_out and ctx_pooled_out, because ctx_pooled_out is not used in span prediction
            q_pooled_out = self.span_query(q_pooled_out)
            ctx_pooled_out = None
        else:
            cells_need_link = None
            if question_rep_pos is not None:
                # Type 2: this is linker inputs
                cell_reps = torch.sum(_q_seq*question_rep_pos.unsqueeze(-1), dim=1)/question_rep_pos.sum(dim=1).unsqueeze(-1)
                # we overwrite the q_pooled_out 
                if self.use_cls:
                    q_pooled_out = (q_outputs[1]+cell_reps)/2
                else:
                    q_pooled_out = cell_reps
                
            if context_ids is not None:
                bsz = context_ids.shape[0]
            else:
                bsz = 1
            if c_target_expert is not None:
                assert not self.ctx_model.use_infer_expert
                ctx_expert_ids = torch.full((bsz,), c_target_expert, dtype=torch.int64)
            else:
                ctx_expert_ids = None
            ctx_outputs = self.get_representation(
                self.ctx_model, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder, expert_id=ctx_expert_ids
            )
            if self.mean_pool:
                ctx_pooled_out = mean_pooling(ctx_outputs[0], ctx_attn_mask)
            else:
                ctx_pooled_out = ctx_outputs[1]

        if ctx_sep_positions is not None:
            # Type 3: this is for reranker inputs 
            # in this case, the q_pooled_out is not used, should be None
            assert not q_pooled_out
            ctx_sep_rep = []
            for i in range(bsz):
                ctx_sep_rep.append(ctx_outputs[0][i][ctx_sep_positions[i]])
            ctx_sep_rep = torch.stack(ctx_sep_rep, dim=0)
            # we overwrite the ctx_pooled_out
            ctx_pooled_out = (ctx_sep_rep * ctx_outputs[1]).sum(dim=-1)

        entropy_loss = None
        if q_outputs[-1] is not None and ctx_outputs[-1] is not None:
            entropy_loss = torch.concat([q_outputs[-1], ctx_outputs[-1]])

        return q_pooled_out, ctx_pooled_out, cells_need_link, entropy_loss

    def forward_multiple_cells(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        question_rep_pos: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
        q_target_expert = None, c_target_expert = None,
    ) -> Tuple[T, T]:
        if self.num_expert == 1:
            q_target_expert = 0
            c_target_expert = 0
        if question_ids is not None:
            bsz = question_ids.shape[0]
        else:
            bsz = 1

        assert not self.question_model.use_infer_expert   # we specify the question target expert
        q_expert_ids = torch.full((bsz,), q_target_expert, dtype=torch.int64)

        q_outputs = self.get_representation(
            self.question_model,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=q_expert_ids,
        )
        _q_seq = q_outputs[0]
        # here we only have linker inputs, thus we do not need to worry about other cases, q_pooled_out is always cls
        q_pooled_out = q_outputs[1]

        if q_pooled_out is not None:
            all_q_pooled_out = []
            for i in range(q_pooled_out.shape[0]):
                cell_reps = torch.sum(_q_seq[i].unsqueeze(0)*question_rep_pos[i].unsqueeze(-1), dim=1)/question_rep_pos[i].sum(dim=1).unsqueeze(-1)
                if self.use_cls:
                    this_q_pooled_out = (q_pooled_out[i].unsqueeze(0)+cell_reps)/2
                else:
                    this_q_pooled_out = cell_reps
                all_q_pooled_out.append(this_q_pooled_out)
        else:
            all_q_pooled_out = None

        if context_ids is not None:
            bsz = context_ids.shape[0]
        else:
            bsz = 1
       
        assert not self.ctx_model.use_infer_expert
        ctx_expert_ids = torch.full((bsz,), c_target_expert, dtype=torch.int64)

        ctx_outputs = self.get_representation(
            self.ctx_model, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder, expert_id=ctx_expert_ids
        )
        if self.mean_pool:
            ctx_pooled_out = mean_pooling(ctx_outputs[0], ctx_attn_mask)
        else:
            ctx_pooled_out = ctx_outputs[1]

        return all_q_pooled_out, ctx_pooled_out, None

    @classmethod
    def create_biencoder_input_rerank(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        hard_neg_fallback: bool = True,
        epoch: int = -1, 
    ) -> BiEncoderBatch:
        ctx_tensors = []
        ctx_sep_positions = []
        positive_ctx_indices = []
        exp_id = 4
        for sample in samples:
            if epoch != -1:
                # During training we take the positives based on the current epoch 
                positive_ctxs = sample['positive_ctxs']
                epoch_i = epoch % len(sample['positive_ctxs'])
                positive_ctx = positive_ctxs[epoch_i]
            else:
                # During dev we take the first one 
                positive_ctx = sample['positive_ctxs'][0]
            neg_ctxs = [c for c in sample['negative_ctxs']] if 'negative_ctxs' in sample else []
            hard_neg_ctxs = [c for c in sample['hard_negative_ctxs']] if 'hard_negative_ctxs' in sample else []
            hard_neg_ctxs = [c for c in hard_neg_ctxs if c['hop'] == positive_ctx['hop']]
            if len(hard_neg_ctxs) < num_hard_negatives:
                print ('WARNING: not enough hard negatives, skipping this example')
                continue
            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(normalize_passage(ctx['text']), title=sample['question'])
                for ctx in all_ctxs
            ]
            sep_positions = [(ctx_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0] for ctx_tensor in sample_ctxs_tensors]
            ctx_tensors.extend(sample_ctxs_tensors)
            ctx_sep_positions.extend(sep_positions)
            positive_ctx_indices.append(0)

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        ctx_segments = torch.zeros_like(ctxs_tensor)
        ctx_sep_positions = torch.tensor(ctx_sep_positions, dtype=torch.long)

        return BiEncoderBatch(
            None,
            None,
            None,
            None,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            None,
            None,
            None, exp_id, ctx_sep_positions=ctx_sep_positions
        )
    
    @classmethod
    def create_biencoder_input2(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None, epoch: int = -1, full_pos=False, 
    ) -> BiEncoderBatch:
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        pid_tensors = []
        full_pos_pids = []
        q_exp_id = 0
        c_exp_id = 1
        for sample in samples:
            if epoch != -1:
                # During training we take the positives based on the current epoch 
                positive_ctxs = sample['positive_ctxs']
                epoch_i = epoch % len(sample['positive_ctxs'])
                positive_ctx = positive_ctxs[epoch_i]
            else:
                # During dev we take the first one 
                positive_ctx = sample['positive_ctxs'][0]
            full_pos_pids.append([int(ctx['passage_id']) for ctx in sample['positive_ctxs']])
            if 'other_positive_ctxs' in sample:
                full_pos_pids[-1] += [int(ctx['passage_id']) for ctx in sample['other_positive_ctxs']]
            neg_ctxs = [c for c in sample['negative_ctxs']] if 'negative_ctxs' in sample else []
            hard_neg_ctxs = [c for c in sample['hard_negative_ctxs']] if 'hard_negative_ctxs' in sample else []
            question = normalize_question(sample['question'])
            if 'pos_question' in positive_ctx:
                # This is expanded query
                question = normalize_question(positive_ctx['pos_question'])
                q_exp_id = 4
                if 'hard_negative_ctxs' in positive_ctx:
                    hard_neg_ctxs += [c for c in positive_ctx['hard_negative_ctxs']]

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            pid_tensors.extend([int(ctx['passage_id']) for ctx in all_ctxs])
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    normalize_passage(ctx['text']), title=ctx['title'] if (insert_title and ctx['title']) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            q_tensor = tensorizer.text_to_tensor(question)
            question_tensors.append(q_tensor)

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        pid_tensors = torch.tensor(pid_tensors, dtype=torch.long)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            None,
            pid_tensors,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            full_pos_pids if full_pos else None, q_exp_id, c_exp_id
        )

    @classmethod
    def create_biencoder_input_span_proposal(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None, epoch: int = -1,
    ) -> BiEncoderBatch:
        question_tensors = []
        span_start_positions = []
        span_end_positions = []
        span_labels = []

        for sample in samples:
            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['bert_end'] < 255]
            random.shuffle(positive_ctxs)
            if len(positive_ctxs) == 0:
                print ('no positive ctx, skipping this example')
                continue
            question = sample['question']
            q_tensor = tensorizer.text_to_tensor(question)
            label_tensor = torch.zeros_like(q_tensor)
            if epoch != -1:
                epoch_i = epoch % len(positive_ctxs)
                selected_pos = positive_ctxs[epoch_i]
            else:
                # During dev we take the first one 
                selected_pos = positive_ctxs[0]
            if 'additional_positive_ctxs' in sample:
                positive_ctxs += sample['additional_positive_ctxs']
            if 'first row start' in positive_ctxs[0]:
                first_row_start = positive_ctxs[0]['first row start']
            else:
                if '[SEP]' in question:
                    first_row_start = (q_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]+1
                else:
                    first_row_start = 1
            selected_start = None
            selected_end = None
            all_pos_d = set()
            for pos in positive_ctxs:
                if pos['bert_end'] >= 255:
                    continue
                ground = torch.tensor(tensorizer.tokenizer.encode(pos['grounding'], add_special_tokens=False), dtype=torch.long)
                start = pos['bert_start']
                end = pos['bert_end']
                if not q_tensor[pos['bert_start']:pos['bert_end']+1].equal(ground):
                    start = 0
                    for i in range(first_row_start, len(q_tensor)-len(ground)):
                        if torch.equal(q_tensor[i:i+len(ground)],ground):
                            start = i
                            break
                    end = start + len(ground)-1
                    if start == 0:
                        print (tensorizer.tokenizer.decode(q_tensor))
                        print (tensorizer.tokenizer.decode(ground))
                        print (pos)
                        print ('cannot find cell')
                if start != 0 and end != 0:
                    all_pos_d.add((start, end))
                    label_tensor[start:end+1] = 1
                    if pos == selected_pos:
                        selected_start = start
                        selected_end = end

            if selected_start is None:
                print ('cannot find positive')
                print (tensorizer.tokenizer.decode(q_tensor))
                print (selected_pos['grounding'], selected_pos['bert_start'], selected_pos['bert_end'])
                continue
            padding_start = (q_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[1][0]+1
            all_neg_spans = [(i, j) for i in range(first_row_start, padding_start) for j in range(i, min(i+10, padding_start)) if (i, j) not in all_pos_d]
            label_tensor[:first_row_start] = -100
            label_tensor[q_tensor==tensorizer.tokenizer.pad_token_id] = -100
            hard_neg_spans = get_hard_neg_spans(all_pos_d, selected_start, selected_end, first_row_start, len(q_tensor))
            reg_neg_num = 15 - len(hard_neg_spans)

            reg_neg_spans = random.sample(all_neg_spans, reg_neg_num)
            hard_neg_spans += reg_neg_spans
            span_start_positions.append([selected_start]+[s[0] for s in hard_neg_spans])
            span_end_positions.append([selected_end]+[s[1] for s in hard_neg_spans])
            question_tensors.append(q_tensor)
            span_labels.append(label_tensor)

        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
        span_labels = torch.cat([q.view(1, -1) for q in span_labels], dim=0)
        question_segments = torch.zeros_like(questions_tensor)
        span_start_positions = torch.tensor(span_start_positions, dtype=torch.long)
        span_end_positions = torch.tensor(span_end_positions, dtype=torch.long)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            None,
            None,
            None,
            None,
            None,
            None,
            None, 5, None, span_start_positions=span_start_positions, span_end_positions=span_end_positions
        )

    @classmethod
    def create_biencoder_input_use_cell(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None, epoch: int = -1, mean_pool=False
    ) -> BiEncoderBatch:
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        question_rep_pos = []
        pid_tensors = []

        for sample in samples:
            question = sample['question']
            q_tensor = tensorizer.text_to_tensor(question)
            
            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['first occurance']]
            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['bert_end'] < 255]
            if len(positive_ctxs) == 0:
                print ('no positive ctx, skipping this example')
                continue
            if epoch != -1:
                # During training we take the positives based on the current epoch 
                epoch_i = epoch % len(positive_ctxs)
                positive_ctx = positive_ctxs[epoch_i]
            else:
                # During dev we take the first one 
                positive_ctx = positive_ctxs[0]
            pos_cell = positive_ctx['grounding']
            pos_pid = positive_ctx['passage_id']
            neg_ctxs = []
            hard_neg_ctxs = positive_ctx['hard_negative_ctxs']
            if 'hard_negative_pool' in sample:
                hard_neg_ctxs = [sample['hard_negative_pool'][str(ctx)] for ctx in hard_neg_ctxs]
            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            assert len(hard_neg_ctxs) > 0
            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
            hard_neg_pids = [ctx['passage_id'] for ctx in hard_neg_ctxs]

            assert len(neg_ctxs) == 0
            all_pids = [pos_pid] + hard_neg_pids
            pid_tensors.extend(all_pids)

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    normalize_passage(ctx['text']), title=ctx['title'] if (insert_title and ctx['title']) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )
            cell_ids = tensorizer.tokenizer.encode(pos_cell, add_special_tokens=False)
            start = positive_ctx['bert_start']
            end = positive_ctx['bert_end']
            rep_pos = torch.zeros(len(q_tensor))
            if q_tensor[start:end+1].tolist() != cell_ids:
                if 'first row start' in sample['positive_ctxs'][0]:
                    first_row_start = sample['positive_ctxs'][0]['first row start']
                else:
                    if '[SEP]' in question:
                        first_row_start = (q_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]+1
                    else:
                        first_row_start = 1
                start = 0
                for i in range(first_row_start, len(q_tensor)-len(cell_ids)):
                    if torch.equal(q_tensor[i:i+len(cell_ids)],torch.tensor(cell_ids)):
                        start = i
                        break
                end = start + len(cell_ids)-1
                if start == 0:
                    print (q_tensor)
                    print (cell_ids)
                    print ('cannot find cell')
            if mean_pool:
                rep_pos[start:end+1] = 1   # change to mean pooling
            else:
                rep_pos[start] = 1
                rep_pos[end] = 1
            question_rep_pos.append(rep_pos)
            question_tensors.append(q_tensor)

        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        question_rep_pos = torch.stack(question_rep_pos)
        pid_tensors = torch.tensor(pid_tensors, dtype=torch.long)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            question_rep_pos,
            pid_tensors,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            None, 2, 3
        )

    @classmethod
    def create_biencoder_input_all_positives_use_cell(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True, mean_pool=False
    ) -> BiEncoderBatch:
        """
        This is only used for validation average rank 
        """
        question_tensors = []
        question_rep_pos = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        pid_tensors = []

        for sample in samples:
            question = sample['question']
            q_tensor = tensorizer.text_to_tensor(question)
            if 'first row start' in sample['positive_ctxs'][0]:
                first_row_start = sample['positive_ctxs'][0]['first row start']
            else:
                if '[SEP]' in question:
                    first_row_start = (q_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]+1
                else:
                    first_row_start = 1
            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['first occurance']]
            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['bert_end'] < 255]
            if len(positive_ctxs) == 0:
                print ('no positive ctx, skipping this example')
                continue
            this_question = []
            for pos in positive_ctxs:
                ground = torch.tensor(tensorizer.tokenizer.encode(pos['grounding'], add_special_tokens=False), dtype=torch.long)
                start = pos['bert_start']
                end = pos['bert_end']
                rep_pos = torch.zeros(len(q_tensor))
                if not q_tensor[pos['bert_start']:pos['bert_end']+1].equal(ground):
                    start = 0
                    for i in range(first_row_start, len(q_tensor)-len(ground)):
                        if torch.equal(q_tensor[i:i+len(ground)],ground):
                            start = i
                            break
                    end = start + len(ground)-1
                    if start == 0:
                        print (tensorizer.tokenizer.decode(q_tensor))
                        print (tensorizer.tokenizer.decode(ground))
                        print (pos)
                        print ('cannot find cell')
                if mean_pool:
                    rep_pos[start:end+1] = 1   # change to mean pooling
                else:
                    rep_pos[start] = 1
                    rep_pos[end] = 1
                this_question.append(rep_pos)
            question_rep_pos.append(torch.stack(this_question))
            question_tensors.append(q_tensor)
            
            neg_ctxs = []
            hard_neg_ctxs = []
            for pos in positive_ctxs:
                hard_neg_ctxs.extend(pos['hard_negative_ctxs'])
            if 'hard_negative_pool' in sample:
                hard_neg_ctxs = [sample['hard_negative_pool'][str(ctx)] for ctx in hard_neg_ctxs]

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
            all_ctxs = positive_ctxs + neg_ctxs + hard_neg_ctxs
            all_pids = [ctx['passage_id'] for ctx in all_ctxs]
            pid_tensors.extend(all_pids)

            hard_negatives_start_idx = len(positive_ctxs)
            hard_negatives_end_idx = len(all_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    normalize_passage(ctx['text']), title=ctx['title'] if (insert_title and ctx['title']) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append([current_ctxs_len + i for i in range(len(positive_ctxs))])
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)
        pid_tensors = torch.LongTensor(pid_tensors)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            question_rep_pos,
            pid_tensors,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            None, 2, 3
        )

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        if "question_model.encoder.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.encoder.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.encoder.embeddings.position_ids"]
        
        # KM: here we just assume that pretrained model has the same expert layer assignment with the finetuned model 
        # we just need to load the newly added expert layer with bert weights 
        mapping_d = {
            'moe_query.4.': 'moe_query.0.',
            'moe_key.4.': 'moe_key.0.',
            'moe_value.4.': 'moe_value.0.',
            'moe_dense.4.': 'moe_dense.0.',
        }
        for k, v in self.state_dict().items():
            if k not in saved_state.model_dict:
                target_k = k
                if 'moe' in target_k:
                    for k1, v1 in mapping_d.items():
                        target_k = target_k.replace(k1, v1)
                if target_k in saved_state.model_dict:
                    logger.info("mapping %s to %s", k, target_k)
                    saved_state.model_dict[k] = saved_state.model_dict[target_k]
                else:
                    logger.info("saved state missing %s", k)
                    saved_state.model_dict[k] = v
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()