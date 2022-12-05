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
    encoder_type: str
    question_labels: T
    full_positive_pids: List

    def _asdict(self):
        return self.__dict__

# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


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


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoderTableLink(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        label_question: bool = False,
    ):
        super(BiEncoderTableLink, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        if self.fix_q_encoder:
            print ('fixing question encoder')
            for param in self.question_model.parameters():
                param.requires_grad = False
        if label_question:
            self.label_question = True
            self.question_labeler = nn.Linear(
                self.question_model.encoder.config.hidden_size, 2
            )
            self.ctx_model = None
        else:
            self.label_question = False

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        question_rep_pos: T,
        pid_tensors: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )
        if self.label_question:
            cells_need_link = self.question_labeler(_q_seq)
            q_pooled_out = None
            ctx_pooled_out = None
        else:
            cells_need_link = None
            cell_reps = torch.sum(_q_seq*question_rep_pos.unsqueeze(-1), dim=1)/question_rep_pos.sum(dim=1).unsqueeze(-1)
            q_pooled_out = cell_reps
            ctx_encoder = (
                self.ctx_model
                if encoder_type is None or encoder_type == "ctx"
                else self.question_model
            )
            _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
                ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
            )
        
        return q_pooled_out, ctx_pooled_out, pid_tensors, cells_need_link

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
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )
        cells_need_link = None
        if q_pooled_out is not None:
            all_q_pooled_out = []
            for i in range(q_pooled_out.shape[0]):
                cell_reps = torch.sum(_q_seq[i].unsqueeze(0)*question_rep_pos[i].unsqueeze(-1), dim=1)/question_rep_pos[i].sum(dim=1).unsqueeze(-1)
                all_q_pooled_out.append(cell_reps)
        else:
            all_q_pooled_out = None

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )
        return all_q_pooled_out, ctx_pooled_out, cells_need_link

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
        query_token: str = None, epoch: int = -1, full_pos=False
    ) -> BiEncoderBatch:
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        pid_tensors = []
        full_pos_pids = []

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
            neg_ctxs = [c for c in sample['negative_ctxs']] if 'negative_ctxs' in sample else []
            hard_neg_ctxs = [c for c in sample['hard_negative_ctxs']] if 'hard_negative_ctxs' in sample else []
            question = normalize_question(sample['question'])

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
            "question",
            None,
            full_pos_pids if full_pos else None
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
        query_token: str = None, epoch: int = -1, label_question=False
    ) -> BiEncoderBatch:
        question_tensors = []
        question_labels = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        question_rep_pos = []
        pid_tensors = []

        for sample in samples:
            question = sample['question']
            q_tensor = tensorizer.text_to_tensor(question)
            question_tensors.append(q_tensor)
            if label_question:
                q_labels = torch.zeros_like(q_tensor)
                for pos in sample['positive_ctxs']:
                    ground = torch.tensor(tensorizer.tokenizer.encode(pos['grounding'], add_special_tokens=False), dtype=torch.long)
                    start = pos['bert_start']
                    end = pos['bert_end']
                    if not q_tensor[pos['bert_start']:pos['bert_end']+1].equal(ground):
                        start = 0
                        for i in range(len(q_tensor)-len(ground)):
                            if torch.equal(q_tensor[i:i+len(ground)],ground):
                                start = i
                                break
                        end = start + len(ground)-1
                        if start == 0:
                            print (tensorizer.tokenizer.decode(q_tensor))
                            print (tensorizer.tokenizer.decode(ground))
                            print (pos)
                            print ('cannot find cell')
                    q_labels[start: end+1] = 1
                first_row_start = sample['positive_ctxs'][0]['first row start']
                assert q_labels[:first_row_start].sum() == 0
                q_labels[:first_row_start] = -100
                q_labels[q_tensor==tensorizer.tokenizer.pad_token_id] = -100
                question_labels.append(q_labels)
            else:
                positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['first occurance']]
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
                    start = 0
                    for i in range(len(q_tensor)-len(cell_ids)):
                        if torch.equal(q_tensor[i:i+len(cell_ids)],torch.tensor(cell_ids)):
                            start = i
                            break
                    end = start + len(cell_ids)-1
                    if start == 0:
                        print (q_tensor)
                        print (cell_ids)
                        print ('cannot find cell')
                rep_pos[start] = 1
                rep_pos[end] = 1
                question_rep_pos.append(rep_pos)
        if label_question:
            batch_max_len = max(len(q_tensor) for q_tensor in question_tensors)
            questions_tensor = torch.full((len(question_tensors), batch_max_len), tensorizer.tokenizer.pad_token_id, dtype=torch.long)
            for i, q_tensor in enumerate(question_tensors):
                questions_tensor[i, :len(q_tensor)] = q_tensor
            question_labels_pad = torch.full((len(question_labels), batch_max_len), -100, dtype=torch.long)
            for i, q_label in enumerate(question_labels):
                question_labels_pad[i, :len(q_label)] = q_label
            question_labels = question_labels_pad
            ctxs_tensor, ctx_segments, question_segments, question_rep_pos, pid_tensors = None, None, None, None, None
        else:
            questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)
            question_labels = None
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
            "question",
            question_labels, None
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
        hard_neg_fallback: bool = True
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
            question_tensors.append(q_tensor)
            # for pos in sample['positive_ctxs']:
            #     ground = torch.tensor(tensorizer.tokenizer.encode(pos['grounding'], add_special_tokens=False), dtype=torch.long)
            #     start = pos['bert_start']
            #     end = pos['bert_end']
            #     if not q_tensor[pos['bert_start']:pos['bert_end']+1].equal(ground):
            #         start = 0
            #         for i in range(len(q_tensor)-len(ground)):
            #             if torch.equal(q_tensor[i:i+len(ground)],ground):
            #                 start = i
            #                 break
            #         end = start + len(ground)-1
            #         if start == 0:
            #             print (tensorizer.tokenizer.decode(q_tensor))
            #             print (tensorizer.tokenizer.decode(ground))
            #             print (pos)
            #             print ('cannot find cell')

            positive_ctxs = [pos for pos in sample['positive_ctxs'] if pos['first occurance']]
            neg_ctxs = []
            hard_neg_ctxs = []
            for pos in positive_ctxs:
                hard_neg_ctxs.extend(pos['hard_negative_ctxs'])
            if 'hard_negative_pool' in sample:
                hard_neg_ctxs = [sample['hard_negative_pool'][str(ctx)] for ctx in hard_neg_ctxs]

            pos_cells = [pos['grounding'] for pos in positive_ctxs]
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

            this_question = []
            for ci, pos_cell in enumerate(pos_cells):
                cell_ids = tensorizer.tokenizer.encode(pos_cell, add_special_tokens=False)
                start = positive_ctxs[ci]['bert_start']
                end = positive_ctxs[ci]['bert_end']
                rep_pos = torch.zeros(len(q_tensor))
                
                if q_tensor[start:end+1].tolist() != cell_ids:
                    start = 0
                    for i in range(len(q_tensor)-len(cell_ids)):
                        if torch.equal(q_tensor[i:i+len(cell_ids)],torch.tensor(cell_ids)):
                            start = i
                            break
                    end = start + len(cell_ids)-1
                    if start == 0:
                        print (q_tensor)
                        print (cell_ids)
                        print ('cannot find cell')

                rep_pos[start] = 1
                rep_pos[end] = 1
                this_question.append(rep_pos)
            question_rep_pos.append(torch.stack(this_question))
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
            "question",
            None, None
        )

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        if "question_model.encoder.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.encoder.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.encoder.embeddings.position_ids"]
        # for k, v in self.state_dict().items():
        #     if k not in saved_state.model_dict:
        #         print ('missing', k)
        #         saved_state.model_dict[k] = v
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        pid_tensors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None, cells_need_link=None, q_labels=None, label_question=False
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        if q_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            grounding_loss = loss_fct(cells_need_link.view(-1, cells_need_link.shape[-1]), q_labels.view(-1))
        else:
            grounding_loss = None
        if ctx_vectors is None or label_question:
            return None, None, grounding_loss
        scores = self.get_scores(q_vectors, ctx_vectors)        

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        batch_mask = []
        for i in range(len(positive_idx_per_question)):
            curr_pos = pid_tensors[positive_idx_per_question[i]]
            curr_mask = curr_pos == pid_tensors
            curr_mask[positive_idx_per_question[i]] = False
            batch_mask.append(curr_mask)

        batch_mask = torch.stack(batch_mask)
        scores.masked_fill_(batch_mask, -float('inf'))

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)
        return loss, correct_predictions_count, grounding_loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores