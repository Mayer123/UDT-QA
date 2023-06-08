# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

from transformers import AutoModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F

def listmle(thres, scores, labels):
    full_scores = torch.cat([thres, scores], dim=1)
    full_labels = torch.cat([torch.full(thres.size(), 0.5, device=thres.device), labels], dim=1)
    if full_scores.shape[1] != full_labels.shape[1]:
        print ('full scores', full_scores.shape)
        print ('full labels', full_labels.shape)
    sorted_labels, indices = full_labels.sort(dim=1, descending=True)
    sorted_scores = full_scores.gather(dim=1, index=indices)
    sorted_scores = sorted_scores.exp() * sorted_labels.ne(-1).float()
    loss = 0.0
    for i in range(sorted_scores.shape[0]):
        for j in range(sorted_scores.shape[1]):
            rest = sorted_scores[i][sorted_labels[i] < sorted_labels[i][j]]
            prob = sorted_scores[i][j] / (sorted_scores[i][j] + rest.sum())
            loss += torch.log(prob)
            if sorted_labels[i][j] == 0.5:
                break
    return -loss

class QAModelV2(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.model_name = args.model_name
        self.sp_weight = args.sp_weight
        self.rank_weight = args.para_weight
        self.ans_weight = args.answer_weight
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.encoder._set_gradient_checkpointing(self.encoder.encoder, value=args.gradient_checkpointing)
        print ("gradient_checkpointing", args.gradient_checkpointing)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.rank = nn.Linear(config.hidden_size, 1) # noan

        self.sp = nn.Linear(config.hidden_size, 1)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.listmle = args.listmle
        self.sentlistmle = args.sentlistmle
        self.max_ans_len = args.max_ans_len
        self.hard_em = args.hard_em

    def forward(self, batch):

        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])

        if "electra" in self.model_name:
            sequence_output = outputs[0]
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.qa_outputs(sequence_output)
        outs = [o.squeeze(-1) for o in logits.split(1, dim=-1)]
        outs = [o.float().masked_fill(batch['token_type_ids'].ne(1), float("-inf")).type_as(o) for o in outs]

        start_logits, end_logits = outs[0], outs[1]
        rank_thres = self.rank(sequence_output[:, 0, :])
        sent_thres = self.sp(sequence_output[:, 0, :])

        sent_rep = []
        for i in range(sequence_output.size(0)):
            this_starts = batch['sent_starts'][i]
            sent_rep.append(torch.stack([sequence_output[i][this_starts[j]:this_starts[j+1]].mean(dim=0) if this_starts[j+1] != -1 else sequence_output[i][-1] for j in range(len(this_starts[:-1]))], dim=0))
        sent_rep = torch.stack(sent_rep, dim=0)
        sp_score = self.sp(sent_rep).squeeze(2)

        gather_index = batch["para_starts"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        para_marker_rep = torch.gather(sequence_output, 1, gather_index)
        para_score = self.rank(para_marker_rep).squeeze(2)

        if self.training:
            rank_loss = listmle(rank_thres, para_score, batch['para_labels'])

            if self.sentlistmle:
                sp_loss = listmle(sent_thres, sp_score, batch['sp_sent_labels'])
            else:
                sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["sp_sent_labels"].float(), reduction="none")
                sp_loss = sp_loss*batch['sent_starts'][:, :-1]*batch['sp_sent_labels'].ne(-1).float()
                sp_loss = sp_loss.sum()

            start_positions, end_positions = batch["starts"], batch["ends"]
            start_losses = [self.loss_fct(start_logits, starts) for starts in torch.unbind(start_positions, dim=1)]
            end_losses = [self.loss_fct(end_logits, ends) for ends in torch.unbind(end_positions, dim=1)]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
            log_prob = - loss_tensor
            log_prob = log_prob.float().masked_fill(log_prob == 0, float('-inf')).type_as(log_prob)
            marginal_probs = torch.sum(torch.exp(log_prob), dim=1)
            marginal_probs = [marginal_probs[idx] for idx in marginal_probs.nonzero()]
            if len(marginal_probs) == 0:
                loss = rank_loss * self.rank_weight + sp_loss * self.sp_weight
            else:
                m_prob = torch.cat(marginal_probs)
                span_loss = - torch.log(m_prob)
                span_loss = span_loss.sum()
                loss = rank_loss * self.rank_weight + span_loss * self.ans_weight + sp_loss * self.sp_weight
            return loss, rank_loss * self.rank_weight, sp_loss * self.sp_weight

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'rank_score': para_score,
            "sp_score": sp_score,
            'sent_thres': sent_thres,
            'rank_thres': rank_thres
            }