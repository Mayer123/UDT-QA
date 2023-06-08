#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import logging
import math
import os
import random
import sys
import time
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F

from dpr.models import init_biencoder_components
from dpr.models.biencoder_joint import MoEBiEncoderJoint, BiEncoderBatch
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.utils.conf_utils import BiencoderDatasetsCfg
from dpr.utils.data_utils import (
    ShardedDataIterator,
    Tensorizer,
    MultiSetDataIterator
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import transformers
transformers.logging.set_verbosity_error()
logger = logging.getLogger()
setup_logger(logger)


class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, cfg: DictConfig):
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, model, optimizer = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg
        )

        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)

        if saved_state:
            self._load_saved_state(saved_state)

        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):

        hydra_datasets = (
            self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        )
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names
            if is_train_set
            else self.ds_cfg.dev_datasets_names,
        )

        # randomized data loading to avoid file system congestion
        datasets_list = [ds for ds in hydra_datasets]
        rnd = random.Random(rank)
        rnd.shuffle(datasets_list)
        [ds.load_data() for ds in datasets_list]

        sharded_iterators = [
            ShardedDataIterator(
                ds.data,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,  strict_batch_size=True
            )
            for ds in hydra_datasets
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [],
            rank=rank,
        )

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )

        max_iterations = train_iterator.max_iterations
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(
                self.optimizer, warmup_steps, total_updates
            )

        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        if not cfg.dev_datasets:
            validation_loss = 0
        else:
            if epoch >= cfg.val_av_rank_start_epoch:
                validation_loss = self.validate_average_rank()
            else:
                validation_loss = self.validate_nll()

        if save_cp:
            #if epoch >= cfg.val_av_rank_start_epoch:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_nll(self) -> float:
        logger.info("NLL validation ...")
        cfg = self.cfg
        self.biencoder.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        total_correct_linking = 0
        total_correct_reranking = 0
        total_correct_span = 0
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        log_result_step = cfg.train.log_batch_step
        batches = 0
        dataset = 0
        linker_batch = 0
        retriever_batch = 0
        reranker_batch = 0
        span_batch = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            #logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)
            linker_b = 0
            retriever_b = 0
            reranker_b = 0
            biencoder_input_span = None
            if 'hard_negative_pool' in samples_batch[0]:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input_use_cell(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False, mean_pool=cfg.mean_pool
                )
                linker_b = True 
                if cfg.do_span:
                    biencoder_input_span = MoEBiEncoderJoint.create_biencoder_input_span_proposal(
                        samples_batch,
                        self.tensorizer,
                        True,
                        num_hard_negatives,
                        num_other_negatives,
                        shuffle=False
                    )
                    span_batch += 1
            elif 'rerank' in samples_batch[0]:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input_rerank(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False
                )
                reranker_b = True
            else:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input2(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False
                )
                retriever_b = True

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )
            encoder_type = ds_cfg.encoder_type
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_input,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
            )
            total_loss += loss.item()

            if linker_b:
                total_correct_linking += correct_cnt
                linker_batch += 1
            elif reranker_b:
                total_correct_reranking += correct_cnt
                reranker_batch += 1
            else:
                total_correct_predictions += correct_cnt
                retriever_batch += 1
            batches += 1
            if biencoder_input_span:
                loss, correct_cnt = _do_biencoder_fwd_pass(
                    self.biencoder,
                    biencoder_input_span,
                    self.tensorizer,
                    cfg,
                    encoder_type=encoder_type,
                    rep_positions=rep_positions,
                )
                total_loss += loss.item()
                total_correct_span += correct_cnt
                batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f",
                    i,
                    time.time() - start_time,
                    loss.item()
                )
        print ('linker_batch=%d, retriever_batch=%d reranker_batch=%d span_batch=%d' % (linker_batch, retriever_batch, reranker_batch, span_batch))

        total_loss = total_loss / max(1, batches)     # for WebQ, the dev set is too small
        total_samples_retriever = retriever_batch * cfg.train.dev_batch_size * self.distributed_factor
        total_samples_linker = linker_batch * cfg.train.dev_batch_size * self.distributed_factor
        total_samples_reranker = reranker_batch * cfg.train.dev_batch_size     # For reranker and span model, there is no communication across GPUs
        total_samples_span = linker_batch * cfg.train.dev_batch_size
        correct_ratio_retriever = float(total_correct_predictions / max(1, total_samples_retriever))
        correct_ratio_linker = float(total_correct_linking / max(1, total_samples_linker))
        correct_ratio_reranker = float(total_correct_reranking / max(1, total_samples_reranker))
        correct_ratio_span = float(total_correct_span / max(1, total_samples_span))
        logger.info(
            "NLL Validation: loss = %f. retriever correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples_retriever,
            correct_ratio_retriever,
        )
        logger.info("linker correct prediction ratio  %d/%d ~  %f", total_correct_linking, total_samples_linker, correct_ratio_linker)
        logger.info("reranker correct prediction ratio  %d/%d ~  %f", total_correct_reranking, total_samples_reranker, correct_ratio_reranker)
        logger.info("span correct prediction ratio  %d/%d ~  %f", total_correct_span, total_samples_span, correct_ratio_span)
        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")

        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        q_representations_ents = []
        ctx_represenations = []
        positive_idx_per_question = []
        positive_idx_per_question_ents = []
        full_pid_tensors = []
        ctx_representations_rerank = []
        positive_idx_per_question_rerank = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg
        total_correct_span = 0
        linker_batch = 0
        log_result_step = cfg.train.log_batch_step
        dataset = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            # samples += 1
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            biencoder_input_span = None
            if 'hard_negative_pool' in samples_batch[0]:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input_all_positives_use_cell(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False, mean_pool=cfg.mean_pool
                )
                linker_batch += 1
                if cfg.do_span:
                    biencoder_input_span = MoEBiEncoderJoint.create_biencoder_input_span_proposal(
                        samples_batch,
                        self.tensorizer,
                        True,
                        num_hard_negatives,
                        num_other_negatives,
                        shuffle=False
                    )
            elif 'rerank' in samples_batch[0]:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input_rerank(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False
                )
            else:
                biencoder_input = MoEBiEncoderJoint.create_biencoder_input2(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=False
                )

            biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), cfg.device)) # This is only used to handle single-gpu case
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            ctxs_sep_positions = biencoder_input.ctx_sep_positions
            bsz = ctxs_ids.size(0)
            if 'rerank' not in samples_batch[0]:
                full_pid_tensors.append(biencoder_input.pid_tensors)

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments, q_rep_pos, q_tgt_expert = (
                    (biencoder_input.question_ids, biencoder_input.question_segments, biencoder_input.question_rep_pos, biencoder_input.q_target_expert)
                    if j == 0
                    else (None, None, None, None, None)
                )
                #assert not q_rep_pos
                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start : batch_start + sub_batch_size]
                if ctxs_sep_positions is not None:
                    ctx_sep_batch = ctxs_sep_positions[batch_start : batch_start + sub_batch_size]
                else:
                    ctx_sep_batch = None
                c_target_expert = biencoder_input.ctx_target_expert

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    #with autocast():
                    if 'hard_negative_pool' in samples_batch[0]:
                        if isinstance(self.biencoder, torch.nn.DataParallel) or isinstance(self.biencoder, torch.nn.parallel.DistributedDataParallel):
                            fwd_call = self.biencoder.module.forward_multiple_cells
                        else:
                            fwd_call = self.biencoder.forward_multiple_cells
                        q_dense, ctx_dense, _ = fwd_call(
                            q_ids,
                            q_segments,
                            q_attn_mask,
                            q_rep_pos,
                            ctx_ids_batch,
                            ctx_seg_batch,
                            ctx_attn_mask,
                            encoder_type=encoder_type,
                            representation_token_pos=rep_positions, q_target_expert=q_tgt_expert, c_target_expert=c_target_expert,
                        )
                    else:
                        model_out = self.biencoder(
                            q_ids,
                            q_segments,
                            q_attn_mask,
                            None,
                            ctx_ids_batch,
                            ctx_seg_batch,
                            ctx_attn_mask,
                            encoder_type=encoder_type,
                            representation_token_pos=rep_positions, 
                            q_target_expert=q_tgt_expert, c_target_expert=c_target_expert, 
                            ctx_sep_positions=ctx_sep_batch,
                        )
                        q_dense = model_out[0]
                        ctx_dense = model_out[1]

                if q_dense is not None:
                    if 'hard_negative_pool' in samples_batch[0]:
                        q_representations_ents.extend([qd.cpu() for qd in q_dense])
                    else:
                        q_represenations.extend(q_dense.cpu().split(1, dim=0))
                if 'rerank' in samples_batch[0]:
                    ctx_representations_rerank.extend(ctx_dense.cpu().split(1, dim=0))
                else:
                    ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            if 'hard_negative_pool' in samples_batch[0]:
                positive_idx_per_question_ents.extend(
                    [[total_ctxs + vv for vv in v] for v in batch_positive_idxs]
                )
            elif 'rerank' in samples_batch[0]:
                positive_idx_per_question_rerank.extend(batch_positive_idxs)
            else:
                positive_idx_per_question.extend(
                    [total_ctxs + v for v in batch_positive_idxs]
                )
            if biencoder_input_span:
                _, correct_cnt = _do_biencoder_fwd_pass(
                    self.biencoder,
                    biencoder_input_span,
                    self.tensorizer,
                    cfg,
                    encoder_type=encoder_type,
                    rep_positions=rep_positions,
                )
                total_correct_span += correct_cnt
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, ctx_rerank_vectors %d, q_vectors %d, q_vectors_ents %d",
                    i,
                    len(ctx_represenations), len(ctx_representations_rerank),
                    len(q_represenations), len(q_representations_ents)
                )
        if len(ctx_represenations) > 0:
            ctx_represenations = torch.cat(ctx_represenations, dim=0)
            logger.info(
                "Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
            )
        else:
            ctx_represenations = None
        if len(ctx_representations_rerank) > 0:
            ctx_representations_rerank = torch.cat(ctx_representations_rerank, dim=0)
            logger.info(
                "Av.rank validation: total ctx_rerank_vectors size=%s", ctx_representations_rerank.size()
            )
        else:
            ctx_representations_rerank = None
        if len(q_represenations) > 0:
            q_represenations = torch.cat(q_represenations, dim=0)
            logger.info(
            "Av.rank validation: total q_vectors size=%s", q_represenations.size()
            )
            q_num = q_represenations.size(0)
            assert q_num == len(positive_idx_per_question)
        else:
            q_represenations = None
            q_num = 0
            
        if len(q_representations_ents) > 0:
            total_q_rep_size = sum([len(q_rep) for q_rep in q_representations_ents])
            logger.info(
                "Av.rank validation: total q_vectors ents size=%s", total_q_rep_size
            )
            assert len(q_representations_ents) == len(positive_idx_per_question_ents)
        else:
            q_representations_ents = None
            total_q_rep_size = 0

        if len(full_pid_tensors) > 0 and all([fpt is not None for fpt in full_pid_tensors]):
            full_pid_tensors = torch.cat(full_pid_tensors, dim=0)
        rank = 0
        if q_represenations is not None:
            scores = sim_score_f(q_represenations, ctx_represenations)
            values, indices = torch.sort(scores, dim=1, descending=True)
            for i, idx in enumerate(positive_idx_per_question):
                # aggregate the rank of the known gold passage in the sorted results for each question
                #gold_idx = (indices[i] == idx).nonzero()
                #rank += gold_idx.item()
                curr_retrieved_pids = full_pid_tensors[indices[i]]
                gold_pids = full_pid_tensors[idx]
                gold_idx = (curr_retrieved_pids == gold_pids).nonzero()[0].item()
                rank += gold_idx


        total_hits_top1 = 0
        total_positives = 0
        link_rank = 0
        if q_representations_ents is not None:
            for i, pos_idx in enumerate(positive_idx_per_question_ents):
                scores = sim_score_f(q_representations_ents[i], ctx_represenations)
                values, indices = torch.sort(scores, dim=1, descending=True)
                total_positives += len(pos_idx)
                retrieved_pids = [full_pid_tensors[v] for v in indices[:, 0]]
                gold_pids = [full_pid_tensors[v] for v in pos_idx]
                hits_top1 = [idx for idx in gold_pids if idx in retrieved_pids]
                total_hits_top1 += len(hits_top1)
                for j in range(len(pos_idx)):
                    retrieved_pids = full_pid_tensors[indices[j]]
                    gold_pids = full_pid_tensors[pos_idx[j]]
                    gold_idx = (retrieved_pids == gold_pids).nonzero()[0].item()
                    link_rank += gold_idx

        rerank_rank = 0 
        rerank_num = len(positive_idx_per_question_rerank)
        if ctx_representations_rerank is not None:
            scores = ctx_representations_rerank.view(len(positive_idx_per_question_rerank), -1)
            values, indices = torch.sort(scores, dim=1, descending=True)
            for i, idx in enumerate(positive_idx_per_question_rerank):
                gold_idx = (indices[i] == idx).nonzero().item()
                rerank_rank += gold_idx

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num, total_hits_top1, total_positives, link_rank, rerank_rank, rerank_num, linker_batch, total_correct_span], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num, remote_hits_top1, remote_positives, remote_link_rank, remote_rerank_rank, remote_rerank_num, remote_linker_batch, remote_correct_span = item
                if i != cfg.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num
                    total_hits_top1 += remote_hits_top1
                    total_positives += remote_positives
                    link_rank += remote_link_rank
                    rerank_rank += remote_rerank_rank
                    rerank_num += remote_rerank_num
                    linker_batch += remote_linker_batch
                    total_correct_span += remote_correct_span

        av_rank = float(rank / max(1, q_num))
        rerank_av_rank = float(rerank_rank / max(1, rerank_num))
        link_recall = float(total_hits_top1 / max(1, total_positives))
        link_av_rank = float(link_rank / max(1, total_positives))

        logger.info("Av.rank validation: average rank %s, total questions=%d", av_rank, q_num)
        logger.info("Av.rank validation: recall for all positives %s total entities=%d average rank %s", link_recall, total_positives, link_av_rank)
        logger.info("Av.rank validation: average rerank rank %s, total reranked questions=%d", rerank_av_rank, rerank_num)
        
        total_samples_span = linker_batch * cfg.train.dev_batch_size
        correct_ratio_span = float(total_correct_span / max(1, total_samples_span))
        logger.info("span correct prediction ratio  %d/%d ~  %f", total_correct_span, total_samples_span, correct_ratio_span)
        return av_rank+link_av_rank+rerank_av_rank-correct_ratio_span

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        #epoch_correct_predictions = 0

        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        seed = cfg.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        dataset = 0
        linker_batch = 0
        span_batch = 0
        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            ds_cfg = self.ds_cfg.train_datasets[dataset]
            special_token = ds_cfg.special_token
            encoder_type = ds_cfg.encoder_type
            shuffle_positives = ds_cfg.shuffle_positives

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            if 'hard_negative_pool' in samples_batch[0]:
                if cfg.do_span and random.random() < cfg.span_ratio:
                    biencoder_batch = MoEBiEncoderJoint.create_biencoder_input_span_proposal(
                        samples_batch,
                        self.tensorizer,
                        True,
                        num_hard_negatives,
                        num_other_negatives,
                        shuffle=True,
                        shuffle_positives=shuffle_positives,
                        query_token=special_token, epoch=epoch
                    )
                    span_batch += 1
                else:
                    biencoder_batch = MoEBiEncoderJoint.create_biencoder_input_use_cell(
                        samples_batch,
                        self.tensorizer,
                        True,
                        num_hard_negatives,
                        num_other_negatives,
                        shuffle=True,
                        shuffle_positives=shuffle_positives,
                        query_token=special_token, epoch=epoch, mean_pool=cfg.mean_pool
                    )
                    linker_batch += 1
            elif 'rerank' in samples_batch[0]:
                biencoder_batch = MoEBiEncoderJoint.create_biencoder_input_rerank(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=True, epoch=epoch
                )
            else:
                biencoder_batch = MoEBiEncoderJoint.create_biencoder_input2(
                    samples_batch,
                    self.tensorizer,
                    True,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=True,
                    shuffle_positives=shuffle_positives,
                    query_token=special_token, epoch=epoch
                )
            if epoch == 0 and i < 50:
                logger.info(f"q side expert {biencoder_batch.q_target_expert} ctx side expert {biencoder_batch.ctx_target_expert}")
                if biencoder_batch.question_ids is not None:
                    logger.info(self.tensorizer.tokenizer.convert_ids_to_tokens(biencoder_batch.question_ids[0]))

            # get the token to be used for representation selection
            from dpr.data.biencoder_data import DEFAULT_SELECTOR

            selector = ds_cfg.selector if ds_cfg else DEFAULT_SELECTOR

            rep_positions = selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer
            )

            loss_scale = (
                cfg.loss_scale_factors[dataset] if cfg.loss_scale_factors else None
            )
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
                loss_scale=loss_scale,
            )
            #epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            loss.backward()
            if cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.biencoder.parameters(), cfg.train.max_grad_norm
                )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0 and cfg.train.eval_per_epoch != 1:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.biencoder.train()

        logger.info("Epoch finished on %d", cfg.local_rank)
        logger.info("linker_batch: %d, span_batch: %d", linker_batch, span_batch)
        self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        #logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))
        os.system("cp %s %s" % (os.path.join(hydra.utils.get_original_cwd(), 'train_dense_encoder_COS.py'), os.path.join(cfg.output_dir, 'train_dense_encoder_COS.py')))
        os.system("cp %s %s" % (os.path.join(hydra.utils.get_original_cwd(), 'dpr/models/biencoder_joint.py'), os.path.join(cfg.output_dir, 'biencoder_joint.py')))
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if self.cfg.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state)

        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(
    cfg,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_pid_tensors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None, local_full_positive_pids: list=None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = cfg.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )
        pid_tensors_to_send = (
            torch.empty_like(local_pid_tensors).cpu().copy_(local_pid_tensors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                pid_tensors_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
                local_full_positive_pids,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []
        global_pid_tensors = []
        if local_full_positive_pids is not None:
            global_full_positive_idxs = []
        else:
            global_full_positive_idxs = None
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, pid_tensors, positive_idx, hard_negatives_idxs, full_positive_pids = item

            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                global_pid_tensors.append(pid_tensors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
                if full_positive_pids is not None:
                    global_full_positive_idxs.extend(full_positive_pids)
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                #if local_cell_reps is not None:
                global_pid_tensors.append(local_pid_tensors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
                if local_full_positive_pids is not None:
                    global_full_positive_idxs.extend(local_full_positive_pids)
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
        global_pid_tensors = torch.cat(global_pid_tensors, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        global_pid_tensors = local_pid_tensors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs
        global_full_positive_idxs = local_full_positive_pids

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        global_pid_tensors,
        positive_idx_per_question,
        hard_negatives_per_question, 
        full_positive_idxs=global_full_positive_idxs,
        loss_scale=loss_scale, loss_mask=True
    )

    return loss, is_correct


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    cfg,
    encoder_type: str,
    rep_positions=0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]:

    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device)) # This is only used to handle single-gpu case

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.question_rep_pos,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos=rep_positions,
            q_target_expert=input.q_target_expert,
            c_target_expert=input.ctx_target_expert,
            ctx_sep_positions=input.ctx_sep_positions,
            span_start_positions=input.span_start_positions,
            span_end_positions=input.span_end_positions,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.question_rep_pos,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type=encoder_type,
                representation_token_pos=rep_positions,
                q_target_expert=input.q_target_expert,
                c_target_expert=input.ctx_target_expert,
                ctx_sep_positions=input.ctx_sep_positions,
                span_start_positions=input.span_start_positions,
                span_end_positions=input.span_end_positions,
            )

    local_q_vector = model_out[0]
    local_ctx_vectors = model_out[1]
    cell_needs_link = model_out[2]

    if local_q_vector is not None and local_ctx_vectors is not None:
        loss_function = BiEncoderNllLoss()

        loss, is_correct = _calc_loss(
            cfg,
            loss_function,
            local_q_vector,
            local_ctx_vectors,
            input.pid_tensors,
            input.is_positive,
            input.hard_negatives,
            input.full_positive_pids,
            loss_scale=loss_scale,
        )
    elif cell_needs_link is not None:
        loss_function = BiEncoderSpanLoss()
        loss, is_correct = loss_function.calc(
                cell_needs_link, local_q_vector
            )
    else:
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            loss, is_correct = model.module.compute_reranker_loss(local_ctx_vectors, input.is_positive)
        else:
            loss, is_correct = model.compute_reranker_loss(local_ctx_vectors, input.is_positive)

    is_correct = is_correct.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.train.gradient_accumulation_steps
    return loss, is_correct

class BiEncoderSpanLoss(object):
    def calc(
        self,
        span_scores: T, cls_scores: T,
        loss_scale: float = None
    ) -> Tuple[T, int]:
        pos_loss = F.cross_entropy(
            torch.cat([span_scores, cls_scores], dim=1),
            torch.zeros(span_scores.shape[0], dtype=torch.long).to(span_scores.device),
            reduction="mean",
        )
        threshold_scores = torch.cat([cls_scores, span_scores[:, 1:]], dim=1)
        cls_loss = F.cross_entropy(
            threshold_scores,
            torch.zeros(threshold_scores.shape[0], dtype=torch.long).to(threshold_scores.device),
            reduction="mean",
        )
        loss = (pos_loss + cls_loss) / 2
        max_score, max_idxs = torch.max(span_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.zeros(span_scores.shape[0]).to(span_scores.device)
        ).sum()

        _, max_idxs = torch.max(threshold_scores, 1)
        correct_threshold_count = (
            max_idxs == torch.zeros(threshold_scores.shape[0]).to(threshold_scores.device)
        ).sum()
        correct_predictions_count = (correct_predictions_count + correct_threshold_count)/2
        if loss_scale:
            loss.mul_(loss_scale)
        return loss, correct_predictions_count

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        pid_tensors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None, full_positive_idxs=None,
        loss_scale: float = None, loss_mask=True
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """

        scores = self.get_scores(q_vectors, ctx_vectors)  
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        batch_mask = []
        if full_positive_idxs is not None:
            for i in range(len(full_positive_idxs)):
                all_pos = full_positive_idxs[i]
                full_mask = pid_tensors == -1
                for p in all_pos:
                    curr_mask = p == pid_tensors
                    full_mask = torch.logical_or(full_mask, curr_mask)
                full_mask[positive_idx_per_question[i]] = False
                batch_mask.append(full_mask)
        else:
            for i in range(len(positive_idx_per_question)):
                curr_pos = pid_tensors[positive_idx_per_question[i]]
                curr_mask = curr_pos == pid_tensors
                curr_mask[positive_idx_per_question[i]] = False
                batch_mask.append(curr_mask)

        batch_mask = torch.stack(batch_mask)
        if loss_mask:
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
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
def main(cfg: DictConfig):
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = BiEncoderTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info(
            "No train files are specified. Run 2 types of validation for specified model file"
        )
        #trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()
