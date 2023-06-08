#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
try:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizer
except:
    from transformers import BertConfig, BertTokenizer, BertModel
    from transformers.optimization import AdamW

from dpr.models.biencoder_joint import MoEBiEncoderJoint
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj
from dpr.models.moe_models import MoEBertModel
from dpr.models.optimization import get_layer_lrs, get_layer_lrs_for_t5, AdamWLayer

logger = logging.getLogger(__name__)
model_mapping = {'bert': (BertConfig, BertTokenizer, BertModel), 
                'facebook/contriever': (BertConfig, BertTokenizer, BertModel)}
moe_model_mapping = {
    'bert': (BertConfig, BertTokenizer, (MoEBertModel, BertModel)),
    'facebook/contriever': (BertConfig, BertTokenizer, (MoEBertModel, BertModel)),
}

def get_any_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    use_moe = cfg.encoder.use_moe

    if not hasattr(cfg.encoder, "num_expert") or cfg.encoder.num_expert == -1:
        raise ValueError("When use_moe, num_expert is required")
    num_expert = cfg.encoder.num_expert
    use_infer_expert = cfg.encoder.use_infer_expert
    per_layer_gating = cfg.encoder.per_layer_gating
    moe_type = cfg.encoder.moe_type

    if cfg.encoder.pretrained_file:
        logger.info("loading biencoder weights from %s, which should be a trained DPR model checkpoint", cfg.encoder.pretrained_file)
        logger.info("because of that, we set encoder pretrained_model_cfg to bert to avoid HF version errors")
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
    question_encoder = HFEncoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        use_moe=use_moe,
        moe_type=moe_type,
        use_infer_expert=use_infer_expert,
        per_layer_gating=per_layer_gating,
        num_expert=num_expert,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    logger.info("sharing question and context encoder")
    ctx_encoder = question_encoder

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False
    fix_q_encoder = cfg.fix_q_encoder if hasattr(cfg, "fix_q_encoder") else False

    logger.info("Using MOE model")
    biencoder = MoEBiEncoderJoint(
        question_encoder, ctx_encoder,
        fix_q_encoder=fix_q_encoder, 
        fix_ctx_encoder=fix_ctx_encoder,
        use_cls=cfg.use_cls if 'use_cls' in cfg else False,
        num_expert=num_expert,
        do_rerank=cfg.do_rerank if 'do_rerank' in cfg else False, 
        do_span=cfg.do_span if 'do_span' in cfg else False,
        mean_pool=cfg.mean_pool if 'mean_pool' in cfg else False,
    )

    logger.info('number of parameters %s', sum(p.numel() for p in biencoder.parameters() if p.requires_grad))

    if cfg.encoder.pretrained_file:
        logger.info("loading biencoder weights from %s, this should be a trained DPR model checkpoint", cfg.encoder.pretrained_file)
        checkpoint = load_states_from_checkpoint(cfg.encoder.pretrained_file)
        model_to_load = get_model_obj(biencoder)
        model_to_load.load_state(checkpoint)

    if "base" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "large" in cfg.encoder.pretrained_model_cfg:
        n_layers = 24
    elif "contriever" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    else:
        raise ValueError("Unknown nlayers for %s" % cfg.encoder.pretrained_model_cfg)
    
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
            use_layer_lr=cfg.train.use_layer_lr,
            n_layers=n_layers,
            layer_decay=cfg.train.layer_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_any_tensorizer(cfg)
    return tensorizer, biencoder, optimizer

def get_any_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_any_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )
        if cfg.special_tokens:
            _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)   # this should be fine


def get_any_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    if '-' in pretrained_cfg_name:
        model_name = pretrained_cfg_name.split('-')[0]
    else:
        model_name = pretrained_cfg_name
    tokenizer_class = model_mapping[model_name][1]
    return tokenizer_class.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case, strip_accents=True
    )

import regex
def init_moe_from_pretrained_mapping(pretrained_sd, moe_sd,
                                     moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []
    pattern_list = [
       (f"{moe_layer_name}.", ""),
       (r"interm_layers.\d+", "intermediate"),
       (r"output_layers.\d+", "output"),
       (r"moe_query.\d+", "query"),
       (r"moe_key.\d+", "key"),
       (r"moe_value.\d+", "value"),
       (r"moe_dense.\d+", "dense"),
    ]

    def normalize_var_name(var_name):
        for ptn in pattern_list:
            var_name = regex.sub(ptn[0], ptn[1], var_name)
        return var_name

    for var_name in moe_sd:
        if moe_layer_name in var_name or "moe" in var_name:
            pretrained_var_name = normalize_var_name(var_name)
            logger.info(f"Loads {var_name} from {pretrained_var_name}")
        else:
            pretrained_var_name = var_name

        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        else:
            missing_vars.append((var_name, pretrained_var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            logger.info("Random init %s", var_name)
            state_dict[var_name] = moe_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    if again_missing_vars:
        print("Missing again variables:", again_missing_vars)
    return state_dict

class HFEncoder(nn.Module):
    def __init__(self, cfg_name: str, use_moe: bool = False, moe_type: str = "mod2", num_expert: int = 0, use_infer_expert: bool = False, per_layer_gating: bool = False, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs):
        super().__init__()
        if '-' in cfg_name:
            model_name = cfg_name.split('-')[0]
        else:
            model_name = cfg_name
        if use_moe:
            logger.info("Using MoE models for HFEncoder")
            logger.info("Number of expert: %d", num_expert)
            config_class, _, model_class = moe_model_mapping[model_name]
            assert num_expert > 0, "num_expert can't be zero when using MoE."
        else:
            config_class, _, model_class = model_mapping[model_name]

        cfg = config_class.from_pretrained(cfg_name)
        self.num_expert = cfg.num_expert = num_expert
        self.use_infer_expert = cfg.use_infer_expert = use_infer_expert
        self.per_layer_gating = cfg.per_layer_gating = per_layer_gating
        self.moe_type = cfg.moe_type = moe_type
        self.use_moe = use_moe
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        assert cfg.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(cfg.hidden_size, projection_dim) if projection_dim != 0 else None
        )
        if use_moe:
            model_class, orig_model = model_class
            orig_encoder = orig_model.from_pretrained(cfg_name, config=cfg, **kwargs)
            self.encoder = model_class(config=cfg)
            self.encoder.load_state_dict(init_moe_from_pretrained_mapping(orig_encoder.state_dict(), self.encoder.state_dict()))
        else:
            self.encoder =  model_class.from_pretrained(cfg_name, config=cfg, **kwargs)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0, expert_id=None, expert_offset: int = 0,
    ) -> Tuple[T, ...]:
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        if expert_id is not None:
            inputs["expert_id"] = expert_id
        outputs = self.encoder(**inputs)
        sequence_output = outputs[0]
        if self.encoder.config.output_hidden_states:
            hidden_states = outputs[2]
        else:
            hidden_states = None

        if self.use_moe and self.use_infer_expert:
            total_entropy_loss = outputs[-1]
        else:
            total_entropy_loss = None

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        if total_entropy_loss is not None:
            return sequence_output, pooled_output, hidden_states, total_entropy_loss
        else:
            return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.encoder.config.hidden_size

def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 50
    unused_ids = [
        tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)
    ]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        del tokenizer.vocab["[unused{}]".format(idx)]
        tokenizer.vocab[special_tokens[idx]] = id
        tokenizer.ids_to_tokens[id] = special_tokens[idx]

    tokenizer._additional_special_tokens = list(special_tokens)
    logger.info(
        "Added special tokenizer.additional_special_tokens %s",
        tokenizer.additional_special_tokens,
    )
    logger.info("Tokenizer's all_special_tokens %s", tokenizer.all_special_tokens)

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    use_layer_lr: bool = True, 
    n_layers: int = 12,
    layer_decay: float = 0.8,
    moe_factor: float = 1.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    if use_layer_lr:
        logger.info("Using Adam w layerwise adaptive learning rate")
        name_to_adapt_lr = get_layer_lrs(
            layer_decay=layer_decay,
            n_layers=n_layers,
        )
        optimizer_grouped_parameters = []
        logger.info(name_to_adapt_lr)
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

            wdecay = weight_decay
            if any(nd in name for nd in no_decay):
                # Parameters with no decay.
                wdecay = 0.0
            if "moe" in name:
                logger.info(f"Applying moe_factor {moe_factor} for LR with {name}")
                lr_adapt_weight *= moe_factor
            optimizer_grouped_parameters.append({
                "params": param,
                "weight_decay": wdecay,
                "lr_adapt_weight": lr_adapt_weight,
            })
        optimizer = AdamWLayer(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps, betas=adam_betas)
    else:
        logger.info("Using Adam")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]