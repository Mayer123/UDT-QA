#!/usr/bin/env python
"""This implements MoE layers for modeling."""


from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import BertLayer

try:
    from transformers.modeling_bert import (
        BertAttention,
        BertIntermediate,
        BertOutput,
        BertEmbeddings,
        BertPooler,
        BertPreTrainedModel,
        BertSelfOutput,
    )
    from transformers.modeling_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except:
    from transformers.models.bert.modeling_bert import (
        BertAttention,
        BertIntermediate,
        BertOutput,
        BertEmbeddings,
        BertPooler,
        BertPreTrainedModel,
        BertSelfOutput,
    )
    from transformers.modeling_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
from transformers.activations import ACT2FN
import logging

logger = logging.getLogger(__name__)

moe_type_to_func = {
    "mod2": lambda layer_idx: (layer_idx + 1) % 2 == 0,
    "mod3": lambda layer_idx: (layer_idx + 1) % 3 == 0,
    "mod4": lambda layer_idx: (layer_idx + 1) % 4 == 0,
    "mod6": lambda layer_idx: (layer_idx + 1) % 6 == 0,
    "mod12": lambda layer_idx: (layer_idx + 1) % 12 == 0,
    "ge6mod3": lambda layer_idx: (layer_idx + 1) >= 6 and (layer_idx + 1) % 3 == 0,
    "le7mod3": lambda layer_idx: layer_idx <= 6 and (layer_idx % 3 == 0),
    "late6": lambda layer_idx: layer_idx >= 6,
    "late3": lambda layer_idx: layer_idx >= 9,
}

# class BertIntermediate(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class BertOutput(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

def slice_for_expert(all_outputs, expert_idx):
    """Selects the expert for forwarding."""
    num_experts, bsz, seq_len, dim = all_outputs.shape
    offset = torch.arange(0, bsz).to(expert_idx.device)
    slice_indices = expert_idx * bsz + offset
    # slice_indices = torch.Tensor([bsz * idx + bs_idx for bs_idx, idx in enumerate(expert_idx)]).type(torch.int64)
    all_outputs = all_outputs.view(bsz * num_experts, seq_len, dim)
    selected_outputs = all_outputs[slice_indices, :, :]
    return selected_outputs


def select_fwd(x, expert_idx, interm_modules, output_modules):
   """Forwards inputs with different experts in the given batch."""

   # Forwards all experts and slices the target expert outputs.
   # Tensor of shape [num_expert, bsz, seq_len, dim].
   # all_interm_outputs = torch.stack([
   #     mm_layer(x.clone()) for mm_layer in interm_modules], dim=0)
   # interm_outputs = slice_for_expert(all_interm_outputs, expert_idx)
   interm_outputs = single_select_fwd(x, expert_idx, interm_modules)

   # Tensor of shape [num_expert, bsz, seq_len, dim].
   # all_final_outputs = torch.stack([
   #     out_layer(interm_outputs.clone(), x.clone())
   #     for out_layer in output_modules], dim=0)
   # final_outputs = slice_for_expert(all_final_outputs, expert_idx)
   final_outputs = single_select_fwd(interm_outputs, expert_idx, output_modules, y=x)

   return final_outputs


def single_select_fwd(x, expert_idx, output_modules, y=None):
    """Forwards inputs with different experts in the given batch."""
    if y is None:
        all_outputs = torch.stack([
            mm_layer(x.clone()) for mm_layer in output_modules], dim=0)
    else:
        all_outputs = torch.stack([
            mm_layer(x.clone(), y.clone()) for mm_layer in output_modules], dim=0)
    final_outputs = slice_for_expert(all_outputs, expert_idx)
    return final_outputs


class MoELayer(nn.Module):
    def __init__(self, config, n_expert=1):
        super().__init__()
        self.interm_layers = nn.ModuleList(
            [BertIntermediate(config) for _ in range(n_expert)]
        )
        self.output_layers = nn.ModuleList(
            [BertOutput(config) for _ in range(n_expert)]
        )

    def forward(self, input_tensor, expert_idx):
        if isinstance(expert_idx, int):
            intermediate_outputs = self.interm_layers[expert_idx](input_tensor)
            hidden_states = self.output_layers[expert_idx](intermediate_outputs, input_tensor)
        elif isinstance(expert_idx, torch.Tensor):
            assert expert_idx.dtype == torch.int64, "only int64 is supported for slicing experts."
            hidden_states = select_fwd(input_tensor, expert_idx, self.interm_layers, self.output_layers)
        else:
            raise ValueError("Unknown expert_idx type", type(expert_idx))
        return hidden_states


def select_expert(bsz, num_expert, expert_id=None):
    if expert_id is None:
        return torch.randint(low=0, high=num_expert, size=(bsz,)).tolist()
    return expert_id


def _expert_forward(fwd_func_list, input_tensor, expert_idx):
    if isinstance(expert_idx, int):
        outputs = fwd_func_list[expert_idx](input_tensor)
    elif isinstance(expert_idx, torch.Tensor):
        assert expert_idx.dtype == torch.int64, "only int64 is supported for slicing experts."
        outputs = single_select_fwd(input_tensor, expert_idx, fwd_func_list)
    else:
        raise ValueError("Unknown expert_idx type", type(expert_idx))
    return outputs


class MoEBertSelfAttention(nn.Module):
    def __init__(self, config, n_expert=1):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.moe_query= nn.ModuleList(
            [nn.Linear(config.hidden_size, self.all_head_size) for _ in range(n_expert)])
        self.moe_key= nn.ModuleList(
            [nn.Linear(config.hidden_size, self.all_head_size) for _ in range(n_expert)])
        self.moe_value= nn.ModuleList(
            [nn.Linear(config.hidden_size, self.all_head_size) for _ in range(n_expert)])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        expert_idx,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = _expert_forward(self.moe_query, hidden_states, expert_idx)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                _expert_forward(self.moe_key, encoder_hidden_states, expert_idx))
            value_layer = self.transpose_for_scores(
                _expert_forward(self.moe_value, encoder_hidden_states, expert_idx))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                _expert_forward(self.moe_key, hidden_states, expert_idx))
            value_layer = self.transpose_for_scores(
                _expert_forward(self.moe_value, hidden_states, expert_idx))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(
                _expert_forward(self.moe_key, hidden_states, expert_idx))
            value_layer = self.transpose_for_scores(
                _expert_forward(self.moe_value, hidden_states, expert_idx))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MoEBertSelfOutput(nn.Module):
    def __init__(self, config, n_expert=1):
        super().__init__()
        self.moe_dense = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(n_expert)])

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, expert_idx):
        if isinstance(expert_idx, int):
            hidden_states = self.moe_dense[expert_idx](hidden_states)
        elif isinstance(expert_idx, torch.Tensor):
            assert expert_idx.dtype == torch.int64, "only int64 is supported for slicing experts."
            hidden_states = single_select_fwd(hidden_states, expert_idx, self.moe_dense)
        else:
            raise ValueError("Unknown expert_idx type", type(expert_idx))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MoEBertAttention(nn.Module):
    def __init__(self, config, n_expert=1):
        super().__init__()
        self.self = MoEBertSelfAttention(config, n_expert=n_expert)
        self.output = MoEBertSelfOutput(config, n_expert=n_expert)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        expert_idx,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            expert_idx,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states, expert_idx)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MoEBertLayer(nn.Module):
    def __init__(self, config, use_attn_moe=False, use_fwd_moe=True, is_tok_moe=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 3.0.2 does not have this yet, set to 0
        self.seq_len_dim = 1
        self.num_expert = config.num_expert
        self.use_attn_moe = use_attn_moe
        self.use_fwd_moe = use_fwd_moe
        self.is_tok_moe = is_tok_moe
        # if self.num_expert <= 1:
        #     raise ValueError("num_expert is required >1")

        if self.use_attn_moe and self.is_tok_moe:
            raise ValueError("Currently, is_tok_moe and use_attn_moe can not be set at the same time.")

        if config.is_decoder and self.is_tok_moe:
            raise ValueError("Currently, decoder does not support token-level MoE.")

        if self.use_attn_moe:
            self.attention = MoEBertAttention(config, n_expert=self.num_expert)
        else:
            self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention  # 3.0.2 does not have this yet, set to False
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config)

        if self.use_fwd_moe:
            self.moe_layer = MoELayer(config, n_expert=self.num_expert)
        else:
            self.output = BertOutput(config)
            self.intermediate = BertIntermediate(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        expert_id=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        batch_size, seq_len, hidden_dim = hidden_states.shape

        attn_inputs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "output_attentions": output_attentions,
            #"past_key_value": self_attn_past_key_value,
        }

        if self.use_attn_moe:
            attn_inputs["expert_idx"] = expert_id
        self_attention_outputs = self.attention(**attn_inputs)
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        if self.is_tok_moe:
            # Reshapes inputs for token level MoE.
            attention_output = attention_output.view(batch_size * seq_len, 1, hidden_dim)
            self.expert_id = select_expert(batch_size * seq_len, self.num_expert, expert_id=expert_id)
        else:
            self.expert_id = select_expert(batch_size, self.num_expert, expert_id=expert_id)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # KM: for transformers 3.0.2, this function arguments have different ordering
        # layer_output = apply_chunking_to_forward(
        #     self.chunk_size_feed_forward, self.seq_len_dim, self.feed_forward_chunk, attention_output
        # )

        if self.is_tok_moe:
            # Reshapes the outupt back to the input shape.
            layer_output = layer_output.view(batch_size, seq_len, hidden_dim)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        if self.use_fwd_moe:
            layer_output = self.moe_layer(attention_output, self.expert_id)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def sample_gumbel(shape, epsilon=1e-8):
    """Samples from gumbel distribution."""
    gk = torch.rand(shape)
    return -torch.log(-torch.log(gk + epsilon) + epsilon)


def gumbel_softmax(logits, tau=1.0, hard=True):
    y = logits + sample_gumbel(logits.shape).to(logits.device)
    y = torch.nn.functional.softmax(y/tau, dim=-1)
    if hard:
        max_y, _ = torch.max(y, dim=-1)
        y = (max_y.view(-1, 1) - y).detach() + y
    return y


def onehot_max(logits):
    _, max_ind = torch.max(logits, dim=-1)
    y = torch.nn.functional.one_hot(max_ind, num_classes=logits.size(-1))
    return y


class ExpertGating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_expert = config.num_expert
        self.fwd_func = nn.Linear(self.hidden_dim, self.num_expert)
        self.fwd_func.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x, tau=1.0, mask=None):
        logits = self.fwd_func(x)
        if mask is not None:
            logits += torch.log(mask)

        if self.training:
            # one_hot = gumbel_softmax(logits, tau=tau, hard=True)
            one_hot = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            one_hot = onehot_max(logits)

        _, max_ind = torch.max(one_hot, dim=-1)
        weight = torch.sum(one_hot, dim=-1, keepdim=True)

        p_y = nn.functional.softmax(logits, dim=-1)
        log_p_y = nn.functional.log_softmax(logits, dim=-1)
        entropic_loss = torch.sum(p_y * log_p_y, dim=-1).mean()

        return weight, max_ind, entropic_loss


class MoEBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        moe_type = config.moe_type.split(":")[0]
        self.use_infer_expert = config.use_infer_expert
        self.per_layer_gating = config.per_layer_gating

        # TODO(chenghao): Makes this configurable later.
        self.per_layer_gate_input = True
        is_moe_layer = moe_type_to_func[moe_type]
        logger.info(f"Using {moe_type} for MoEBERT")
        logger.info(f"Using {config.moe_type}")
        self.is_tok_moe = False
        if "tok" in config.moe_type.split(":")[1]:
            self.is_tok_moe = True
            logger.info("Using token-level moe")
        else:
            logger.info("Using sentence-level moe")

        if "attn" in config.moe_type.split(":")[1]:
            use_attn_moe = True
            logger.info("Using moe for attention layers")
        else:
            use_attn_moe = False
        if "fwd" in config.moe_type.split(":")[1]:
            use_fwd_moe = True
            logger.info("Using moe for forward layers")
        else:
            use_fwd_moe = False

        self.layer = nn.ModuleList([
            MoEBertLayer(config, use_attn_moe=use_attn_moe, use_fwd_moe=use_fwd_moe, is_tok_moe=self.is_tok_moe)
            if is_moe_layer(layer_id) else BertLayer(config) for layer_id in range(config.num_hidden_layers)])
        self.num_gate_layer = sum([
            1.0
            if is_moe_layer(layer_id) else 0.0 for layer_id in range(config.num_hidden_layers)])
        if self.use_infer_expert:
            logger.info("Using gating for selecting expert")
            if self.per_layer_gating:
                logger.info("Using per layer gating")
                self.expert_gate = nn.ModuleList([ExpertGating(config) if is_moe_layer(layer_id) else None for layer_id in range(config.num_hidden_layers)])
            else:
                logger.info("Using single layer for all gating")
                if self.per_layer_gate_input:
                    logger.info("Each layer will select different gate")
                else:
                    logger.info("All layers will select the same gate")
                self.expert_gate = ExpertGating(config)
        else:
            logger.info("Using predefined expert")
            self.expert_gate = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        expert_id=None,
        expert_id_offset=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.expert_gate is not None and expert_id is not None:
            logger.warning("When expert_id is specified, the expert_gate is not used")

        if not self.use_infer_expert:
            if expert_id is None:
                raise ValueError("expert_id is required when use_infer_expert=False")
            if self.is_tok_moe and expert_id is not None:
                raise ValueError("Input expert_id is not allowed for token-level MoE without infer expert.")

        batch_size, seq_len, hidden_dim = hidden_states.shape
        next_decoder_cache = () if use_cache else None
        expert_weight = None
        total_entropy_loss = 0.0
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            is_moe_layer = isinstance(layer_module, MoEBertLayer)

            if self.expert_gate is not None and is_moe_layer:
                if self.is_tok_moe:
                    # For token level moe, reshapes the input into the shape of
                    # [batch_size x seq_len, hidden_dim].
                    gate_input = hidden_states.view(-1, hidden_dim)
                else:
                    # TODO(chenghao): makes the sentinel token configurable.
                    gate_input = hidden_states[:, 0, :]
                if self.per_layer_gating:
                    expert_weight, local_expert_id, entro_loss = self.expert_gate[i](gate_input)
                    total_entropy_loss += entro_loss
                elif self.per_layer_gate_input or expert_weight is None:
                    expert_weight, local_expert_id, entro_loss = self.expert_gate(gate_input)
                    total_entropy_loss += entro_loss
                else:
                    expert_weight = expert_weight.clone()
                    local_expert_id = local_expert_id.clone()
            else:
                local_expert_id = expert_id
                expert_weight = None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                if is_moe_layer:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        expert_id=local_expert_id,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
            else:
                if is_moe_layer:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,  
                        output_attentions,
                        expert_id=local_expert_id,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,  # for 3.0.2 compatibility, comment this out
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]
            # if expert_weight is not None:
            #     if self.is_tok_moe:
            #         hidden_states *= expert_weight.view(batch_size, seq_len, 1)
            #     else:
            #         hidden_states *= expert_weight.view(batch_size, 1, 1)
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        total_entropy_loss /= self.num_gate_layer

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return (hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions)


class MoEBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = MoEBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings


    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        expert_id=None,
        expert_id_offset=0,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            expert_id=expert_id,
            expert_id_offset=expert_id_offset,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ] + (embedding_output,)  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
