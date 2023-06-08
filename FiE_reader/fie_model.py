# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

from transformers.models.electra.modeling_electra import ElectraPreTrainedModel, ElectraEmbeddings, ElectraSelfOutput, ElectraIntermediate, ElectraOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import math
import torch.nn.functional as F

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def listmle(thres, scores, labels):
    full_scores = torch.cat([thres, scores], dim=1)
    full_labels = torch.cat([torch.full(thres.size(), 0.5, device=thres.device), labels], dim=1)
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

def crosslistmle(scores, labels):
    sorted_labels, indices = labels.sort(dim=1, descending=True)
    sorted_scores = scores.gather(dim=1, index=indices)
    sorted_scores = sorted_scores.exp()
    loss = 0.0
    for i in range(sorted_scores.shape[0]):
        for j in range(sorted_scores.shape[1]):
            rest = sorted_scores[i][sorted_labels[i] < sorted_labels[i][j]]
            prob = sorted_scores[i][j] / (sorted_scores[i][j] + rest.sum())
            loss += torch.log(prob)
            if len(rest) == 0:
                break
    return -loss


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Electra
class ElectraSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

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
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False, num_ctx=None, global_tokens_embed=None
    ):
        mixed_query_layer = self.query(hidden_states)

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
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        if global_tokens_embed is not None:
            b_size = hidden_states.shape[0]
            # KM: global_tokens_embed has shape (num_q, 10, hidden_size), since each question has unique 10 global tokens after contextualization
            global_query_layer = self.transpose_for_scores(self.query(global_tokens_embed))
            global_key_layer = self.transpose_for_scores(self.key(global_tokens_embed))
            global_value_layer = self.transpose_for_scores(self.value(global_tokens_embed))

            # KM: we reshape the num_ctx passage path to be a long sequence as done in FiD, so that global tokens can attend to them all
            full_ctx_key_layer = key_layer.permute(0, 2, 1, 3)
            real_b_size = torch.div(b_size, num_ctx, rounding_mode='floor')
            full_ctx_key_layer = full_ctx_key_layer.view(real_b_size, num_ctx*full_ctx_key_layer.shape[1], self.num_attention_heads, self.attention_head_size)
            full_ctx_key_layer = full_ctx_key_layer.permute(0, 2, 1, 3)

            full_ctx_value_layer = value_layer.permute(0, 2, 1, 3)
            full_ctx_value_layer = full_ctx_value_layer.view(real_b_size, num_ctx*full_ctx_value_layer.shape[1], self.num_attention_heads, self.attention_head_size)
            full_ctx_value_layer = full_ctx_value_layer.permute(0, 2, 1, 3)

            full_ctx_key_layer = torch.cat([global_key_layer, full_ctx_key_layer], dim=2)
            full_ctx_value_layer = torch.cat([global_value_layer, full_ctx_value_layer], dim=2)

            global_attention_scores = torch.matmul(global_query_layer, full_ctx_key_layer.transpose(-1, -2))
            global_attention_scores = global_attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                global_attention_mask = attention_mask.permute(0, 3, 1, 2)
                global_attention_mask = global_attention_mask.view(real_b_size, num_ctx*global_attention_mask.shape[1], 1, 1)
                global_attention_mask = global_attention_mask.permute(0, 2, 3, 1)
                global_attention_mask = torch.cat([torch.zeros(real_b_size, 1, 1, global_tokens_embed.shape[1]).to(global_attention_mask.device), global_attention_mask], dim=3)
                global_attention_scores = global_attention_scores + global_attention_mask
            global_attention_probs = nn.Softmax(dim=-1)(global_attention_scores)
            global_attention_probs = self.dropout(global_attention_probs)
            global_context_layer = torch.matmul(global_attention_probs, full_ctx_value_layer)
            global_context_layer = global_context_layer.permute(0, 2, 1, 3).contiguous()
            new_global_context_layer_shape = global_context_layer.size()[:-2] + (self.all_head_size,)
            global_context_layer = global_context_layer.view(*new_global_context_layer_shape)
            # KM: After global tokens attention is done, we concatenate the previous layer global tokens with each passage path, for regular attention computation
            # KM: we repeat this num_ctx times to allow each passage path to attend to global tokens
            global_key_layer = global_key_layer.unsqueeze(1).repeat(1, num_ctx, 1, 1, 1).view(b_size, self.num_attention_heads, global_tokens_embed.shape[1], self.attention_head_size)
            global_value_layer = global_value_layer.unsqueeze(1).repeat(1, num_ctx, 1, 1, 1).view(b_size, self.num_attention_heads, global_tokens_embed.shape[1], self.attention_head_size)
            key_layer = torch.cat([global_key_layer, key_layer], dim=2)
            value_layer = torch.cat([global_value_layer, value_layer], dim=2)
            attention_mask = torch.cat([torch.zeros(b_size, 1, 1, global_tokens_embed.shape[1]).to(hidden_states.device), attention_mask], dim=3)
        query_layer = self.transpose_for_scores(mixed_query_layer)

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
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
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

        outputs = (context_layer,)
        if global_tokens_embed is not None:
            outputs = outputs + (global_context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Electra
class ElectraAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ElectraSelfAttention(config)
        self.output = ElectraSelfOutput(config)
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
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False, num_ctx=None, global_tokens_embed=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions, num_ctx, global_tokens_embed
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        if global_tokens_embed is not None:
            global_attention_output = self.output(self_outputs[1], global_tokens_embed)
            outputs = (attention_output, global_attention_output) + self_outputs[2:]  # add attentions if we output them
        else:
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class ElectraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElectraAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ElectraAttention(config)
        self.intermediate = ElectraIntermediate(config)
        self.output = ElectraOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False, num_ctx=None, global_tokens_embed=None
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value, num_ctx=num_ctx, global_tokens_embed=global_tokens_embed
        )
        attention_output = self_attention_outputs[0]
        if global_tokens_embed is not None:
            global_attention_output = self_attention_outputs[1]
            outputs = self_attention_outputs[2:]
        else:
            outputs = self_attention_outputs[1:]

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

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        if global_tokens_embed is not None:
            global_layer_output = self.feed_forward_chunk(global_attention_output)
            outputs = (layer_output, global_layer_output) + outputs
        else:
            outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class FiElectraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])
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
        return_dict=True, num_ctx=None, global_tokens_embed=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        if global_tokens_embed is not None:
            real_b_size = torch.div(hidden_states.size(0), num_ctx, rounding_mode='floor')
            global_tokens_embed = global_tokens_embed.expand(real_b_size, -1, -1)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return module(*inputs, past_key_value, output_attentions, num_ctx, global_tokens_embed)

                #     return custom_forward
                output_attentions = torch.tensor(output_attentions, dtype=torch.bool)
                num_ctx = torch.tensor(num_ctx, dtype=torch.long)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,   # past_key_value is always None with gradient checkpointing
                    output_attentions,
                    num_ctx,
                    global_tokens_embed
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    num_ctx, global_tokens_embed
                )

            hidden_states = layer_outputs[0]
            if global_tokens_embed is not None:
                global_tokens_embed = layer_outputs[1]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class FiElectraModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = FiElectraEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        num_ctx=None, global_tokens_embed=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, num_ctx=num_ctx, global_tokens_embed=global_tokens_embed
        )

        return hidden_states

class FiEModel(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.model_name = args.model_name
        self.sp_weight = args.sp_weight
        self.ans_weight = args.answer_weight
        self.encoder = FiElectraModel.from_pretrained(args.model_name)
        self.encoder.encoder.gradient_checkpointing = args.gradient_checkpointing
        self.sentlistmle = args.sentlistmle
        if "electra" in args.model_name and self.sentlistmle:
            self.pooler = BertPooler(config)

        self.span_outputs = nn.Linear(config.hidden_size*2, 1)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.num_pasg = args.num_ctx

        self.global_tokens = torch.LongTensor([[_ for _ in range(500, 500+args.num_global_tokens)]])
        self.max_ans_len = args.max_ans_len

    def forward(self, batch):

        global_tokens_embed = self.encoder.embeddings(self.global_tokens.to(self.encoder.device))
        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], num_ctx=batch['num_ctx'], global_tokens_embed=global_tokens_embed)

        if "electra" in self.model_name:
            sequence_output = outputs[0]
            if self.sentlistmle:
                pooled_output = self.pooler(sequence_output)
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]

        span_positions = batch['token_type_ids'].unsqueeze(-1) + batch['token_type_ids'].unsqueeze(-2)
        span_positions = torch.tril(torch.triu(span_positions, 0), self.max_ans_len)
        span_logits = []
        for i in range(span_positions.size(0)):
            valid_span_indices = span_positions[i].eq(2).nonzero()
            start_rep = torch.index_select(sequence_output[i], 0, valid_span_indices[:, 0])
            end_rep = torch.index_select(sequence_output[i], 0, valid_span_indices[:, 1])
            span_reps = torch.cat([start_rep, end_rep], dim=-1)
            span_logits.append(self.span_outputs(span_reps))

        if self.sentlistmle:
            sent_thres = self.sp(pooled_output)
        else:
            sent_thres = None

        sp_score = None
        if self.training:
            sp_loss = None
            b_size, seq_len = batch['input_ids'].shape[0], batch['input_ids'].shape[1]
            real_b_size = torch.div(b_size, self.num_pasg, rounding_mode='floor')
            global_span_logits = []
            for b in range(0, b_size, self.num_pasg):
                global_span_logits.append(torch.cat(span_logits[b:b+self.num_pasg], dim=0))
            span_labels = torch.zeros((b_size, seq_len, seq_len), device=batch['input_ids'].device, dtype=torch.long)
            for b in range(b_size):
                for i in range(len(batch['starts'][b])):
                    if batch['starts'][b][i] == -1:
                        break 
                    span_labels[b, batch['starts'][b][i], batch['ends'][b][i]] = 1

            span_positions = span_positions.view(real_b_size, batch['num_ctx'], span_positions.shape[1], span_positions.shape[2])
            span_labels = span_labels.view(real_b_size, batch['num_ctx'], span_labels.shape[1], span_labels.shape[2])
            span_labels = span_labels.view(span_labels.shape[0], -1)
            span_positions = span_positions.view(span_positions.shape[0], -1)
            span_loss = 0
            for b in range(real_b_size):
                valid_span_logits = global_span_logits[b]
                span_log_probs = F.log_softmax(valid_span_logits, dim=0)
                valid_span_labels = span_labels[b][span_positions[b].eq(2)] 
                valid_span_labels = valid_span_labels.nonzero()
                marginal_probs = torch.logsumexp(span_log_probs[valid_span_labels.squeeze(-1)], dim=0)
                span_loss -= marginal_probs

            loss = span_loss * self.ans_weight 
            if sp_loss is not None:
                loss += sp_loss * self.sp_weight
                return loss, span_loss * self.ans_weight, sp_loss * self.sp_weight
            else:
                return loss, span_loss * self.ans_weight, None

        return {
            'span_logits': span_logits,
            'span_positions': span_positions,
            "sp_score": sp_score,
            'sent_thres': sent_thres,
            }