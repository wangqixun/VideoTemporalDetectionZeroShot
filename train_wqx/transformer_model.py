import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import time
# import matplotlib.pyplot as plt
import math
import yaml
# from fast_ctc_decode import beam_search, viterbi_search
# import parasail
from collections import deque, defaultdict, OrderedDict
import re



class SelfAttention(nn.Module):
    def __init__(self, 
        num_attention_heads=12,
        hidden_size=768,
        attention_probs_dropout_prob=0.1,
        position_embedding_type='absolute',
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type

    def transpose_for_scores(self, x):
        # [bs, N, heads, head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [bs, heads, N, head_size]
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            attention_scores_res=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # Self attention
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        # Not Self attention
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            attention_mask = attention_mask

        # shape: [bs, N, all_head_size] -> [bs, heads, N, head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # shape = [bs, heads, N, N]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        # RealFormer 
        if attention_scores_res is not None:
            attention_scores = attention_scores + attention_scores_res
            attention_scores_res = attention_scores

        # 消除由于长度不一引起的 padding 影响
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # v: context_layer.shape = [bs, N, hidden_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # shape: [bs, heads, N, head_size] -> [bs, N, heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # padding 部分置零
        # context_layer = context_layer * (attention_mask[:, 0, 0, :] > -5000).type_as(context_layer).unsqueeze(-1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if attention_scores_res is not None:
            outputs += (attention_scores_res,)
        else:
            outputs += (None,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, 
        hidden_size=768,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Attention(nn.Module):
    def __init__(self, 
        num_attention_heads=12,
        hidden_size=768,
        attention_probs_dropout_prob=0.1,
        position_embedding_type='absolute',
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.self = SelfAttention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
        )
        self.output = SelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            attention_scores_res=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            attention_scores_res,
        )
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Intermediate(nn.Module):
    def __init__(self, 
        hidden_size=768,
        intermediate_size=1024,
        hidden_act='gelu',
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = getattr(F, hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self,
        hidden_size=768,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, 
        num_attention_heads=12,
        hidden_size=768,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        position_embedding_type='absolute',
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.attention = Attention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.intermediate = Intermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.output = Output(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            attention_scores_res=None,
    ):
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            attention_scores_res,
        )
        attention_output = hidden_states[0]
        attention_scores_res = hidden_states[-1]
        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output)

        return hidden_states, attention_scores_res


class TransformerLayerDecoder(nn.Module):
    def __init__(self, 
        num_attention_heads=12,
        hidden_size=768,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        position_embedding_type='absolute',
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.attention_self = Attention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.attention_encoder = Attention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.intermediate = Intermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.output = Output(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
            self,
            hidden_states,                     # q
            attention_mask=None,               # 做 self attention 时候需要的mask。做翻译的decoder的时候需要。视频零样本时序预测不需要
            head_mask=None,                    # head上加mask。基本不需要
            encoder_hidden_states=None,        # kv
            encoder_attention_mask=None,       # 做 cross attention 时候需要的mask。用来消除文本长度不齐进行padding带来的影响
            output_attentions=False,           # 输出attention结果。不要输出
            attention_scores_res_self=False,   # real former 中上一层的 self attention
            attention_scores_res_encoder=False,# real former 中上一层的 cross attention
    ):
        hidden_states = self.attention_self(
            hidden_states,
            attention_mask,
            head_mask,
            None,
            None,
            output_attentions,
            attention_scores_res_self,
        )
        hidden_states, attention_scores_res_self = hidden_states

        hidden_states = self.attention_encoder(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            attention_scores_res_encoder,
        )
        attention_output, attention_scores_res_encoder = hidden_states

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output)
        return hidden_states, attention_scores_res_self, attention_scores_res_encoder


class VisionLanguageDecoder(nn.Module):
    def __init__(self, 
        num_transformer_decoder_layers=4,
        num_attention_heads=12,
        hidden_size=768,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        position_embedding_type='absolute',
        layer_norm_eps=1e-12,
        real_former=False,
    ):
        super().__init__()

        self.transformer = nn.ModuleList([
            TransformerLayerDecoder(
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_act=hidden_act,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                position_embedding_type=position_embedding_type,
                layer_norm_eps=layer_norm_eps,
            ) 
            for _ in range(num_transformer_decoder_layers)
        ])
        self.real_former = real_former

    def forward(self, 
        vision_feature,                # bs, N_v, 768
        language_feature,              # bs, N_l, 768
        language_mask,                 # bs, N_l
        vision_attention_mask=None,
        language_attention_mask=None,
    ):
        # 做 self attention 时候需要的mask。做翻译的decoder的时候需要。视频零样本时序预测不需要
        if vision_attention_mask is None:
            vision_attention_mask = 0
        
        # 做 cross attention 时候需要的mask。用来消除文本长度不齐进行padding带来的影响
        if language_attention_mask is None:
            language_attention_mask = (1 - language_mask[..., None]) * -9999

        hidden_states = vision_feature
        if self.real_former:
            attention_scores_res_self = 0
            attention_scores_res_encoder = 0
        else:
            attention_scores_res_self = None
            attention_scores_res_encoder = None

        for i, layer_module in enumerate(self.transformer):
            hidden_states = layer_module(
                hidden_states=hidden_states,                               # q
                attention_mask=vision_attention_mask,                      # q mask
                encoder_hidden_states=language_feature,                    # kv
                encoder_attention_mask=language_attention_mask,            # kv mask
                attention_scores_res_self=attention_scores_res_self,       # self realf ormer
                attention_scores_res_encoder=attention_scores_res_encoder, # cross realf ormer
            )
            hidden_states, attention_scores_res_self, attention_scores_res_encoder = hidden_states

        return hidden_states




if __name__ == '__main__':
    from rich import print

    self_attention = SelfAttention()
    self_output = SelfOutput()
    attention = Attention()
    intermediate = Intermediate()
    output = Output()
    transformer_layer = TransformerLayer()
    transformer_decoder_layer = TransformerLayerDecoder()

    x0 = torch.rand(10, 13, 768)
    att_realformer0 = None
    x1, att_realformer1 = transformer_layer(x0, attention_scores_res=att_realformer0)
    x2, att_realformer2 = transformer_layer(x1, attention_scores_res=att_realformer1)
    x3, att_realformer3 = transformer_layer(x2, attention_scores_res=att_realformer2)
    print(x0.shape, )
    print(x1.shape, att_realformer1.shape if att_realformer1 is not None else att_realformer1)
    print(x2.shape, att_realformer2.shape if att_realformer2 is not None else att_realformer2)
    print(x3.shape, att_realformer3.shape if att_realformer3 is not None else att_realformer3)


    # hidden_states,
    # attention_mask=selfattention_mask,
    # head_mask=None,
    # encoder_hidden_states=input_encoder,
    # encoder_attention_mask=encoderattention_mask,
    # output_attentions=False,
    # attention_scores_res_self=attention_scores_res_self,
    # attention_scores_res_encoder=attention_scores_res_encoder,

    x = torch.rand(10, 7, 768)
    x0 = torch.rand(10, 13, 768)
    att_realformer0, att_realformer00 = 0, 0
    x1, att_realformer1, att_realformer11 = transformer_decoder_layer(x0, encoder_hidden_states=x, attention_scores_res_self=att_realformer0, attention_scores_res_encoder=att_realformer00)
    x2, att_realformer2, att_realformer22 = transformer_decoder_layer(x1, encoder_hidden_states=x, attention_scores_res_self=att_realformer1, attention_scores_res_encoder=att_realformer11)
    x3, att_realformer3, att_realformer33 = transformer_decoder_layer(x2, encoder_hidden_states=x, attention_scores_res_self=att_realformer2, attention_scores_res_encoder=att_realformer22)
    print(x0.shape, )
    print(x1.shape, att_realformer1.shape if att_realformer1 is not None else att_realformer1)
    print(x2.shape, att_realformer2.shape if att_realformer2 is not None else att_realformer2)
    print(x3.shape, att_realformer3.shape if att_realformer3 is not None else att_realformer3)



























    pass