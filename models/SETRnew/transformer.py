import logging
import math
import os

import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange

def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class TransConfig(object):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_channels=3,
        out_channels=11,
        kpt_num=None,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=1024,
        decoder_features = [512, 256, 128, 64],
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kpt_num = kpt_num
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.decoder_features = decoder_features
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class MultiHeadedSelfAttention(nn.Module):  # multi-head attention
    def __init__(self, config: TransConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.proj_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def trans(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        ## 最后x.shape = (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = self.trans(q), self.trans(k), self.trans(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            attention_scores -= 10000.0 * (1.0 - mask)
        # Normalize the attention scores to probabilities.
        attention_scores = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        # 注意力加权, reshape得到[batch_size, length, embedding_dimension]
        context_layer = torch.matmul(attention_scores, v).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]  # relu
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dense2(self.act(self.dense1(x)))


class TransLayer(nn.Module): # embedding层
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pwff = PositionWiseFeedForward(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, mask):
        h = self.dropout(self.proj(self.attn(x, mask)))
        x = x + h
        h = self.dropout(self.pwff(self.norm1(x)))
        x = x + h
        x = self.norm2(x)
        return x


class TransEncoder(nn.Module): # n * encoder layers
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask):
        all_encoder_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            all_encoder_layers.append(x)
        return all_encoder_layers

