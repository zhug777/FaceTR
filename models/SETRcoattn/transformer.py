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
        out_channels_x=11,
        out_channels_y=11,
        kpt_num=None,
        hidden_size=1024,
        num_hidden_blocks=3,
        num_attention_heads=16,
        intermediate_size=1024,
        decoder_features_x=[512, 256, 128, 64],
        decoder_features_y=[512, 256, 128, 64],
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
        self.out_channels_x = out_channels_x
        self.out_channels_y = out_channels_y
        self.kpt_num = kpt_num
        self.hidden_size = hidden_size
        self.num_hidden_blocks = num_hidden_blocks
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.decoder_features_x = decoder_features_x
        self.decoder_features_y = decoder_features_y
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class MultiHeadedSelfAttention(nn.Module):  # multi-head attention
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def trans(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        ## 最后x.shape = (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask):
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
        self.proj_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn = MultiHeadedSelfAttention(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pwff = PositionWiseFeedForward(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        h = self.dropout(self.proj(self.attn(q, k, v, mask)))
        x = x + h
        h = self.dropout(self.pwff(self.norm1(x)))
        x = x + h
        x = self.norm2(x)
        return x


class CoTransBlock(nn.Module): # embedding层
    def __init__(self, config):
        super().__init__()
        self.proj_x_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_x_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_x_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_y_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_y_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_y_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_x = MultiHeadedSelfAttention(config)
        self.attn_y = MultiHeadedSelfAttention(config)
        self.proj_x = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_y = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm_x1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_y1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pwff_x = PositionWiseFeedForward(config)
        self.pwff_y = PositionWiseFeedForward(config)
        self.norm_x2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.block_x = TransLayer(config)
        self.block_y = TransLayer(config)

    def forward(self, x, y, mask):
        x_q, x_k, x_v = self.proj_x_q(x), self.proj_x_k(x), self.proj_x_v(x)
        y_q, y_k, y_v = self.proj_y_q(y), self.proj_y_k(y), self.proj_y_v(y)

        x_h = self.dropout(self.proj_x(self.attn_x(x_q, y_k, y_v, mask)))
        x = x + x_h
        x_h = self.dropout(self.pwff_x(self.norm_x1(x)))
        x = x + x_h
        x = self.block_x(self.norm_x2(x), mask)

        y_h = self.dropout(self.proj_y(self.attn_y(y_q, x_k, x_v, mask)))
        y = y + y_h
        y_h = self.dropout(self.pwff_y(self.norm_y1(y)))
        y = y + y_h
        y = self.block_y(self.norm_y2(y), mask)
        return x, y


class CoTransEncoder(nn.Module): # n * encoder layers
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CoTransBlock(config) for _ in range(config.num_hidden_blocks)])

    def forward(self, x, y, mask):
        encoder_outputs = []
        for layer in self.layers:
            x, y = layer(x, y, mask)
            encoder_outputs.append([x, y])
        return encoder_outputs
