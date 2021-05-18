import logging
import math
import os

import torch
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
        img_size=(16, 16),
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


class TransLayerNorm(nn.Module): # 实现layernorm
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(TransLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps # 防止除0的参数      

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
      

class TransSelfAttention(nn.Module): # multi-head attention
    def __init__(self, config: TransConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.proj_q = nn.Linear(config.hidden_size, self.all_head_size)
        self.proj_k = nn.Linear(config.hidden_size, self.all_head_size)
        self.proj_v = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def trans(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        ## 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
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
        # 注意力加权, 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = torch.matmul(attention_scores, v).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TransSelfOutput(nn.Module): # multi-head attention输出concat之后连接的线性变换层
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, input_tensor):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm(x + input_tensor)
        return x


class TransAttention(nn.Module): # 对应embedding层中multi-head attention + add&norm的部分
    def __init__(self, config):
        super().__init__()
        self.self = TransSelfAttention(config)
        self.output = TransSelfOutput(config)

    def forward(self, x, mask):
        self_outputs = self.self(x, mask)
        attention_output = self.output(self_outputs, x)
        
        return attention_output

# TransIntermediate与TransOutput一同构成embedding层中feed forward + add&norm的部分
class TransIntermediate(nn.Module):  
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act] ## relu 

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        return x

class TransOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, input_tensor):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm(x + input_tensor)
        return x


class TransLayer(nn.Module): # embedding层
    def __init__(self, config):
        super().__init__()
        self.attention = TransAttention(config)
        self.intermediate = TransIntermediate(config)
        self.output = TransOutput(config)

    def forward(self, x, mask):
        attention_output = self.attention(x, mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransEncoder(nn.Module): # n * encoder layers
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask=None):
        all_encoder_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            all_encoder_layers.append(x) 
        return all_encoder_layers
