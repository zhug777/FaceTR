import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from models.SETRcoattn.transformer import CoTransEncoder, TransConfig, ACT2FN


class InputDense2d(nn.Module):
    def __init__(self, config):
        super(InputDense2d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        return self.norm(self.act(self.dense(x)))


class TransEmbeddings(nn.Module): # 生成word embedding
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)#.expand(input_shape[:2])
       
        pos_embeddings = self.pos_embeddings(position_ids)
        embeddings = x + pos_embeddings
        embeddings = self.dropout(self.LayerNorm(embeddings))
        return embeddings


class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        assert config.img_size[0] % config.patch_size[0] == 0, "请重新输入img size 参数 必须整除16"
        assert config.img_size[1] % config.patch_size[1] == 0, "请重新输入img size 参数 必须整除16"
        self.hh = config.img_size[0] // config.patch_size[0]
        self.ww = config.img_size[1] // config.patch_size[1]
        self.patch_size = config.patch_size
        self.patch_embed1 = InputDense2d(config)
        self.patch_embed2 = InputDense2d(config)
        self.pos_embed1 = TransEmbeddings(config)
        self.pos_embed2 = TransEmbeddings(config)
        self.coattn_encoder = CoTransEncoder(config)
        #assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % 256 == 0, "不能除尽"
        #self.final_dense1 = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // 256)
        #self.final_dense2 = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // 256)

    def forward(self, x, mask=None, output_all_encoders=False):
        x = rearrange(x, 'b c (hh p1) (ww p2) -> b (hh ww) (p1 p2 c)', p1 = self.patch_size[0], p2 = self.patch_size[1])
        y = x
        x = self.patch_embed1(x)
        x = self.pos_embed1(x)
        y = self.patch_embed2(y)
        y = self.pos_embed2(y)

        encoder_output = self.coattn_encoder(x, y, mask)
        x, y = encoder_output[-1][0], encoder_output[-1][1]
        #x = self.final_dense1(encoder_output[0])
        #y = self.final_dense2(encoder_output[1])
        x = rearrange(x, "b (h w) c -> b c h w", h = self.hh, w = self.ww, c = self.config.hidden_size)
        y = rearrange(y, "b (h w) c -> b c h w", h = self.hh, w = self.ww, c = self.config.hidden_size)
        if not output_all_encoders:
            encoder_output = encoder_output[-1]
        return encoder_output, x, y


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )
    def forward(self, x):
        return self.layer(x)


class Decoder2D(nn.Module):
    def __init__(self, config, out_channels, features):
        super().__init__()
        in_channels = config.hidden_size
        self.decoder_1 = DecoderLayer(in_channels, features[0])
        self.decoder_2 = DecoderLayer(features[0], features[1])
        self.decoder_3 = DecoderLayer(features[1], features[2])
        self.decoder_4 = DecoderLayer(features[2], features[3])
        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class CoSETRModel(nn.Module):
    def __init__(self, img_size=(224, 224),
                        patch_size=(16, 16), 
                        in_channels=3, 
                        out_channels_x=11,
                        out_channels_y=106,
                        hidden_size=1024, 
                        num_hidden_blocks=4,
                        num_attention_heads=16,
                        decode_features_x=[512, 256, 128, 64],
                        decode_features_y=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(img_size=img_size,
                            patch_size=patch_size,
                            in_channels=in_channels, 
                            out_channels_x=out_channels_x,
                            out_channels_y=out_channels_y,
                            hidden_size=hidden_size, 
                            num_hidden_blocks=num_hidden_blocks,
                            num_attention_heads=num_attention_heads)
        self.encoder = Encoder2D(config)
        self.decoder_x = Decoder2D(config, out_channels=config.out_channels_x, features=config.decoder_features_x)
        self.decoder_y = Decoder2D(config, out_channels=config.out_channels_y, features=config.decoder_features_y)

    def forward(self, x):
        _, final_x, final_y = self.encoder(x, output_all_encoders=False)
        x = self.decoder_x(final_x)
        y = self.decoder_y(final_y)
        return x, y

'''
class CoSETRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = TransConfig(img_size=tuple(cfg.MODEL.IMAGE_SIZE),
                            patch_size=tuple(cfg.MODEL.PATCH_SIZE), 
                            in_channels=cfg.MODEL.IN_CHANNELS, 
                            out_channels_x=cfg.MODEL.NUM_SEGMENTS,
                            out_channels_y=cfg.MODEL.NUM_KPTS, 
                            hidden_size=cfg.MODEL.DIM_MODEL, 
                            num_hidden_layers=cfg.MODEL.ENCODER_LAYERS, 
                            num_attention_heads=cfg.MODEL.N_HEAD,
                            hidden_act=cfg.MODEL.ATTENTION_ACTIVATION,
                            intermediate_size=cfg.MODEL.DIM_FEEDFORWARD,
                            decoder_features_x=cfg.MODEL.NUM_DECONV_FILTERS,
                            decoder_features_y=cfg.MODEL.NUM_DECONV_FILTERS)
        self.encoder = Encoder2D(config)
        self.decoder_x = Decoder2D(config, out_channels=config.out_channels_x, features=config.decoder_features_x)
        self.decoder_y = Decoder2D(config, out_channels=config.out_channels_y, features=config.decoder_features_y)

    def forward(self, x):
        _, final_x, final_y = self.encoder(x, output_all_encoders=False)
        x = self.decoder_x(final_x)
        y = self.decoder_y(final_y)
        return x, y
'''
   

