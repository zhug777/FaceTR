import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from models.SETR.transformer_model import TransModel2d, TransConfig, ACT2FN


class Encoder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.patch_embed = InputDense2d(config)
        #self.pos_embed = TransEmbeddings(config)
        self.encoder = TransModel2d(config)
        assert config.img_size[0] * config.img_size[1] * config.hidden_size % 256 == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.img_size[0] * config.img_size[1] * config.hidden_size // 256)
        self.img_size = config.img_size
        self.hh = self.img_size[0] // 16
        self.ww = self.img_size[1] // 16

    def forward(self, x, output_all_encoders=False):
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        assert h % self.img_size[0] == 0, "请重新输入img size 参数 必须整除"
        assert w % self.img_size[1] == 0, "请重新输入img size 参数 必须整除"
        hh = h // self.img_size[0] 
        ww = w // self.img_size[1] 

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.img_size[0], p2 = self.img_size[1])
        #x = self.patch_embed(x)
        #x = self.pos_embed(x)
        encode_x = self.encoder(x)
        x = self.final_dense(encode_x[-1])
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.img_size[0] // 16, p2 = self.img_size[1] // 16, h = hh, w = ww, c = self.config.hidden_size)
        if not output_all_encoders:
            encode_x = encode_x[-1]
        return encode_x, x


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
    def __init__(self, config):
        super().__init__()
        in_channels = config.hidden_size
        out_channels = config.out_channels
        features = config.decoder_features
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
        
'''
class Decoder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.hidden_size
        out_channels = config.out_channels
        features = config.decoder_features
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x
'''

class SETRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()      
        config = TransConfig(img_size=tuple(cfg.MODEL.PATCH_SIZE), 
                            in_channels=cfg.MODEL.IN_CHANNELS, 
                            out_channels=cfg.MODEL.NUM_SEGMENTS, 
                            hidden_size=cfg.MODEL.DIM_MODEL, 
                            num_hidden_layers=cfg.MODEL.ENCODER_LAYERS, 
                            num_attention_heads=cfg.MODEL.N_HEAD,
                            hidden_act=cfg.MODEL.ATTENTION_ACTIVATION,
                            intermediate_size=cfg.MODEL.DIM_FEEDFORWARD,
                            decoder_features=cfg.MODEL.NUM_DECONV_FILTERS)
        
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(config)

    def forward(self, x):
        _, final_x = self.encoder_2d(x, output_all_encoders=False)
        x = self.decoder_2d(final_x)
        return x 

   

