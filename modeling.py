# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# Modified for Bitcoin Price Forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps=1e-3):
        super().__init__()
        if output_size and output_size != hidden_size:
            self.ln = nn.LayerNorm(output_size, eps=eps)
        else:
            self.ln = nn.Identity()
    
    def forward(self, x):
        return self.ln(x)

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, context_size=None, dropout=0):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.lin_c = nn.Linear(context_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = nn.GLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a, c=None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        if self.out_proj:
            x = x + self.out_proj(a)
        return self.layer_norm(x)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        
        # 多模态Embedding层
        self.day_embed = nn.Embedding(7, config.d_model)       # 星期几
        self.peak_embed = nn.Embedding(2, config.d_model)      # 峰值时段
        self.fin_embed = nn.Linear(17, config.d_model)         # 金融特征
        self.med_embed = nn.Linear(6, config.d_model)          # 媒体特征
        self.mkt_embed = nn.Linear(1, config.d_model)          # 市场情绪
        
        # 时序编码器
        self.lstm = nn.LSTM(config.d_model, config.d_model, num_layers=3)
        self.attention = nn.MultiheadAttention(config.d_model, num_heads=4)
        
        # 输出层
        self.output_layer = nn.Linear(config.d_model, 1)       # 单点输出

    def forward(self, x):
        # 输入分解 [B, T, 24]
        finance = x[:, :, :17]          # 金融17维
        media = x[:, :, 17:23]          # 媒体6维
        market = x[:, :, 23:]           # 市场情绪1维
        time_feat = x[:, :, -3].long()  # 星期几（假设为最后第3列）
        is_peak = (media[:, :, 0] > 1000).long()  # 峰值判断
        
        # Embedding处理
        fin_emb = self.fin_embed(finance)
        med_emb = self.med_embed(media)
        mkt_emb = self.mkt_embed(market)
        day_emb = self.day_embed(time_feat)
        peak_emb = self.peak_embed(is_peak)
        
        # 特征合并
        combined = fin_emb + med_emb + mkt_emb + day_emb + peak_emb  # [B, T, d_model]
        
        # LSTM + Attention
        lstm_out, _ = self.lstm(combined.permute(1,0,2))      # [T, B, d_model]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 单点预测
        output = self.output_layer(attn_out[-1])              # 取最后一个时间步
        return output.squeeze(1)                              # [B]