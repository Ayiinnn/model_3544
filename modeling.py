# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# Modified for Bitcoin Price Forecasting
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


from torch.nn import LayerNorm

class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps=1e-3):
        super().__init__()
        if output_size and output_size != hidden_size:
            self.ln = nn.LayerNorm(output_size, eps=eps)
        else:
            self.ln = nn.Identity()

    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class GRN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x


class Modified_GRN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0):
        super().__init__()

        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)

        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)

        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)  # (B, T, input_size) → (B, T, hidden_size)
        if c is not None:
            x = x + self.lin_c(c)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x


class VSN(nn.Module):

    def __init__(self, config, num_inputs):
        super().__init__()
        self.joint_grn = GRN(config.hidden_size * num_inputs, config.hidden_size, output_size=num_inputs,
                             context_hidden_size=config.hidden_size)
        self.var_grns = nn.ModuleList(
            [GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(num_inputs)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)
        # [B,k,d,d_model]->[B,k,H] [B,d,d_model]->[B,H]
        # 金融部分沿用原来的VSN

        return variable_ctx, sparse_weights


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.ll_conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()

        self.ll_conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # 第一个
        out = self.ll_conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        # 第二个
        out = self.ll_conv2(out)
        out = self.chomp2(out)

        # 残差
        out += residual

        out = self.relu2(out)
        out = self.dropout(out)

        return out


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TCN, self).__init__()
        layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ContinuousEmbedding(nn.Module):  # 极简的连续变量嵌入层
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1, 1, hidden_size))  # [1,1,1,H]
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # [1,1,1,H]
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入: [B, K, D]
        输出: [B, K, D, H]
        """
        x = x.unsqueeze(-1)
        return x * self.weight + self.bias  # [B,K,D,1] * [1,1,1,H] → [B,K,D,H]


class CovariateEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.var_weights = nn.Linear(config.hidden_size, 1)
        self.k_weights = nn.Linear(config.hidden_size, 1)
        self.context_grns = nn.ModuleList(
            [GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(3)])
        self.ce_grn = GRN(config.hidden_size, config.hidden_size, dropout=config.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # x：[B,K,N,H]
        weights = self.var_weights(x).squeeze(-1)  # [B,K,N,H] -> [B,K,N,1]
        sparse_weights = F.softmax(weights, dim=-1)  # [B,K,N,1] -> [B,K,N]
        variable_ctx = torch.einsum('bknh,bkn->bkh', x, sparse_weights)  # [B,K,N,H] * [B,K,N] -> [B,K,H]
        k_weights = self.k_weights(variable_ctx).squeeze(-1)  # ->[B, K，1]
        sparse_k_weights = F.softmax(k_weights, dim=1)  # [B, K]
        reduced_ctx = torch.einsum('bkh,bk->bh', variable_ctx, sparse_k_weights)  # [B, H]
        cs, ch, cc = tuple(m(reduced_ctx) for m in self.context_grns)
        ce = self.ce_grn(variable_ctx)
        return cs, ce, ch, cc


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Embedding层
        self.conti_embed = ContinuousEmbedding(config.hidden_size)

        # 情绪编码
        self.senti_encoder = CovariateEncoder(config)
        # 金融编码
        self.finVSN = VSN(config, config.fin_varible_num)
        self.tem_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

        # ce变换
        self.ce_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, bidirectional=False,
                                  batch_first=True)
        self.enrichment_grn = Modified_GRN(config.hidden_size, config.hidden_size,
                                           context_hidden_size=config.hidden_size, dropout=config.dropout)
        self.input_gate = GLU(config.hidden_size, config.hidden_size)
        self.input_gate_ln = LayerNorm(config.hidden_size, eps=1e-3)

        # 时序编码
        self.position_encoder = PositionalEncoding(config.hidden_size)

        self.tcn = TCN(num_inputs=config.hidden_size, num_channels=[config.hidden_size] * 6, kernel_size=2, dropout=0.1)

        self.attention_gate = GLU(config.hidden_size, config.hidden_size)
        self.attention_ln = LayerNorm(config.hidden_size, eps=1e-3)
        self.positionwise_grn = GRN(config.hidden_size,
                                    config.hidden_size,
                                    dropout=config.dropout)

        self.decoder_gate = GLU(config.hidden_size, config.hidden_size)
        self.decoder_ln = LayerNorm(config.hidden_size, eps=1e-3)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 3 * config.hidden_size),
            nn.ReLU(),
            nn.Linear(3 * config.hidden_size, 1)
        )  # 输出3个点

    def forward(self, x):
        # 注意config.model 和 config.hidden_size 应一致
        # 输入分解 [B, T, 20]
        assert len(x.shape) == 3, f"输入应为3维 [B,T,features]，实际得到 {x.shape}"  # 检查
        B, T, _ = x.shape  # 原始输入形状 #检查

        finance = x[:, :, :13]
        senti = x[:, :, 13:]

        fin_inp = self.conti_embed(finance)
        senti_inp = self.conti_embed(senti)

        assert fin_inp.shape == (B, T, 13, self.d_model), f"金融嵌入后应为(B,T,13,H)，实际{fin_inp.shape}"
        assert senti_inp.shape == (B, T, 7, self.d_model), f"情绪嵌入后应为(B,T,7,H)，实际{senti_inp.shape}"

        '''
        嵌入后的数据：
        senti_inp: [B，k=1000，d=13, H=d_model]
        fin_inp: [B，k=1000，d=7,H=d_model]
        '''

        # 情绪编码
        cs, ce, ch, cc = self.senti_encoder(senti_inp)

        '''
        cs/ch/cc: [B，H=hidden_size]
        ce: [B，k=1000, H=hidden_size]
        '''
        assert ce.shape == (B, T, self.d_model), f"情绪上下文(ce)应为(B,T,H)，实际{ce.shape}"

        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0)
        ce, _ = self.ce_encoder(ce)  # 一层LSTM

        assert ce.shape == (B, T, self.d_model), f"情绪上下文处理后(ce)应为(B,T,H)，实际{ce.shape}"

        # 金融编码
        fin_features, _ = self.finVSN(fin_inp, cs)
        assert fin_features.shape == (B, T, self.d_model), f"金融特征应为(B,T,H)，实际{fin_features.shape}"

        '''
        VSN：[B,k,d,d_model]->[B,k,H]
        '''

        fin, state = self.tem_encoder(fin_features, (ch, cc))  # LSTM，维度不变
        assert fin.shape == (B, T, self.d_model), f"LSTM输出应为(B,T,H)，实际{fin.shape}"
        main_features = fin + self.input_gate(fin_features)  # skip_connection
        main_features = self.input_gate_ln(main_features)

        # 融合
        enriched = self.enrichment_grn(main_features, c=ce)
        original_enriched = enriched
        enriched = self.position_encoder(enriched)
        assert enriched.shape == (B, T, self.d_model), f"融合特征应为(B,T,H)，实际{enriched.shape}"

        enriched = enriched.transpose(1, 2)
        assert enriched.shape == (B, self.d_model, T), f"TCN输入应为(B,H,T)，实际{enriched.shape}"

        output = self.tcn(enriched)
        assert output.shape == (B, self.d_model, T), f"TCN输出应为(B,H,T)，实际{output.shape}"
        output = output.transpose(1, 2)
        assert output.shape == (B, T, self.d_model), f"TCN输出转置后应为(B,T,H)，实际{output.shape}"

        # gate
        x = self.attention_gate(output)
        x = output + original_enriched
        x = self.attention_ln(output)

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        x = self.decoder_gate(x)
        x = x + main_features
        x = self.decoder_ln(x)

        # 三点预测
        x = x[:, -3:, :]  # (B, 3, H)
        out = self.output_layer(x)  # (B, 3, 1)
        assert out.shape == (B, 3, 1), f"最终输出应为(B,3,1)，实际{out.shape}"

        return out
