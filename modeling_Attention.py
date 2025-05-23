# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# Modified for Bitcoin Price Forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F

if os.environ.get("TFT_SCRIPTING", False):
    from torch.nn import LayerNorm
else:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

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
        self.joint_grn = GRN(config.hidden_size*num_inputs, config.hidden_size, output_size=num_inputs, context_hidden_size=config.hidden_size)
        self.var_grns = nn.ModuleList([GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(num_inputs)])
    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[...,i,:]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)
        # [B,k,d,d_model]->[B,k,H] [B,d,d_model]->[B,H]
        # 金融部分沿用原来的VSN

        return variable_ctx, sparse_weights

class ContinuousEmbedding(nn.Module):   #极简的连续变量嵌入层
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1, 1, hidden_size))  # [1,1,1,H]
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))     # [1,1,1,H]
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入: [B, K, D]
        输出: [B, K, D, H]
        """
        x = x.unsqueeze(-1) 
        return x * self.weight + self.bias #[B,K,D,1] * [1,1,1,H] → [B,K,D,H]


class CovariateEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.var_weights = nn.Linear(config.hidden_size, 1)
        self.k_weights = nn.Linear(config.hidden_size, 1)
        self.context_grns = nn.ModuleList([GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(3)])
        self.ce_grn = GRN(config.hidden_size, config.hidden_size, dropout=config.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        #x：[B,K,N,H]
        weights = self.var_weights(x).squeeze(-1)  #[B,K,N,H] -> [B,K,N,1]
        sparse_weights = F.softmax(weights, dim=-1)  #[B,K,N,1] -> [B,K,N]
        variable_ctx = torch.einsum('bknh,bkn->bkh', x, sparse_weights) #[B,K,N,H] * [B,K,N] -> [B,K,H]
        k_weights = self.k_weights(variable_ctx).squeeze(-1)  # ->[B, K，1]
        sparse_k_weights = F.softmax(k_weights, dim=1)        # [B, K]
        reduced_ctx = torch.einsum('bkh,bk->bh', variable_ctx, sparse_k_weights)  # [B, H]
        cs, ch, cc = tuple(m(reduced_ctx) for m in self.context_grns)
        ce = self.ce_grn(variable_ctx)
        return cs, ce, ch, cc
    
class InterpretableMultiHeadAttention(nn.Module):
   
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        assert config.hidden_size % config.n_head == 0
        self.d_head = config.hidden_size // config.n_head
        self.qkv_linears = nn.Linear(config.hidden_size, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer("_mask", torch.triu(torch.full((config.example_length, config.example_length), float('-inf')), 1).unsqueeze(0))

        
    def forward(self, x: Tensor, mask_future_timesteps: bool = True) -> Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        attn_score = attn_score + self._mask

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # attn_vec = torch.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_vec


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        
        # Embedding层  
        self.conti_embed = ContinuousEmbedding(config.hidden_size)
        
        #self.day_embed = nn.Embedding(7, config.d_model)       # 星期几
        #self.peak_embed = nn.Embedding(2, config.d_model)      # 峰值时段
        #self.fin_embed = nn.Linear(17, config.d_model)         # 金融特征
        #self.med_embed = nn.Linear(6, config.d_model)          # 媒体特征
        #self.mkt_embed = nn.Linear(1, config.d_model)          # 市场情绪
        
        #情绪编码
        self.senti_encoder = CovariateEncoder(config)
        #金融编码
        self.finVSN = VSN(config, config.fin_varible_num)
        self.tem_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        
        #ce变换
        self.ce_encoder = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size, bidirectional = False, batch_first=True)
        self.enrichment_grn = Modified_GRN(config.hidden_size, config.hidden_size, context_hidden_size=config.hidden_size, dropout=config.dropout)
        self.input_gate = GLU(config.hidden_size, config.hidden_size)
        self.input_gate_ln = LayerNorm(config.hidden_size, eps=1e-3)
        # 时序编码器
        #self.lstm = nn.LSTM(config.d_model, config.d_model, num_layers=3)
        self.attention = InterpretableMultiHeadAttention(config)
        #self.attention = nn.MultiheadAttention(config.d_model, num_heads=4)
        self.attention_gate = GLU(config.hidden_size, config.hidden_size)
        self.attention_ln = LayerNorm(config.hidden_size, eps=1e-3)
        self.positionwise_grn = GRN(config.hidden_size,
                                    config.hidden_size,
                                    dropout=config.dropout)
        
        self.decoder_gate = GLU(config.hidden_size, config.hidden_size)
        self.decoder_ln = LayerNorm(config.hidden_size, eps=1e-3)
        
        
        # 输出层
        #self.output_layer = nn.Linear(config.d_model, 1)       # 单点输出
        #self.output_layer = nn.Linear(config.hidden_size, 1)  # 单点输出
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 3 * config.hidden_size),  
            nn.ReLU(),
            nn.Linear(3 * config.hidden_size, 1)                  
        )  # 输出3个点

    def forward(self, x):
        
        # 输入分解 [B, T, 20]
        finance = x[:, :, :13]          # 金融13维
        senti = x[:, :, 13:]          # 媒体+市场7维
        #time_feat = x[:, :, -3].long()  # 星期几（假设为最后第3列）
        #is_peak = (media[:, :, 0] > 1000).long()  # 峰值判断
        
        # Embedding处理
  
        #fin_emb = self.fin_embed(finance)
        #med_emb = self.med_embed(media)
        #mkt_emb = self.mkt_embed(market)
        #day_emb = self.day_embed(time_feat) #后两个可以先不加入，跑通之后再说？
        #peak_emb = self.peak_embed(is_peak)

        fin_inp = self.conti_embed(finance)
        senti_inp = self.conti_embed(senti)
        
        # 特征合并
        #combined = fin_emb + med_emb + mkt_emb + day_emb + peak_emb  # [B, T, d_model]
        
        '''
        嵌入后的数据：
        senti_inp: [B，k=1000，d=13, H=d_model]
        fin_inp: [B，k=1000，d=7,H=d_model]
        '''
        
        #嵌入后部分
        #情绪编码
        cs, ce, ch, cc = self.senti_encoder(senti_inp)
        '''
        cs/ch/cc: [B，H=hidden_size]
        ce: [B，k=1000, H=hidden_size]
        '''
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0) 
        ce , _ = self.ce_encoder(ce)              #一层LSTM

        #金融编码
        fin_features , _ = self.finVSN(fin_inp,cs) 
        '''
        VSN：[B,k,d,d_model]->[B,k,H]
        '''
        fin, state = self.tem_encoder(fin_features, (ch, cc)) #LSTM，维度不变
        main_features = fin + self.input_gate(fin_features)  #skip_connection
        main_features = self.input_gate_ln(main_features)

        #融合
        enriched = self.enrichment_grn(main_features, c=ce)
        
        #把enriched输入Attention
        output, _ = self.attention(enriched, mask_future_timesteps=True )
    
        #截取decoder时段的数据
        #output = output[:, self.encoder_length:, :]
        #temporal_features = temporal_features[:, self.encoder_length:, :]
        #enriched = enriched[:, self.encoder_length:, :]
        
        #gate
        x = self.attention_gate(output)
        x = output + enriched
        x = self.attention_ln(output)

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        x = self.decoder_gate(x)
        x = x + main_features
        x = self.decoder_ln(x)

        
        #output = self.output_layer(output[-1])              
        #return output.squeeze(1)             
        # 三点预测
        x = x[:, -3:, :]  # (B, 3, H)
        out = self.output_layer(x)  # (B, 3, 1)
        return out


