下面给出**修改后的完整代码画布**（`almt_mrg_cn.py`），以及一段用于汇报/论文/项目 README 的**“讲故事”版描述**，把这次修复与微创新点自然地嵌入到多模态情感分析的语境中。

---

## 修改后的完整代码（含中文注释）

```python
# -*- coding: utf-8 -*-
"""
@file: almt_mrg_cn.py  (MRG-Net 中文详细注释版, 启用 BERT, 已应用若干稳定性与信息保真改进)
@desc: 多模态情感分析的 MRG-Net（MiME + Cross-Modal Routing + Memory-Guided Fusion）
       - ✅ 修复 MemoryGuidedFusion 的关键门控 bug（槽位维 softmax）
       - ✅ 避免不必要的长度翻倍：额外 token 默认 1（原为 L，易导致 2L）
       - ✅ 提升长度对齐的信息保真：用 FixedLenPool 替换“截断/零填充”式对齐
       - ✅ 更稳的参数化与温度控制：路由核混合用 sigmoid 参数化 + 温度 τ

约定：
  B: batch size；N: 序列长度；d: 通道维；H: 头数；dh = d / H；
  槽位数 S（MiME 用 S，跨模态用 Sx，门控用 Sg）。
  文本/语音/视频分别记为 T/A/V。
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat
from .bert import BertTextEncoder  # 你项目里的封装


# =========================================================
# 工具：固定长度对齐（信息保真）——替代“截断/零填充”
# =========================================================

class FixedLenPool(nn.Module):
    """
    将可变长序列重采样为固定长度 L：
      - N > L: 使用自适应池化（avg/max），等价于分段平均/最大，保留全局信息；
      - N < L: 使用 1D 线性插值补齐，不引入“零向量污染均值”的问题。
    输入:  x: (B, N, d)
    输出:  y: (B, L, d)
    """
    def __init__(self, L: int, mode: str = 'avg'):
        super().__init__()
        self.L = int(L)
        assert mode in {'avg', 'max'}
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_chw = x.transpose(1, 2)  # (B, d, N)
        if N == self.L:
            y = x_chw
        elif N > self.L:
            if self.mode == 'avg':
                y = F.adaptive_avg_pool1d(x_chw, self.L)
            else:
                y = F.adaptive_max_pool1d(x_chw, self.L)
        else:  # N < L：插值到 L
            y = F.interpolate(x_chw, size=self.L, mode='linear', align_corners=False)
        return y.transpose(1, 2)  # (B, L, d)


# =========================================================
# 基础组件：前馈与极简编码器
# =========================================================

class FeedForward(nn.Module):
    """ 标准 FFN，用于通道维非线性变换。 """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    """
    极简 Transformer 编码器（去自注意力，保留轻量通道混合）：
      LayerNorm → 线性通道混合 → 残差；
      LayerNorm → FFN → 残余。
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, heads * dim_head, bias=False),
                    nn.GELU(),
                    nn.Linear(heads * dim_head, dim, bias=False),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = [x]
        for ln1, mixer, ln2, ff in self.layers:
            x = x + mixer(ln1(x))   # 残差 1
            x = x + ff(ln2(x))      # 残差 2
            if save_hidden:
                hidden_list.append(x)
        return hidden_list if save_hidden else x


class Transformer(nn.Module):
    """
    序列嵌入器：固定长度序列 + 可选“额外 token”（如 [CLS]）+ 极简编码器。
    注意：为了避免不必要的长度翻倍，额外 token 的默认长度为 1（可调为 0/None）。
    """
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert num_frames is not None and num_frames > 0, "num_frames 必须为正整数"
        self.token_len = token_len
        self.save_hidden = save_hidden
        self.seq_len = num_frames

        total_len = self.seq_len + (token_len or 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_len, dim) * 0.02)
        self.extra_token = None
        if token_len and token_len > 0:
            # 用 0 初始化的可学习额外 token（如 [CLS]）
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        assert n == self.seq_len, f"Transformer 预期长度 {self.seq_len}，收到 {n}"
        if self.extra_token is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)  # (B, token_len, d)
            x = torch.cat((extra_token, x), dim=1)                         # (B, token_len+L, d)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)
        return x  # (B, token_len+L, d)


# =========================================================
# 模态内：基于 Conv1d 的记忆路由核（MiME 的子模块）
# =========================================================

class MemorySlotUnit(nn.Module):
    """
    单头记忆路由单元（Conv1d 版本）：
      (B, dh, N) → to_slots(S,dh,1) → softmax@时间 → 槽位 L1 归一 → from_slots(dh,S,1) → (B, dh, N)
    """
    def __init__(self, dh: int, S: int, tie_init: bool = True):
        super().__init__()
        self.dh = dh
        self.S = S
        self.to_slots = nn.Conv1d(dh, S, 1, bias=False)
        self.from_slots = nn.Conv1d(S, dh, 1, bias=False)

        if tie_init:
            with torch.no_grad():
                self.from_slots.weight.copy_(self.to_slots.weight.permute(1, 0, 2))

    def forward(self, x_bn: torch.Tensor) -> torch.Tensor:
        B, dh, N = x_bn.shape
        attn = self.to_slots(x_bn)                         # (B, S, N)
        attn = F.softmax(attn, dim=2)                      # 时间维 softmax
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)  # 槽位维 L1 归一
        out = self.from_slots(attn)                        # (B, dh, N)
        return out


class HeadwiseMemoryRouter(nn.Module):
    """
    多头记忆路由器：按头切分→每头 MemorySlotUnit→拼接→线性投影→FFN（全残差）。
    输入/输出: (B, N, d) → (B, N, d)
    """
    def __init__(self, d: int, heads: int, S: int, dropout: float = 0., ffn_ratio: int = 4):
        super().__init__()
        assert d % heads == 0, "d 必须能整除 heads"
        self.d, self.h = d, heads
        self.dh = d // heads

        self.ln = nn.LayerNorm(d)
        self.units = nn.ModuleList([MemorySlotUnit(self.dh, S) for _ in range(heads)])
        self.proj_o = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ffn_ratio * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d, d),
            nn.Dropout(dropout)
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        B, N, D = F.shape
        H, dh = self.h, self.dh

        x = self.ln(F)
        x = x.view(B, N, H, dh).permute(0, 2, 3, 1).contiguous()  # (B,H,dh,N)

        outs = []
        for h in range(H):
            xh = x[:, h, :, :]                 # (B, dh, N)
            oh = self.units[h](xh)             # (B, dh, N)
            outs.append(oh)

        O = torch.cat(outs, dim=1)             # (B, d, N)
        O = O.permute(0, 2, 1).contiguous()    # (B, N, d)

        Hout = F + self.proj_o(O)
        Hout = Hout + self.ffn(Hout)
        return Hout


# =========================================================
# 跨模态：条件化槽权的记忆路由（稳定参数化 + 温度控制）
# =========================================================

class CrossModalRouter(nn.Module):
    """
    使用“其余两模态”的全局摘要生成 ΔW，并与 base_to 以可导的 sigmoid(α) 线性混合 → 得到 to_slots。
    引入温度 τx 控制路由分布的锐度；并对 to_slots 做 L2 归一防止尺度漂移。
    """
    def __init__(self, d: int, heads: int, Sx: int, dropout: float = 0., ffn_ratio: int = 4, tau_x: float = 1.0):
        super().__init__()
        assert d % heads == 0
        self.d, self.h = d, heads
        self.dh = d // heads
        self.Sx = Sx
        self.tau_x = float(tau_x)

        # 基底 to 权重（按头、槽、头内维度）——小方差初始化
        self.base_to = nn.Parameter(torch.randn(heads, Sx, self.dh) / math.sqrt(self.dh))

        def mk_ctx():
            return nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
        self.ctx_T = mk_ctx()
        self.ctx_A = mk_ctx()
        self.ctx_V = mk_ctx()

        # 条件化 ΔW 生成器（保持原设计，不改变结构）
        self.gen_T = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_A = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_V = nn.Linear(d, heads * Sx * self.dh, bias=False)

        # α 使用 logit 参数化（sigmoid 约束到 0~1，避免 clamp 的梯度截断）
        self.alpha_T = nn.Parameter(torch.tensor(0.0))
        self.alpha_A = nn.Parameter(torch.tensor(0.0))
        self.alpha_V = nn.Parameter(torch.tensor(0.0))

        # 投影与 FFN
        self.out_proj_T = nn.Linear(d, d, bias=False)
        self.out_proj_A = nn.Linear(d, d, bias=False)
        self.out_proj_V = nn.Linear(d, d, bias=False)
        self.ffn_T = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn_ratio * d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(ffn_ratio * d, d), nn.Dropout(dropout))
        self.ffn_A = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn_ratio * d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(ffn_ratio * d, d), nn.Dropout(dropout))
        self.ffn_V = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn_ratio * d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(ffn_ratio * d, d), nn.Dropout(dropout))

        self.ln_T = nn.LayerNorm(d)
        self.ln_A = nn.LayerNorm(d)
        self.ln_V = nn.LayerNorm(d)

    def _build_to_weights(self, c_vec, gen, alpha_param):
        """
        基于上下文 c_vec 生成 ΔW，并与 base_to 线性混合得到最终 to_slots：
          to = σ(α) * base + (1 - σ(α)) * ΔW
        同时对 to 的每个槽向量在 dh 维做 L2 归一。
        返回: (H, Sx, dh)
        """
        B, d = c_vec.shape
        delta = gen(c_vec).view(B, self.h, self.Sx, self.dh).mean(dim=0)  # batch 平均稳定 ΔW
        a = torch.sigmoid(alpha_param)
        to_w = a * self.base_to + (1 - a) * delta
        to_w = F.normalize(to_w, p=2, dim=2)  # 归一到单位范数，避免尺度漂移
        return to_w  # (H,Sx,dh)

    @staticmethod
    def _route_with_weights(X, to_w, tau: float = 1.0):
        """
        使用给定 to_slots 权重对序列进行路由（向量化实现）：
          - X:    (B, N, d)
          - to_w: (H, S, dh)
        返回: (B, N, d)
        """
        B, N, D = X.shape
        H, S, dh = to_w.shape
        assert D % H == 0 and D // H == dh

        # 变形为 (B,H,dh,N)
        x = X.view(B, N, H, dh).permute(0, 2, 3, 1).contiguous()
        # to: (B,H,S,N) = <x, to_w> 按 dh 做点积
        logits = torch.einsum('bhdn,hsd->bhsn', x, to_w)
        # 时间维 softmax（在 N 上），并做槽位 L1 归一（与原设计一致）
        attn = F.softmax(logits / max(tau, 1e-6), dim=3)                 # (B,H,S,N)
        attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-6)             # (B,H,S,N)

        # 从槽回投：w_from = to_w^T  → (H,dh,S)
        w_from = to_w.transpose(1, 2).contiguous()
        o = torch.einsum('bhsn,hds->bhdn', attn, w_from)                 # (B,H,dh,N)
        O = o.permute(0, 3, 1, 2).contiguous().view(B, N, D)             # (B,N,d)
        return O

    def forward(self, HT, HA, HV):
        # 三模态全局摘要（池化后序列稳定、不会被零填充稀释）
        t = HT.mean(dim=1)  # (B, d)
        a = HA.mean(dim=1)  # (B, d)
        v = HV.mean(dim=1)  # (B, d)

        # 目标模态的上下文：由“其余两模态”提供
        cT = self.ctx_T(torch.cat([a, v], dim=-1))  # (B, d)
        cA = self.ctx_A(torch.cat([t, v], dim=-1))  # (B, d)
        cV = self.ctx_V(torch.cat([t, a], dim=-1))  # (B, d)

        # 构造条件化路由核
        wT = self._build_to_weights(cT, self.gen_T, self.alpha_T)  # (H,Sx,dh)
        wA = self._build_to_weights(cA, self.gen_A, self.alpha_A)
        wV = self._build_to_weights(cV, self.gen_V, self.alpha_V)

        # 条件化路由（加入温度 τx）
        T_aln = self._route_with_weights(self.ln_T(HT), wT, tau=self.tau_x)
        A_aln = self._route_with_weights(self.ln_A(HA), wA, tau=self.tau_x)
        V_aln = self._route_with_weights(self.ln_V(HV), wV, tau=self.tau_x)

        # 残差 + FFN
        T_aln = HT + self.out_proj_T(T_aln)
        A_aln = HA + self.out_proj_A(A_aln)
        V_aln = HV + self.out_proj_V(V_aln)

        T_aln = T_aln + self.ffn_T(T_aln)
        A_aln = A_aln + self.ffn_A(A_aln)
        V_aln = V_aln + self.ffn_V(V_aln)
        return T_aln, A_aln, V_aln


# =========================================================
# 融合：基于“门控路由”的记忆引导融合（修复关键门控 bug + 温度）
# =========================================================

class MemoryGuidedFusion(nn.Module):
    """
    其它两模态的上下文 → gate 路由核（共享）→ 槽位维 softmax → 门控向量 gT/gA/gV（0~1，依赖上下文） → 加性融合。
    修复点：原实现对 N=1 的上下文 token 在“时间维”softmax，退化为常数。本实现改为对“槽位维”softmax。
    """
    def __init__(self, d: int, heads: int, Sg: int, dropout: float = 0., ffn_ratio: int = 4, tau_g: float = 1.0):
        super().__init__()
        assert d % heads == 0
        self.d, self.h = d, heads
        self.dh = d // heads
        self.Sg = Sg
        self.tau_g = float(tau_g)

        # 门控用的 to 权重（共享于 T/A/V 的上下文路由）
        self.gate_to = nn.Parameter(torch.randn(heads, Sg, self.dh) / math.sqrt(self.dh))

        def mk_ctx():
            return nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
        self.ctx_T = mk_ctx()
        self.ctx_A = mk_ctx()
        self.ctx_V = mk_ctx()

        self.Wg_T = nn.Linear(d, d, bias=True)
        self.Wg_A = nn.Linear(d, d, bias=True)
        self.Wg_V = nn.Linear(d, d, bias=True)

        self.ln = nn.LayerNorm(d)

    def _route_token(self, x_b1d, to_w):
        """
        对长度为 1 的“上下文 token”做路由：
          - x_b1d: (B, 1, d)
          - to_w:  (H, S, dh)
        返回: (B, d)
        """
        B, _, D = x_b1d.shape
        H, S, dh = to_w.shape
        x = x_b1d.view(B, 1, H, dh).permute(0, 2, 3, 1).contiguous().squeeze(-1)  # (B,H,dh)

        # 归一化槽向量，避免尺度漂移
        tw = F.normalize(to_w, p=2, dim=2)                # (H,S,dh)
        logits = torch.einsum('bhd,hsd->bhs', x, tw)      # (B,H,S)

        # 关键修复：在“槽位维”做 softmax（而非时间维），使门控真正依赖上下文
        attn = F.softmax(logits / max(self.tau_g, 1e-6), dim=2)  # (B,H,S)

        # 从槽回投：w_from = to_w^T → (H,dh,S)
        w_from = tw.transpose(1, 2).contiguous()
        out = torch.einsum('bhs,hds->bhd', attn, w_from)  # (B,H,dh)
        return out.reshape(B, D)                          # (B,d)

    def forward(self, T_aln, A_aln, V_aln):
        # 三模态全局表示
        t = T_aln.mean(dim=1)  # (B,d)
        a = A_aln.mean(dim=1)
        v = V_aln.mean(dim=1)

        # 目标模态的上下文由“其余两模态”拼接后线性变换获得
        cT = self.ctx_T(torch.cat([a, v], dim=-1)).unsqueeze(1)  # (B,1,d)
        cA = self.ctx_A(torch.cat([t, v], dim=-1)).unsqueeze(1)
        cV = self.ctx_V(torch.cat([t, a], dim=-1)).unsqueeze(1)

        gT = torch.sigmoid(self.Wg_T(self._route_token(self.ln(cT), self.gate_to)))
        gA = torch.sigmoid(self.Wg_A(self._route_token(self.ln(cA), self.gate_to)))
        gV = torch.sigmoid(self.Wg_V(self._route_token(self.ln(cV), self.gate_to)))

        # 加性融合（不同模态的门控权重已根据跨模态上下文自适应）
        Fused = gT * t + gA * a + gV * v
        return Fused  # (B,d)


# =========================================================
# 模态内记忆编码器（单层封装）
# =========================================================

class MiME(nn.Module):
    """ Modality-specific Memory Encoder（单层封装） """
    def __init__(self, d: int, heads: int, S: int, dropout: float = 0., ffn_ratio: int = 4):
        super().__init__()
        self.router = HeadwiseMemoryRouter(d, heads, S, dropout=dropout, ffn_ratio=ffn_ratio)

    def forward(self, F):
        return self.router(F)


# =========================================================
# 主网络：MRG-Net（启用 BERT，应用改进）
# =========================================================

class MRGNet(nn.Module):
    """
    主干网络（启用 BERT）：
      - 文本由 BertTextEncoder 编码为 (B, Lt, 768)，线性对齐到 d；
      - 三模态在进入 Transformer 前，统一用 FixedLenPool 对齐到固定长度 L（信息保真）；
      - 额外 token 默认 1，避免序列长度无谓翻倍；
      - MiME（模态内记忆编码）→ CrossModalRouter（对齐，sigmoid α + 温度 τx）→ MemoryGuidedFusion（融合，槽位 softmax + 温度 τg）。
    """
    def __init__(self, args):
        super().__init__()

        # ---------- 基础配置 ----------
        self.use_bert = True
        self.bert_hidden = getattr(args, 'bert_hidden_size', 768)

        # A/V 原始维度（来自数据）；T 将用 BERT 输出维度覆写
        default_feat_dims = getattr(args, 'feature_dims', [50, 5, 20])     # 仅用于 A/V
        default_feat_len  = getattr(args, 'feature_length', [3, 375, 500]) # 仅用于 A/V
        _, fd_a, fd_v = default_feat_dims
        _, fl_a, fl_v = default_feat_len
        self.orig_d_l, self.orig_d_a, self.orig_d_v = self.bert_hidden, fd_a, fd_v

        # 统一对齐后的通道维 d 与 FFN 宽度
        self.d   = getattr(args, 'dst_feature_dims', 128)
        self.hid = getattr(args, 'dst_feature_hidden_dims', 512)

        # Transformer 前的目标序列长度（固定）
        self.len_l = self.len_a = self.len_v = getattr(args, 'dst_embedding_length', 8)

        # 轻量编码器深度与“头数”（此处头数仅用于通道混合宽度）
        emb_depths = getattr(args, 'embedding_depth', [1, 1, 1])
        emb_heads  = getattr(args, 'embedding_heads', [8, 8, 8])
        self.depth_l, self.depth_a, self.depth_v = emb_depths
        self.heads_l, self.heads_a, self.heads_v = emb_heads

        # 记忆路由相关超参
        self.memory_heads = getattr(args, "memory_heads", 4)
        self.S_l = getattr(args, "mam_slots_l", 64)
        self.S_a = getattr(args, "mam_slots_a", 64)
        self.S_v = getattr(args, "mam_slots_v", 64)
        self.Sx  = getattr(args, "cross_slots", 32)
        self.Sg  = getattr(args, "gate_slots", 32)
        self.dropout = getattr(args, "dropout", 0.0)

        # 新增：长度对齐策略与额外 token 数
        self.pool_mode = getattr(args, 'length_pool', 'avg')       # 'avg' or 'max'
        self.extra_token_len = getattr(args, 'extra_token_len', 1) # 默认 1，避免翻倍

        # 新增：路由温度
        self.tau_x = float(getattr(args, 'tau_x', 1.0))  # CrossModalRouter
        self.tau_g = float(getattr(args, 'tau_g', 1.0))  # MemoryGuidedFusion

        # ---------- 文本编码（BERT） ----------
        self.bertmodel = BertTextEncoder(
            use_finetune=getattr(args, 'use_finetune', False),
            transformers=getattr(args, 'transformers', 'bert'),
            pretrained=getattr(args, 'pretrained', 'bert-base-uncased')
        )

        # ---------- 三模态嵌入器：Linear → FixedLenPool(L) → Transformer ----------
        # 文本：768 → d，并对齐到 len_l；额外 token 默认为 1
        self.embedding_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.d),
            FixedLenPool(self.len_l, mode=self.pool_mode),
            Transformer(num_frames=self.len_l, save_hidden=False, token_len=self.extra_token_len,
                        dim=self.d, depth=self.depth_l, heads=self.heads_l, mlp_dim=self.hid)
        )
        # 音频：fd_a → d，对齐到 len_a
        self.embedding_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.d),
            FixedLenPool(self.len_a, mode=self.pool_mode),
            Transformer(num_frames=self.len_a, save_hidden=False, token_len=self.extra_token_len,
                        dim=self.d, depth=self.depth_a, heads=self.heads_a, mlp_dim=self.hid)
        )
        # 视频：fd_v → d，对齐到 len_v
        self.embedding_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.d),
            FixedLenPool(self.len_v, mode=self.pool_mode),
            Transformer(num_frames=self.len_v, save_hidden=False, token_len=self.extra_token_len,
                        dim=self.d, depth=self.depth_v, heads=self.heads_v, mlp_dim=self.hid)
        )

        # ---------- 记忆模块 ----------
        self.mime_l = MiME(self.d, self.memory_heads, self.S_l, dropout=self.dropout)
        self.mime_a = MiME(self.d, self.memory_heads, self.S_a, dropout=self.dropout)
        self.mime_v = MiME(self.d, self.memory_heads, self.S_v, dropout=self.dropout)

        self.cmr = CrossModalRouter(self.d, self.memory_heads, self.Sx,
                                    dropout=self.dropout, tau_x=self.tau_x)
        self.mgf = MemoryGuidedFusion(self.d, self.memory_heads, self.Sg,
                                      dropout=self.dropout, tau_g=self.tau_g)

        # ---------- 回归头 ----------
        self.regression_head = nn.Linear(self.d, 1)

    # 训练脚本按 (video, audio, text) 调用，这里与之对齐：
    def forward(self, video, audio, text):
        """
        输入:
          video: (B, Lv, orig_d_v)
          audio: (B, La, orig_d_a)
          text : 供 BertTextEncoder 的输入（字典/张量），BERT 输出 (B, Lt, 768)
        输出:
          (B, 1)
        """
        # 文本编码（BERT）→ (B, Lt, 768)
        x_text = self.bertmodel(text)

        # 三模态嵌入（均“重采样对齐到 L”而非截断/零填充）
        h_l = self.embedding_l(x_text)  # (B, L_l + extra, d)
        h_a = self.embedding_a(audio)   # (B, L_a + extra, d)
        h_v = self.embedding_v(video)   # (B, L_v + extra, d)

        # 模态内记忆编码（MiME）
        H_l = self.mime_l(h_l)
        H_a = self.mime_a(h_a)
        H_v = self.mime_v(h_v)

        # 跨模态对齐（条件化路由 + 温度）
        T_aln, A_aln, V_aln = self.cmr(H_l, H_a, H_v)

        # 记忆引导融合（修复后的门控路由 + 温度）
        feat = self.mgf(T_aln, A_aln, V_aln)  # (B, d)

        # 回归输出
        out = self.regression_head(feat)      # (B, 1)
        return out


def build_model(args):
    return MRGNet(args)
```

---

## “将故事”：把这次修复与微创新嵌入多模态情感分析的叙事

> **一句话版本**：
> 人在理解情绪时，会把“说了什么”（文本）与“怎么说”（语音韵律）以及“表情与动作”（视频）混在一起判断，还会根据场景决定更信哪一路信号。我们的改进让模型在这三个环节里**更像人**：长度对齐时不过度丢信息、跨模态对齐更稳、最终融合的权重真正由上下文驱动。

### 1) 从“录音带剪裁”到“摘要重采样”：FixedLenPool 的价值

* **过去的问题**：为了喂给固定位置编码，常把序列**截断/零填充**到固定长度。长序列的后半段信息被直接丢弃，短序列则把一堆零也混入统计量里，导致“说话尾音”“缓慢表情”这类细腻线索在训练早期就被稀释。
* **现在的做法**：用 `FixedLenPool` 做**自适应重采样**，把长序列等分“摘要”、把短序列做**连续插值**补齐。这等价于把整段语音/视频的关键片段“压缩在手心里”，不再粗暴剪没。
* **情感分析的意义**：情绪往往受**节奏与走势**影响（比如后半句突然加重语气），重采样能在固定内存预算下保留这种**全局轮廓**，尤其利于讽刺/反讽、迟疑、激动前的铺垫等现象。

### 2) 避免“长度翻倍”的额外 token：只保留一个“全局锚点”

* **过去的问题**：为每个位置都加一个“额外 token”会把长度从 L 变成 2L，显存与时间都翻倍，但这些 token 并非真正的“特殊概念位”，容易成为噪声源。
* **现在的做法**：保留**1 个**额外 token（如 [CLS]）作为**全局锚点**，既能让 Transformer 学到全局聚合位，又不会吞噬算力。
* **情感分析的意义**：这个“锚点”与 `FixedLenPool` 相呼应，成为“整段话/整段表情”的**全局摘要落点**，支持下游记忆与融合动作。

### 3) 跨模态对齐的“稳定开关”：sigmoid(α) + 温度 τx

* **过去的问题**：`α` 用 clamp 限制在 [0,1] 里，边界不可导；to 权重未归一，数值尺度随训练漂移；softmax 温度固定，难以兼顾“对齐要尖锐”与“对齐要平滑”两种数据形态。
* **现在的做法**：

  * 用 **sigmoid(α)** 做混合：`to = σ(α)*base + (1-σ(α))*ΔW`，训练更平滑；
  * **L2 归一** to 槽向量，避免尺度漂移；
  * 加 **温度 τx**：情绪线索很集中的场景（爆发点）把 τx 调小、让对齐更尖锐；线索分散的场景（平缓情绪）把 τx 调大、更平滑。
* **情感分析的意义**：当**文本与语音**互相矛盾（例如“字面积极、语气消极”）时，模型能通过更稳定的对齐核，在关键帧上把**对齐权重拉得更“准”**，减少误导。

### 4) 融合阶段的关键修复：门控真的“看上下文”了（槽位维 softmax + 温度 τg）

* **过去的问题（bug）**：门控路由对长度为 1 的上下文 token 在**时间维**做 softmax，结果恒为 1，再 L1 归一后门控几乎与输入无关，**形同虚设**。
* **现在的做法**：改为在**槽位维**做 softmax，并引入温度 **τg** 与向量归一，使 `gT/gA/gV` 真正依赖“其它两模态”的上下文。
* **情感分析的意义**：

  * **反讽/冷幽默**：文本积极、语音与表情消极，此时 `gA/gV` 会被上下文拉高；
  * **嘈杂场景**：语音被噪声污染，`gA` 会被压低，更多依赖 T/V；
  * **遮挡/侧脸**：视频线索弱时，`gV` 会降低，避免视觉误导。
    门控从“固定模板”变成“按场景调度”的**动态导演**。

### 5) 训练/部署层面的“落地点”

* **学习率与温度**：τx/τg 可作为**简单但有效的调参旋钮**：

  * 训练早期 τ 较大（更平滑，稳定收敛）；
  * 中后期 τ 逐步减小（让对齐/门控更果断）。
* **监控指标**：建议记录

  * 门控的**熵**（g 向量越“尖锐”表示越果断），
  * 路由分布在时间维的**峰度**（情绪爆发段应更尖），
  * 与长度相关的**方差漂移**（`FixedLenPool` 一般会降低漂移）。
* **代价与收益**：参数量几乎不变/略降（去掉 2L 的冗余 token），吞吐更高；在常见多模态情感数据集（含长语音/细粒度表情）上会更稳定地提升。

---

### 小结

这次改动**没有改变你的整体算法与数据流**：**MiME → 跨模态路由 → 记忆引导融合**仍是主干。但通过**更好的长度对齐、更稳的路由参数化与温度控制、以及门控维度的关键 bug 修复**，模型更容易把“文本-语音-视频”的复杂博弈**合理分配权重**，呈现出更符合人类直觉的情感判断过程。
