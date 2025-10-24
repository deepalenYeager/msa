'''
* @name: almt.py  (MoME-SAGE with Memory Unit)
* @description: MoME-SAGE using a 1D Affective Memory Unit (AMU) inspired by the provided reference.
'''

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from ..subNets import BertTextEncoder
from einops import rearrange, repeat


# =========================
# Embedding (kept lightweight)
# =========================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    Lightweight encoder used only for early embedding (not the core of the model).
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        inner = heads * dim_head
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, inner, bias=False),
                    nn.GELU(),
                    nn.Linear(inner, dim, bias=False),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = [x]
        for ln1, mixer, ln2, ff in self.layers:
            x = x + mixer(ln1(x))
            x = x + ff(ln2(x))
            if save_hidden:
                hidden_list.append(x)
        return hidden_list if save_hidden else x

class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)
        return x


# =========================
# Affective Memory Unit (1D)
# =========================
class AffectiveMemoryUnit1D(nn.Module):
    """
    A 1D sequence variant of the provided External_attention block.

    Input shape:  (B, N, d)
    Inner shape:  (B, d, N) for Conv1d
    Steps:
      x_in = LN(x)
      x_in -> Conv1d(d->d, k=1)   # optional pre-proj (conv1 in reference)
      attn = Conv1d(d->Kslots, k=1, bias=False)
      attn = softmax(attn, dim=-1)         # over positions N
      attn = attn / (eps + sum(attn, dim=1, keepdim=True))  # L1 over slots
      out  = Conv1d(Kslots->d, k=1, bias=False)             # weight-tied w.r.t attn layer
      out  -> Conv1d(d->d, k=1, bias=False) + LayerNorm(d)  # conv2+norm
      y = ReLU(out + residual)
    """
    def __init__(self, d, slots=64, preproj=True, use_ln=True, dropout=0.0):
        super().__init__()
        self.d = d
        self.k = slots
        self.use_ln = use_ln
        self.preproj = preproj

        self.ln = nn.LayerNorm(d)

        # conv1: d->d (optional, mirrors reference conv2d 1x1 before attention)
        self.conv_in = nn.Conv1d(d, d, kernel_size=1, bias=True) if preproj else nn.Identity()

        # attention convs
        self.linear_0 = nn.Conv1d(d, self.k, kernel_size=1, bias=False)  # c->k
        self.linear_1 = nn.Conv1d(self.k, d, kernel_size=1, bias=False)  # k->c (tied)

        # tie weights: linear_1.weight = linear_0.weight.T (permute in conv1d sense)
        with torch.no_grad():
            self.linear_1.weight.copy_(self.linear_0.weight.permute(1,0,2))

        # conv2 + norm
        self.conv_out = nn.Conv1d(d, d, kernel_size=1, bias=False)
        self.out_ln = nn.LayerNorm(d) if use_ln else nn.Identity()

        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # He init like reference
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: (B, N, d)
        idn = x
        x = self.ln(x)                         # pre-norm for stability

        x = x.transpose(1, 2)                  # (B, d, N)

        if isinstance(self.conv_in, nn.Conv1d):
            x = self.conv_in(x)                # (B, d, N)

        attn = self.linear_0(x)                # (B, k, N)
        attn = F.softmax(attn, dim=-1)         # softmax over positions
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # L1 across slots

        x = self.linear_1(attn)                # (B, d, N)
        x = self.conv_out(x)                   # (B, d, N)

        x = x.transpose(1, 2)                  # (B, N, d)
        if self.use_ln:
            x = self.out_ln(x)
        x = self.drop(x)

        x = x + idn
        x = F.relu(x)
        return x


# =========================
# Dynamic Memory Router (for conditioned K/V)
# =========================
class DynamicMemoryRouter(nn.Module):
    """
    Same math as AMU but with runtime-supplied K/V (cannot tie weights).
    Input:  F (B,N,d), Mk/Mv (H,S,dh) with d=H*dh
    """
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh = d, heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.conv_out = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d, d),
            nn.Dropout(dropout),
        )

    def forward(self, F, Mk, Mv):
        # F: (B,N,d); Mk/Mv: (H,S,dh)
        B, N, D = F.shape
        H, S, dh = Mk.shape
        x = self.ln(F).view(B, N, H, dh).transpose(1, 2)  # (B,H,N,dh)

        # routing
        R = torch.einsum('bhnd,hsd->bhns', x, Mk)         # (B,H,N,S)
        R = F.softmax(R, dim=2)                           # softmax over N (positions)
        R = R / (1e-9 + R.sum(dim=3, keepdim=True))       # L1 over slots S

        O = torch.einsum('bhns,hsd->bhnd', R, Mv)         # (B,H,N,dh)
        O = O.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,d)

        y = F + self.conv_out(O)
        y = y + self.ffn(y)
        return y


# =========================
# MoME-SAGE building blocks
# =========================
class MiME(nn.Module):
    """
    Modality-internal Memory Encoder (single layer), using AMU.
    """
    def __init__(self, d, slots=64, dropout=0.0):
        super().__init__()
        self.amu = AffectiveMemoryUnit1D(d=d, slots=slots, preproj=True, use_ln=True, dropout=dropout)

    def forward(self, F):
        return self.amu(F)  # (B,N,d)

class CMR(nn.Module):
    """
    Cross-Modal Memory Routing:
      - Build conditioned K/V from other modality contexts
      - Route each target modality with dynamic K/V (DynamicMemoryRouter)
    """
    def __init__(self, d, heads, Sx, dropout=0.0):
        super().__init__()
        self.d, self.h, self.dh, self.Sx = d, heads, d // heads, Sx

        # shared base (not directly used as params; we generate K/V dynamically)
        self.ctx_T = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d), nn.GELU())
        self.ctx_A = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d), nn.GELU())
        self.ctx_V = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d), nn.GELU())

        # context -> K/V generator
        self.gen_k_T = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_v_T = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_k_A = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_v_A = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_k_V = nn.Linear(d, heads * Sx * self.dh, bias=False)
        self.gen_v_V = nn.Linear(d, heads * Sx * self.dh, bias=False)

        # mixing factors (shared over batch)
        self.alpha_T = nn.Parameter(torch.tensor(0.5))
        self.alpha_A = nn.Parameter(torch.tensor(0.5))
        self.alpha_V = nn.Parameter(torch.tensor(0.5))

        # global anchors (optional prior)
        self.MkX = nn.Parameter(torch.randn(heads, Sx, self.dh) / math.sqrt(self.dh))
        self.MvX = nn.Parameter(torch.randn(heads, Sx, self.dh) / math.sqrt(self.dh))

        self.router_T = DynamicMemoryRouter(d, heads, dropout=dropout)
        self.router_A = DynamicMemoryRouter(d, heads, dropout=dropout)
        self.router_V = DynamicMemoryRouter(d, heads, dropout=dropout)

    def _kv_from_ctx(self, ctx_vec, gen_k, gen_v, alpha):
        # ctx_vec: (B,d) -> (H,Sx,dh), mix with global MkX/MvX
        B, d = ctx_vec.shape
        H, Sx, dh = self.h, self.Sx, self.dh
        Kc = gen_k(ctx_vec).view(B, H, Sx, dh).mean(dim=0)  # average over B for stability
        Vc = gen_v(ctx_vec).view(B, H, Sx, dh).mean(dim=0)
        a = torch.clamp(alpha, 0., 1.)
        Mk = a * self.MkX + (1 - a) * Kc
        Mv = a * self.MvX + (1 - a) * Vc
        return Mk, Mv

    def forward(self, HT, HA, HV):
        # global summaries of each stream
        t = HT.mean(dim=1); a = HA.mean(dim=1); v = HV.mean(dim=1)

        # contexts for each target
        c_T = self.ctx_T(torch.cat([a, v], dim=-1))   # condition T on (A,V)
        c_A = self.ctx_A(torch.cat([t, v], dim=-1))   # condition A on (T,V)
        c_V = self.ctx_V(torch.cat([t, a], dim=-1))   # condition V on (T,A)

        # conditioned K/V
        MkT, MvT = self._kv_from_ctx(c_T, self.gen_k_T, self.gen_v_T, self.alpha_T)
        MkA, MvA = self._kv_from_ctx(c_A, self.gen_k_A, self.gen_v_A, self.alpha_A)
        MkV, MvV = self._kv_from_ctx(c_V, self.gen_k_V, self.gen_v_V, self.alpha_V)

        # route in memory space
        T_aln = self.router_T(HT, MkT, MvT)
        A_aln = self.router_A(HA, MkA, MvA)
        V_aln = self.router_V(HV, MkV, MvV)
        return T_aln, A_aln, V_aln

class MGF(nn.Module):
    """
    Memory-Guided Fusion:
      - Use AMU on pooled single-step tokens (B,1,d) to produce gates
      - Additive fusion without expanding dimension
    """
    def __init__(self, d, gate_slots=32, dropout=0.0):
        super().__init__()
        self.amu_gate = AffectiveMemoryUnit1D(d=d, slots=gate_slots, preproj=True, use_ln=True, dropout=dropout)
        self.Wg_T = nn.Linear(d, d, bias=True)
        self.Wg_A = nn.Linear(d, d, bias=True)
        self.Wg_V = nn.Linear(d, d, bias=True)

    def forward(self, T_aln, A_aln, V_aln):
        t = T_aln.mean(dim=1).unsqueeze(1)  # (B,1,d)
        a = A_aln.mean(dim=1).unsqueeze(1)
        v = V_aln.mean(dim=1).unsqueeze(1)

        # produce gates via AMU (sequence len = 1 is allowed)
        gT = torch.sigmoid(self.Wg_T(self.amu_gate(t).squeeze(1)))  # (B,d)
        gA = torch.sigmoid(self.Wg_A(self.amu_gate(a).squeeze(1)))
        gV = torch.sigmoid(self.Wg_V(self.amu_gate(v).squeeze(1)))

        F = gT * t.squeeze(1) + gA * a.squeeze(1) + gV * v.squeeze(1)
        return F  # (B,d)


# =========================
# MoME-SAGE main network
# =========================
class MoME_SAGE(nn.Module):
    def __init__(self, args):
        super(MoME_SAGE, self).__init__()

        # ---- Backbones ----
        if args.use_bert:
            self.bertmodel = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        self.use_bert = args.use_bert

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.orig_length_l, self.orig_length_a, self.orig_length_v = args.feature_length

        self.dst_embedding_d_l = self.dst_embedding_d_a = self.dst_embedding_d_v = args.dst_feature_dims
        self.dst_embedding_hidden_d_l = self.dst_embedding_hidden_d_a = self.dst_embedding_hidden_d_v = args.dst_feature_hidden_dims
        self.dst_embedding_length_l = self.dst_embedding_length_a = self.dst_embedding_length_v = args.dst_embedding_length
        self.embedding_depth_l, self.embedding_depth_a, self.embedding_depth_v = args.embedding_depth
        self.embedding_heads_l, self.embedding_heads_a, self.embedding_heads_v = args.embedding_heads

        # ---- New hyper-params (with defaults) ----
        self.d = args.dst_feature_dims
        self.memory_slots_l = getattr(args, "mam_slots_l", 64)
        self.memory_slots_a = getattr(args, "mam_slots_a", 64)
        self.memory_slots_v = getattr(args, "mam_slots_v", 64)
        self.cross_heads   = getattr(args, "memory_heads", 4)   # for CMR only
        self.cross_slots   = getattr(args, "cross_slots", 32)
        self.gate_slots    = getattr(args, "gate_slots", 32)
        self.dropout       = getattr(args, "dropout", 0.0)

        # ---- Embedding ----
        self.embedding_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.dst_embedding_d_l),
            Transformer(num_frames=self.orig_length_l,
                        save_hidden=False,
                        token_len=self.dst_embedding_length_l,
                        dim=self.dst_embedding_d_l,
                        depth=self.embedding_depth_l,
                        heads=self.embedding_heads_l,
                        mlp_dim=self.dst_embedding_hidden_d_l)
        )
        self.embedding_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.dst_embedding_d_a),
            Transformer(num_frames=self.orig_length_a,
                        save_hidden=False,
                        token_len=self.dst_embedding_length_a,
                        dim=self.dst_embedding_d_a,
                        depth=self.embedding_depth_a,
                        heads=self.embedding_heads_a,
                        mlp_dim=self.dst_embedding_hidden_d_a)
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.dst_embedding_d_v),
            Transformer(num_frames=self.orig_length_v,
                        save_hidden=False,
                        token_len=self.dst_embedding_length_v,
                        dim=self.dst_embedding_d_v,
                        depth=self.embedding_depth_v,
                        heads=self.embedding_heads_v,
                        mlp_dim=self.dst_embedding_hidden_d_v)
        )

        # ---- Modality-specific memory encoders (AMU) ----
        self.mime_l = MiME(self.d, slots=self.memory_slots_l, dropout=self.dropout)
        self.mime_a = MiME(self.d, slots=self.memory_slots_a, dropout=self.dropout)
        self.mime_v = MiME(self.d, slots=self.memory_slots_v, dropout=self.dropout)

        # ---- Cross-modal memory routing (Dynamic) ----
        self.cmr = CMR(self.d, heads=self.cross_heads, Sx=self.cross_slots, dropout=self.dropout)

        # ---- Memory-guided fusion ----
        self.mgf = MGF(self.d, gate_slots=self.gate_slots, dropout=self.dropout)

        # ---- Regression head ----
        self.regression_head = nn.Linear(self.d, 1)

    def forward(self, text, audio, video):
        b = video.size(0)

        # Backbone text encoding
        if self.use_bert:
            x_text = self.bertmodel(text)
        else:
            x_text = text

        # Lightweight embeddings (keep [:, :8] as in original code)
        h_l = self.embedding_l(x_text)[:, :8]  # (B, Nl, d)
        h_a = self.embedding_a(audio)[:, :8]   # (B, Na, d)
        h_v = self.embedding_v(video)[:, :8]   # (B, Nv, d)

        # Modality-specific memory encoding
        H_l = self.mime_l(h_l)
        H_a = self.mime_a(h_a)
        H_v = self.mime_v(h_v)

        # Cross-modal memory routing (alignment in memory space)
        T_aln, A_aln, V_aln = self.cmr(H_l, H_a, H_v)

        # Memory-guided fusion (additive, no 2d concat)
        feat = self.mgf(T_aln, A_aln, V_aln)   # (B,d)

        # Regression
        output = self.regression_head(feat)    # (B,1)
        return output
