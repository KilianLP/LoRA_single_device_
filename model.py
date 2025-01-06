#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:47:53 2025

@author: kilianpreuss
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = 1.5
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    n_translation_tokens: int = 1
    max_batch_size: int = 32
    max_seq_len: int = 2048

    alpha: float = 0.01
    r: int = 8


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention_LoRA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.r = args.r

        self.scale_factor = args.alpha/self.r

        self.aq = nn.Parameter(torch.randn(self.r,args.dim))
        self.bq = nn.Parameter(torch.zeros(args.n_heads * self.head_dim,self.r))

        self.ak = nn.Parameter(torch.randn(self.r,args.dim))
        self.bk = nn.Parameter(torch.zeros(self.n_kv_heads * self.head_dim,self.r))

        self.av = nn.Parameter(torch.randn(self.r,args.dim))
        self.bv = nn.Parameter(torch.zeros(self.n_kv_heads * self.head_dim,self.r))

        self.ao = nn.Parameter(torch.randn(self.r,args.dim))
        self.bo = nn.Parameter(torch.zeros(args.n_heads * self.head_dim,self.r))

        #self.__init__weights()

        self.wq = nn.Linear(args.dim,args.n_heads * self.head_dim,bias=False)

        self.wk = nn.Linear(args.dim,self.n_kv_heads * self.head_dim,bias=False)

        self.wv = nn.Linear(args.dim,self.n_kv_heads * self.head_dim,bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim,args.dim,bias=False)

    def __init__weights(self):
        nn.init.kaiming_uniform_(self.aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.av, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.ak, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.ao, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        # torch.bmm(x,(self.bq @ self.aq).T.unsqueeze(0))
        xq = self.wq(x) + self.scale_factor * torch.matmul(x,(self.bq @ self.aq).T)
        xk = self.wk(x) + self.scale_factor * torch.matmul(x,(self.bk @ self.ak).T)
        xv = self.wv(x) + self.scale_factor * torch.matmul(x,(self.bv @ self.av).T)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output) + self.scale_factor * torch.matmul(x,(self.bo @ self.ao).T)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)


    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention_LoRA(args) # Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size + params.n_translation_tokens, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)


        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    #@torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        start_pos = 0
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


    def prepare_lora_gradients(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                print("don't requires_grad:" + name)
                param.requires_grad = False
            else:
                print("requires_grad:" + name)
                param.requires_grad = True


        if self.params.n_translation_tokens != 0:
            self.tok_embeddings.weight.data[self.params.vocab_size:].requires_grad = True


    def load_state_dict_lora(self,state_dict):
        if self.params.n_translation_tokens != 0:
            self.tok_embeddings.weight.data[:self.params.vocab_size] = state_dict["tok_embeddings.weight"]
            state_dict.pop("tok_embeddings.weight")

        self.load_state_dict(state_dict, strict = False)






