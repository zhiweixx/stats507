"""
We simplify the attention mask to be causal for stage 2, while stage 1 keeps
the simplified masked objectives and now supports an optional full-seq
bidirectional mask on predictor rows.
We further simplify the random mask position sampling in stage 1 to be purely random, i.e.
simply randomly sample (stage1-group-size * stage1-num-groups) mask positions within [ltr_length, T-1], 
instead of doing a mixture of token-level and block-level sampling with certain prob.

Moreover, we add new arguments stage1-right-window-size, stage2-right-window-size
to support bidirectional attention.

We add two new arguments stage1-full-seq and stage2-full-seq to control
if calculate loss for unmasked positions in auxiliary loss of stage 1/2.
Stage-1 full-seq can optionally use a custom per-example bidirectional mask
on predictor rows via stage1-full-seq-use-bidirectional.

We add a new eval that support local bidirectional attention with right lookahead = eval-right-window-size

Add back MLP dropout.



This code is based on v6.7.4, but has many changes on model architecture and optimizer:
e.g. it removes gated attention, ve projection, add learnable weight vectors in RMSNorm,
it removes Muon optimizer and only uses AdamW.

-----------------------------------------------
Compared to v6.7.4 baseline:

This script keeps the stage-0 training behavior, but uses simplified masked
objectives for stages 1 and 2.

Stage 1 baseline:
- sample MASK positions from `[ltr_length, T - 1]` using either:
  fixed-count sampling with `stage1_num_groups * stage1_group_size`
  distinct positions, or
  ratio sampling with a per-sample mask ratio followed by iid Bernoulli draws
- mask all selected positions at once
- run a single full-sequence masked forward pass
- in token-only mode, when `stage1_right_window_size >= 1`, every row may
  attend up to that many positions on the right
- when `stage1_full_seq` is disabled, compute the stage-1 auxiliary loss only
  on predictor rows `M - 1` for masked positions `M`
- when `stage1_full_seq` is enabled, compute the auxiliary loss on all valid
  rows under pure causal attention, or under a custom per-example mask that
  gives predictor rows `M - 1` right lookahead while blocking non-predictor
  rows from attending to `M - 1`
- use the same masked-loss averaging convention as the rest of the script

Stage 2 baseline:
- sample `stage2_num_groups` non-overlapping contiguous MASK blocks
  of length `stage2_group_size`, with gap `stage2_group_gap`
- mask all selected positions at once
- run a single full-sequence masked forward pass; when
  `stage2_right_window_size >= 1`, every row may attend up to that many
  positions on the right
- when `stage2_full_seq` is disabled, compute the auxiliary loss only on
  predictor rows `M - 1` for masked positions `M`
- when `stage2_full_seq` is enabled, compute the auxiliary loss on all valid
  rows under pure causal attention
- use the same masked-loss averaging convention as the rest of the script

The stage-2 training objective is:
`ntp_loss_stage2_downscale * L_ntp + adaptive_stage2_downscale * L_mtp_baseline`

This baseline intentionally removes draft-based stage-2 training and
plan1/plan2/plan3 evaluation logic. Final evaluation is left-to-right only.

"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
import tiktoken
from datetime import timedelta
from types import SimpleNamespace
from functools import partial
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
try:
    import wandb
except ImportError:
    wandb = None

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

def _parse_positive_token_count(value: str) -> int:
    cleaned = value.replace(",", "").replace("_", "")
    try:
        parsed = int(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("final-eval-size must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("final-eval-size must be a positive integer")
    return parsed


parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--num-epochs", type=int, default=16)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--adam-lr", type=float, default=3e-3)
parser.add_argument("--warmup-ratio", type=float, default=0.01)
parser.add_argument("--warmdown-ratio", type=float, default=0.1)
parser.add_argument("--final-lr-frac", type=float, default=0.01)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--weight-decay", type=float, default=0.8)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=262144)
# total_bsize = device_bsize * 8 * 2048 * grad_accum_steps
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=12)
parser.add_argument("--n_head", type=int, default=12)
parser.add_argument("--n_kv_head", type=int, default=12)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_project", type=str, default="overtrain-dclm")
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--log-grad-norms", action="store_true")
parser.add_argument("--mask-token-id", type=int, default=None)
parser.add_argument("--adaptive-stage0-epochs", type=int, default=3)
parser.add_argument("--adaptive-stage1-epochs", type=int, default=3)
parser.add_argument("--stage1-group-size", type=int, default=4)
parser.add_argument("--stage1-num-groups", type=int, default=32)
parser.add_argument("--stage1-mask-sampling", type=str, default="fixed_count", choices=("fixed_count", "uniform_ratio"))
parser.add_argument("--stage1-mask-ratio-min", type=float, default=0.05)
parser.add_argument("--stage1-mask-ratio-max", type=float, default=0.5)
parser.add_argument("--stage1-right-window-size", type=int, default=0)
parser.add_argument("--stage1-full-seq", action="store_true")
parser.add_argument("--stage1-full-seq-use-bidirectional", action="store_true")
parser.add_argument("--stage2-group-size", type=int, default=4)
parser.add_argument("--stage2-num-groups", type=int, default=32)
parser.add_argument("--stage2-group-gap", type=int, default=4)
parser.add_argument("--stage2-right-window-size", type=int, default=0)
parser.add_argument("--stage2-full-seq", action="store_true")
parser.add_argument("--adaptive-loss-downscale", type=float, default=0.1)
parser.add_argument("--adaptive-stage2-downscale", type=float, default=0.1)
parser.add_argument("--ltr-length", type=int, default=8)
parser.add_argument("--final-eval-size", type=_parse_positive_token_count, default=None)
parser.add_argument("--do_bidirectional_eval", action="store_true")
parser.add_argument("--eval-right-window-size", type=int, default=None)
parser.add_argument("--eval-crop-bucket", type=int, default=256)
parser.add_argument("--save-final-checkpoint", type=str, default="")
parser.add_argument("--load-final-checkpoint", type=str, default="")
parser.add_argument("--eval-only-final", action="store_true")
parser.add_argument("--ntp-loss-downscale", type=float, default=1.0)
parser.add_argument("--ntp-loss-stage2-downscale", type=float, default=1.0)
args = parser.parse_args()
args.stage1_right_window_size = max(0, args.stage1_right_window_size)
args.stage2_right_window_size = max(0, args.stage2_right_window_size)

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json
if args.eval_only_final and not args.load_final_checkpoint:
    raise ValueError("--eval-only-final requires --load-final-checkpoint")
if not (0 <= args.adaptive_stage0_epochs <= args.num_epochs):
    raise ValueError("--adaptive-stage0-epochs must be in [0, num-epochs]")
if args.adaptive_stage1_epochs < 0:
    raise ValueError("--adaptive-stage1-epochs must be non-negative")
if args.adaptive_stage0_epochs + args.adaptive_stage1_epochs > args.num_epochs:
    raise ValueError("--adaptive-stage0-epochs + --adaptive-stage1-epochs must be <= num-epochs")
if args.stage1_full_seq_use_bidirectional and not args.stage1_full_seq:
    raise ValueError("--stage1-full-seq-use-bidirectional requires --stage1-full-seq")
if args.stage1_full_seq_use_bidirectional and args.stage1_right_window_size < 1:
    raise ValueError("--stage1-full-seq-use-bidirectional requires --stage1-right-window-size >= 1")
if args.stage2_full_seq and args.stage2_right_window_size >= 1:
    raise ValueError("--stage2-full-seq requires --stage2-right-window-size < 1")
if args.ltr_length < 1:
    raise ValueError("--ltr-length must be at least 1")
if not (0.0 <= args.stage1_mask_ratio_min <= 1.0):
    raise ValueError("--stage1-mask-ratio-min must be in [0, 1]")
if not (0.0 <= args.stage1_mask_ratio_max <= 1.0):
    raise ValueError("--stage1-mask-ratio-max must be in [0, 1]")
if args.stage1_mask_ratio_min > args.stage1_mask_ratio_max:
    raise ValueError("--stage1-mask-ratio-min must be <= --stage1-mask-ratio-max")
if not (0.0 <= args.adaptive_stage2_downscale <= 1.0):
    raise ValueError("--adaptive-stage2-downscale must be in [0, 1]")
if not (0.0 <= args.ntp_loss_stage2_downscale <= 1.0):
    raise ValueError("--ntp-loss-stage2-downscale must be in [0, 1]")
if args.stage2_group_gap <= 0:
    raise ValueError("--stage2-group-gap must be positive")
if args.n_layer <= 0:
    raise ValueError("--n_layer must be positive")
if args.n_head <= 0:
    raise ValueError("--n_head must be positive")
if args.n_kv_head <= 0:
    raise ValueError("--n_kv_head must be positive")
if args.n_embd <= 0:
    raise ValueError("--n_embd must be positive")
if args.n_embd % args.n_head != 0:
    raise ValueError("--n_embd must be divisible by --n_head")
if args.n_head % args.n_kv_head != 0:
    raise ValueError("--n_head must be divisible by --n_kv_head")
if args.adam_lr <= 0:
    raise ValueError("--adam-lr must be positive")
if not (0.0 <= args.warmup_ratio <= 1.0):
    raise ValueError("--warmup-ratio must be in [0, 1]")
if not (0.0 <= args.warmdown_ratio <= 1.0):
    raise ValueError("--warmdown-ratio must be in [0, 1]")
if not (0.0 <= args.final_lr_frac <= 1.0):
    raise ValueError("--final-lr-frac must be in [0, 1]")
if args.max_grad_norm <= 0:
    raise ValueError("--max-grad-norm must be positive")


def _compute_mlp_hidden_dim(n_embd: int) -> int:
    return 256 * ((8 * n_embd // 3 + 255) // 256)

# =============================================================================
# Hyperparameters
# =============================================================================

# Architecture
DEPTH = args.n_layer if args.n_layer is not None else 12
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 12
N_KV_HEAD = args.n_kv_head if args.n_kv_head is not None else 12
HEAD_DIM = N_EMBD // N_HEAD
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = 10_000_000
DATA_DIR = "dclm_data"

# Base optimizer hyperparameters
BASE_ADAM_LR = args.adam_lr

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
ADAM_LR = BASE_ADAM_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WARMUP_RATIO = args.warmup_ratio
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = args.final_lr_frac
MAX_GRAD_NORM = args.max_grad_norm

# Stagewise masked training
ADAPTIVE_STAGE0_EPOCHS = args.adaptive_stage0_epochs
ADAPTIVE_STAGE1_EPOCHS = args.adaptive_stage1_epochs
STAGE1_GROUP_SIZE = args.stage1_group_size
STAGE1_NUM_GROUPS = args.stage1_num_groups
STAGE1_TOTAL_MASKS = STAGE1_NUM_GROUPS * STAGE1_GROUP_SIZE
STAGE1_MASK_SAMPLING = args.stage1_mask_sampling
STAGE1_MASK_RATIO_MIN = args.stage1_mask_ratio_min
STAGE1_MASK_RATIO_MAX = args.stage1_mask_ratio_max
STAGE1_RIGHT_WINDOW_SIZE = args.stage1_right_window_size
STAGE1_FULL_SEQ = args.stage1_full_seq
STAGE1_FULL_SEQ_USE_BIDIRECTIONAL = args.stage1_full_seq_use_bidirectional
STAGE2_GROUP_SIZE = args.stage2_group_size
STAGE2_NUM_GROUPS = args.stage2_num_groups
STAGE2_GROUP_GAP = args.stage2_group_gap
STAGE2_RIGHT_WINDOW_SIZE = args.stage2_right_window_size
STAGE2_FULL_SEQ = args.stage2_full_seq
ADAPTIVE_LOSS_DOWNSCALE = args.adaptive_loss_downscale
ADAPTIVE_STAGE2_DOWNSCALE = args.adaptive_stage2_downscale
LTR_LENGTH = args.ltr_length
EVAL_CROP_BUCKET = args.eval_crop_bucket
NTP_LOSS_DOWNSCALE = args.ntp_loss_downscale
NTP_LOSS_STAGE2_DOWNSCALE = args.ntp_loss_stage2_downscale
assert ADAPTIVE_LOSS_DOWNSCALE >= 0.0 and ADAPTIVE_LOSS_DOWNSCALE <= 1.0, f"Value Error: ADAPTIVE_LOSS_DOWNSCALE must be in [0.0, 1.0]"
assert ADAPTIVE_STAGE2_DOWNSCALE >= 0.0 and ADAPTIVE_STAGE2_DOWNSCALE <= 1.0, f"Value Error: ADAPTIVE_STAGE2_DOWNSCALE must be in [0.0, 1.0]"
assert NTP_LOSS_DOWNSCALE >= 0.0 and NTP_LOSS_DOWNSCALE <= 1.0, f"Value Error: NTP_LOSS_DOWNSCALE must be in [0.0, 1.0]"
assert NTP_LOSS_STAGE2_DOWNSCALE >= 0.0 and NTP_LOSS_STAGE2_DOWNSCALE <= 1.0, f"Value Error: NTP_LOSS_STAGE2_DOWNSCALE must be in [0.0, 1.0]"
# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self):
        self.summary = {}
        self.url = None
    def log(self, *a, **kw):
        pass
    def log_code(self, *a, **kw):
        pass
    def finish(self):
        pass


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def persist_result_json(path, updates):
    if not path or int(os.environ.get("RANK", 0)) != 0:
        return
    payload = _load_json_if_exists(path)
    payload.update(updates)
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_checkpoint_payload(orig_model, mask_token_id, run_name, step, current_epoch, extra_summary=None):
    state_dict = {name: tensor.detach().cpu() for name, tensor in orig_model.state_dict().items()}
    payload = {
        "model_state": state_dict,
        "model_config": asdict(orig_model.config),
        "mask_token_id": int(mask_token_id),
        "run": run_name,
        "step": int(step),
        "current_epoch": int(current_epoch),
        "args": vars(args).copy(),
    }
    if extra_summary:
        payload["training_summary"] = extra_summary
    return payload


def save_final_checkpoint(path, orig_model, mask_token_id, run_name, step, current_epoch, extra_summary=None):
    _ensure_parent_dir(path)
    torch.save(
        build_checkpoint_payload(orig_model, mask_token_id, run_name, step, current_epoch, extra_summary=extra_summary),
        path,
    )


def load_final_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)

# =============================================================================
# Flash Attention (FA3 on Hopper, SDPA fallback elsewhere)
# =============================================================================

def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_fa3()

def _merge_with_causal_window(attn_mask, q, k, window_size):
    Tq, Tk = q.size(2), k.size(2)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    window = window_size[0]
    if attn_mask is None:
        base = col_idx <= row_idx
        if window >= 0 and window < Tk:
            base = base & ((row_idx - col_idx) <= window)
        return base.view(1, 1, Tq, Tk)
    if attn_mask.dtype != torch.bool:
        raise TypeError("custom attention mask must be boolean")
    causal = (col_idx <= row_idx).view(1, 1, Tq, Tk)
    left = attn_mask & causal
    if window >= 0 and window < Tk:
        left_window = ((row_idx - col_idx) <= window).view(1, 1, Tq, Tk)
        left = left & left_window
    extra_right = attn_mask & (~causal)
    return left | extra_right


def _sdpa_attention(q, k, v, window_size, enable_gqa, attn_mask=None, causal=False):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if attn_mask is None and (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)
    if attn_mask is None and Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    merged_mask = _merge_with_causal_window(attn_mask, q, k, window_size)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=merged_mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), attn_mask=None):
    """Flash Attention for training. q,k,v: (B, T, H, D)."""
    if attn_mask is None and _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, attn_mask=attn_mask, causal=causal)
    return y.transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 32768
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_KV_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.1         


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, (x.size(-1),), weight=weight, eps=self.eps)


def _init_trunc_normal_(tensor, std):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _linear_fan_in_std(linear):
    return linear.in_features ** -0.5

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos_sin, window_size, attn_mask=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=(attn_mask is None), window_size=window_size, attn_mask=attn_mask)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = _compute_mlp_hidden_dim(config.n_embd)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        # self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin, window_size, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), cos_sin, window_size, attn_mask=attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def apply_runtime_precision_fixup(self):
        # Match slowrun's CUDA runtime precision: bf16 embeddings and bf16 norm weights.
        if self.transformer.wte.weight.device.type != "cuda":
            return
        self.transformer.wte.to(dtype=torch.bfloat16)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def init_weights(self):
        _init_trunc_normal_(self.transformer.wte.weight, std=self.config.n_embd ** -1)
        _init_trunc_normal_(self.lm_head.weight, std=_linear_fan_in_std(self.lm_head))
        self.final_norm.weight.fill_(1.0)
        for block in self.transformer.h:
            block.attn_norm.weight.fill_(1.0)
            block.mlp_norm.weight.fill_(1.0)
            block.attn.q_norm.weight.fill_(1.0)
            block.attn.k_norm.weight.fill_(1.0)
            _init_trunc_normal_(block.attn.c_q.weight, std=_linear_fan_in_std(block.attn.c_q))
            _init_trunc_normal_(block.attn.c_k.weight, std=_linear_fan_in_std(block.attn.c_k))
            _init_trunc_normal_(block.attn.c_v.weight, std=_linear_fan_in_std(block.attn.c_v))
            _init_trunc_normal_(block.attn.c_proj.weight, std=_linear_fan_in_std(block.attn.c_proj))
            _init_trunc_normal_(block.mlp.c_gate.weight, std=_linear_fan_in_std(block.mlp.c_gate))
            _init_trunc_normal_(block.mlp.c_fc.weight, std=_linear_fan_in_std(block.mlp.c_fc))
            _init_trunc_normal_(block.mlp.c_proj.weight, std=_linear_fan_in_std(block.mlp.c_proj))
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.apply_runtime_precision_fixup()

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) if w[0] >= 0 else 12 * h * q * t for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self):
        no_decay = set()
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, RMSNorm)):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    no_decay.add(full_name)
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or name in no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = DistShardedAdamW(
            [
                {"params": decay_params, "lr": ADAM_LR, "betas": ADAM_BETAS, "eps": ADAM_EPS, "weight_decay": WEIGHT_DECAY},
                {"params": no_decay_params, "lr": ADAM_LR, "betas": ADAM_BETAS, "eps": ADAM_EPS, "weight_decay": 0.0},
            ],
            max_grad_norm=MAX_GRAD_NORM,
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx=None, targets=None, loss_reduction='mean', attn_mask=None, loss_mask=None, input_embeds=None):
        if (idx is None) == (input_embeds is None):
            raise ValueError("exactly one of idx or input_embeds must be provided")
        if idx is not None:
            B, T = idx.size()
            x = self.transformer.wte(idx)
        else:
            B, T, _ = input_embeds.size()
            x = input_embeds.to(self.transformer.wte.weight.dtype)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, self.window_sizes[i], attn_mask=attn_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is not None:
            if loss_mask is None:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction='none',
            ).view(B, T)
            valid = loss_mask & (targets != -1)
            if loss_reduction == 'none':
                return per_tok * valid
            valid_f = valid.to(per_tok.dtype)
            masked_sum = (per_tok * valid_f).sum()
            if loss_reduction == 'sum':
                return masked_sum
            if loss_reduction == 'mean':
                return masked_sum / valid_f.sum().clamp_min(1.0)
            raise ValueError(f"unsupported loss_reduction={loss_reduction}")
        return logits

# =============================================================================
# Optimizer helpers
# =============================================================================

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))


class DistShardedAdamW(torch.optim.Optimizer):
    """Distributed AdamW with ZeRO-2 style state sharding for large tensors."""

    def __init__(self, param_groups, max_grad_norm):
        super().__init__(param_groups, defaults={})
        self.max_grad_norm = float(max_grad_norm)
        self._step_t = torch.tensor(0.0)
        self._lr_t = torch.tensor(0.0)
        self._beta1_t = torch.tensor(0.0)
        self._beta2_t = torch.tensor(0.0)
        self._eps_t = torch.tensor(0.0)
        self._wd_t = torch.tensor(0.0)

    def _world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    def _rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def _should_shard(self, p, world_size):
        grad = p.grad
        return (
            world_size > 1
            and grad is not None
            and grad.ndim > 0
            and p.numel() >= 1024
            and grad.shape[0] % world_size == 0
        )

    def _reduce_group(self, group, world_size):
        infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if self._should_shard(p, world_size):
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
            else:
                future = None
                if world_size > 1:
                    future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
        return dict(param_infos=infos)

    def _wait_reductions(self, reduce_infos):
        for info in reduce_infos:
            for pinfo in info["param_infos"].values():
                if pinfo["future"] is not None:
                    pinfo["future"].wait()

    def _grad_norm_device(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return p.grad.device
        return torch.device("cpu")

    def _compute_global_grad_norm(self, reduce_infos, world_size):
        total_sq = torch.zeros((), dtype=torch.float64, device=self._grad_norm_device())
        for info in reduce_infos:
            for pinfo in info["param_infos"].values():
                grad_sq = pinfo["grad_slice"].detach().float().square().sum(dtype=torch.float64)
                if pinfo["is_small"] and world_size > 1:
                    grad_sq /= world_size
                total_sq += grad_sq
        if world_size > 1:
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        return total_sq.sqrt().item()

    def _build_adamw_entries(self, group, info, rank, world_size):
        entries = []
        for p in group["params"]:
            pinfo = info["param_infos"].get(p)
            if pinfo is None:
                continue
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            entries.append((p, pinfo, p_slice, state))
        return entries

    def _step_adamw_single(self, p, pinfo, p_slice, state, group, clip_coef, gather_futures):
        grad_slice = pinfo["grad_slice"]
        if clip_coef < 1.0:
            grad_slice.mul_(clip_coef)
        state["step"] += 1
        self._step_t.fill_(state["step"])
        self._lr_t.fill_(group["lr"])
        self._beta1_t.fill_(group["betas"][0])
        self._beta2_t.fill_(group["betas"][1])
        self._eps_t.fill_(group["eps"])
        self._wd_t.fill_(group["weight_decay"])
        adamw_step_fused(
            p_slice,
            grad_slice,
            state["exp_avg"],
            state["exp_avg_sq"],
            self._step_t,
            self._lr_t,
            self._beta1_t,
            self._beta2_t,
            self._eps_t,
            self._wd_t,
        )
        if not pinfo["is_small"]:
            future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
            gather_futures.append(future)

    def _step_adamw_bucket(self, bucket_entries, group, clip_coef, gather_futures):
        if not bucket_entries:
            return
        if len(bucket_entries) == 1:
            self._step_adamw_single(*bucket_entries[0], group, clip_coef, gather_futures)
            return

        current_steps = []
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        for p, pinfo, p_slice, state in bucket_entries:
            current_steps.append(state["step"])
            params.append(p_slice)
            grads.append(pinfo["grad_slice"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

        if any(step != current_steps[0] for step in current_steps[1:]):
            for entry in bucket_entries:
                self._step_adamw_single(*entry, group, clip_coef, gather_futures)
            return

        step = current_steps[0] + 1
        for _, _, _, state in bucket_entries:
            state["step"] = step

        if clip_coef < 1.0:
            torch._foreach_mul_(grads, clip_coef)

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step

        torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        grad_sq = torch._foreach_mul(grads, grads)
        torch._foreach_lerp_(exp_avg_sqs, grad_sq, 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        denom = torch._foreach_sqrt(denom)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-(lr / bias1))

        for p, pinfo, p_slice, _ in bucket_entries:
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_futures.append(future)

    def _compute_adamw(self, group, info, rank, world_size, clip_coef, gather_futures):
        entries = self._build_adamw_entries(group, info, rank, world_size)
        buckets = {}
        for entry in entries:
            _, pinfo, p_slice, _ = entry
            key = (pinfo["is_small"], tuple(p_slice.shape), p_slice.dtype)
            buckets.setdefault(key, []).append(entry)
        for bucket_entries in buckets.values():
            self._step_adamw_bucket(bucket_entries, group, clip_coef, gather_futures)

    @torch.no_grad()
    def step(self):
        world_size = self._world_size()
        rank = self._rank()
        reduce_infos = [self._reduce_group(group, world_size) for group in self.param_groups]
        self._wait_reductions(reduce_infos)
        grad_norm = self._compute_global_grad_norm(reduce_infos, world_size)
        clip_coef = 1.0
        if self.max_grad_norm > 0.0 and math.isfinite(grad_norm) and grad_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
        gather_futures = []
        for group, info in zip(self.param_groups, reduce_infos):
            self._compute_adamw(group, info, rank, world_size, clip_coef, gather_futures)
        for future in gather_futures:
            future.wait()
        if args.log_grad_norms:
            return {"grad_norm": grad_norm}
        return None


def _model_device(model):
    return next(model.parameters()).device
# =============================================================================
# Dataloader: BOS-aligned best-fit packing
# =============================================================================

class DataLoader:
    """Pre-tokenized chunk dataloader. Yields (inputs, targets, epoch) forever."""

    def __init__(self, filepath, B, T, device="cuda"):
        data = torch.load(filepath, weights_only=True)
        chunks = data['chunks']
        valid_counts = data['valid_counts']
        file_B = data['batch_size']
        sequence_size = data['sequence_size']
        assert sequence_size == T + 1, f"Data sequence_size {sequence_size} != T+1={T+1}"

        # Gather all valid sequences into one tensor
        all_seqs = []
        for chunk, vc in zip(chunks, valid_counts):
            rows = chunk.view(file_B, sequence_size)[:vc]
            all_seqs.append(rows)
        all_seqs = torch.cat(all_seqs, dim=0).long()  # (N, T+1)

        # DDP sharding: each rank gets every world_size-th batch
        _, rank, _, world_size = get_dist_info()
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, world_size, B, sequence_size)

        self.rank_data = all_seqs[:, rank].contiguous()  # (num_steps, B, T+1)
        self.num_steps = num_steps
        self.total_tokens = usable * T  # trainable tokens across all ranks
        self.device = device
        self.pos = 0
        self.epoch = 1

    def __iter__(self):
        return self

    def _shuffle(self):
        """Shuffle batch order for the new epoch, consistent across ranks."""
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(self.num_steps, generator=g)
        self.rank_data = self.rank_data[perm]
        
    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            print0(f"Starting epoch {self.epoch}")
            self._shuffle()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch


def _assert_loader_vocab_compatible(loader, max_token_id, split_name):
    if loader.rank_data.numel() == 0:
        return
    observed_max = int(loader.rank_data.max().item())
    if observed_max > max_token_id:
        raise ValueError(
            f"{split_name} data contains token id {observed_max}, but this model only supports token ids <= {max_token_id}. "
            "Regenerate the packed dataset with the same tokenizer used by this trainer."
        )

# =============================================================================
# Adaptive masked objective
# =============================================================================


_right_window_mask_cache = {}


def get_causal_plus_right_window_mask(T, right_window_size, device):
    if right_window_size < 1:
        return None
    right_window_size = int(right_window_size)
    key = (str(device), T, right_window_size)
    mask = _right_window_mask_cache.get(key)
    if mask is None or mask.device != device:
        row_idx = torch.arange(T, device=device).view(T, 1)
        col_idx = torch.arange(T, device=device).view(1, T)
        causal = col_idx <= row_idx
        right_limit = torch.clamp(row_idx + right_window_size, max=T - 1)
        right_band = (col_idx > row_idx) & (col_idx <= right_limit)
        mask = (causal | right_band).view(1, 1, T, T)
        _right_window_mask_cache[key] = mask
    return mask


def build_stage1_full_seq_bidirectional_mask(loss_mask, right_window_size):
    if right_window_size < 1:
        raise ValueError("stage1 full-seq bidirectional mask requires right_window_size >= 1")
    B, T = loss_mask.shape
    device = loss_mask.device
    row_idx = torch.arange(T, device=device).view(1, T, 1)
    col_idx = torch.arange(T, device=device).view(1, 1, T)
    causal = col_idx <= row_idx
    right_limit = torch.clamp(row_idx + int(right_window_size), max=T - 1)
    right_band = (col_idx > row_idx) & (col_idx <= right_limit)
    predictor_rows = loss_mask.unsqueeze(-1)
    predictor_cols = loss_mask.unsqueeze(1)
    allow = torch.where(
        predictor_rows,
        causal | right_band,
        causal & (~predictor_cols),
    )
    return allow.unsqueeze(1)


def _stage1_total_masks_from_metadata(args_dict):
    if args_dict.get("stage1_mask_sampling", STAGE1_MASK_SAMPLING) != "fixed_count":
        return None
    return args_dict.get("stage1_num_groups", STAGE1_NUM_GROUPS) * args_dict.get("stage1_group_size", STAGE1_GROUP_SIZE)


class AdaptiveDreamTrainer:
    """Builds stage-1 and stage-2 masked batches.

    Sample target positions G in token space and apply loss on predictor rows
    P = G - 1 under the shift-1 interface.
    """

    def __init__(self, seq_len, mask_token_id,
                 stage1_group_size, stage1_num_groups,
                 stage1_mask_sampling, stage1_mask_ratio_min, stage1_mask_ratio_max,
                 stage2_group_size, stage2_num_groups, stage2_group_gap,
                 stage1_right_window_size, stage2_right_window_size,
                 stage1_full_seq, stage1_full_seq_use_bidirectional, stage2_full_seq,
                 stage0_epochs, total_epochs, ltr_length=16, seed=1234):
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.stage1_group_size = stage1_group_size
        self.stage1_num_groups = stage1_num_groups
        self.stage1_total_masks = stage1_group_size * stage1_num_groups
        self.stage1_mask_sampling = str(stage1_mask_sampling)
        self.stage1_mask_ratio_min = float(stage1_mask_ratio_min)
        self.stage1_mask_ratio_max = float(stage1_mask_ratio_max)
        self.stage1_right_window_size = max(0, int(stage1_right_window_size))
        self.stage1_full_seq = bool(stage1_full_seq)
        self.stage1_full_seq_use_bidirectional = bool(stage1_full_seq_use_bidirectional)
        self.stage2_group_size = stage2_group_size
        self.stage2_num_groups = stage2_num_groups
        self.stage2_group_gap = stage2_group_gap
        self.stage2_right_window_size = max(0, int(stage2_right_window_size))
        self.stage2_full_seq = bool(stage2_full_seq)
        self.stage0_epochs = stage0_epochs
        self.total_epochs = total_epochs
        self.ltr_length = ltr_length
        self.rank = int(os.environ.get('RANK', 0))
        self.cpu_gen = torch.Generator(device='cpu')
        self.cpu_gen.manual_seed(seed + self.rank)
        if self.stage1_mask_sampling not in ("fixed_count", "uniform_ratio"):
            raise ValueError("stage1_mask_sampling must be one of: fixed_count, uniform_ratio")
        if not (0.0 <= self.stage1_mask_ratio_min <= self.stage1_mask_ratio_max <= 1.0):
            raise ValueError("stage1 mask ratio bounds must satisfy 0 <= min <= max <= 1")
        if self.stage1_full_seq_use_bidirectional and not self.stage1_full_seq:
            raise ValueError("stage1_full_seq_use_bidirectional requires stage1_full_seq")
        if self.stage1_full_seq_use_bidirectional and self.stage1_right_window_size < 1:
            raise ValueError("stage1_full_seq_use_bidirectional requires stage1_right_window_size >= 1")
        if self.stage2_full_seq and self.stage2_right_window_size >= 1:
            raise ValueError("stage2_full_seq requires stage2_right_window_size < 1")
        required_stage2_space = (
            self.stage2_num_groups * self.stage2_group_size
            + (self.stage2_num_groups - 1) * self.stage2_group_gap
        )
        available_stage2_space = self.seq_len - self.ltr_length
        if required_stage2_space > available_stage2_space:
            raise ValueError(
                "stage2 block packing requires "
                "stage2_num_groups * stage2_group_size + (stage2_num_groups - 1) * stage2_group_gap <= seq_len - ltr_length"
            )

    def _randint(self, low, high):
        return int(torch.randint(low, high, (1,), generator=self.cpu_gen).item())

    def stage1_enabled(self):
        if self.stage1_mask_sampling == "uniform_ratio":
            return self.stage1_mask_ratio_max > 0.0
        return self.stage1_num_groups > 0 and self.stage1_group_size > 0

    def sample_stage1_mask_ratio(self):
        if self.stage1_mask_ratio_min == self.stage1_mask_ratio_max:
            return self.stage1_mask_ratio_min
        unit = torch.rand((), generator=self.cpu_gen).item()
        return self.stage1_mask_ratio_min + (self.stage1_mask_ratio_max - self.stage1_mask_ratio_min) * unit

    def sample_stage1_positions(self, T):
        target_lo = self.ltr_length
        target_hi = T - 1
        if target_lo > target_hi:
            raise ValueError(f"cannot sample stage1 mask positions from [{target_lo}, {target_hi}] for T={T}")
        available = target_hi - target_lo + 1
        if self.stage1_mask_sampling == "uniform_ratio":
            mask_ratio = self.sample_stage1_mask_ratio()
            sampled = torch.rand(available, generator=self.cpu_gen) < mask_ratio
            return [target_lo + idx for idx, keep in enumerate(sampled.tolist()) if keep]
        if self.stage1_total_masks > available:
            raise ValueError(
                "stage1 mask sampling requires "
                "stage1_num_groups * stage1_group_size <= T - ltr_length"
            )
        choices = torch.randperm(available, generator=self.cpu_gen).tolist()[:self.stage1_total_masks]
        return sorted(target_lo + idx for idx in choices)

    def sample_stage2_block_starts(self, T):
        if self.stage2_num_groups <= 0:
            return []
        start_lo = self.ltr_length
        start_hi = T - self.stage2_group_size
        if start_lo > start_hi:
            raise ValueError(
                f"cannot sample stage2 block starts from [{start_lo}, {start_hi}] for T={T}"
            )
        required = (
            self.stage2_num_groups * self.stage2_group_size
            + (self.stage2_num_groups - 1) * self.stage2_group_gap
        )
        available = T - self.ltr_length
        if required > available:
            raise ValueError(
                "stage2 block packing requires "
                "stage2_num_groups * stage2_group_size + (stage2_num_groups - 1) * stage2_group_gap <= T - ltr_length"
            )
        min_sep = self.stage2_group_size + self.stage2_group_gap
        num_starts = start_hi - start_lo + 1
        compressed_slots = num_starts - (self.stage2_num_groups - 1) * (min_sep - 1)
        if compressed_slots < self.stage2_num_groups:
            raise ValueError(
                f"cannot sample {self.stage2_num_groups} stage2 blocks with min separation {min_sep} from T={T}"
            )
        compressed = torch.randperm(compressed_slots, generator=self.cpu_gen).tolist()[:self.stage2_num_groups]
        compressed.sort()
        return [start_lo + c + i * (min_sep - 1) for i, c in enumerate(compressed)]

    def build_batch(self, x):
        B, T = x.shape
        if not self.stage1_enabled():
            masked_idx = x.clone()
            loss_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
            return masked_idx, loss_mask
        masked_idx = x.clone()
        loss_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)

        for b in range(B):
            all_targets = self.sample_stage1_positions(T)
            if all_targets:
                masked_idx[b, all_targets] = self.mask_token_id

                # Supervision lives on predictor rows P = G - 1.
                predictors = [pos - 1 for pos in all_targets]
                loss_mask[b, predictors] = True

        return masked_idx, loss_mask

    def compute_loss(self, model, x, y, backward_scale=None):
        if not self.stage1_enabled():
            return torch.zeros((), dtype=torch.float32, device=x.device)
        total = None
        masked_idx, loss_mask = self.build_batch(x)
        if self.stage1_full_seq:
            if self.stage1_full_seq_use_bidirectional:
                attn_mask = build_stage1_full_seq_bidirectional_mask(
                    loss_mask, self.stage1_right_window_size
                )
                loss = model(masked_idx, y, attn_mask=attn_mask)
            else:
                loss = model(masked_idx, y)
        else:
            attn_mask = get_causal_plus_right_window_mask(
                masked_idx.size(1), self.stage1_right_window_size, x.device
            )
            if attn_mask is None:
                loss = model(masked_idx, y, loss_mask=loss_mask)
            else:
                loss = model(masked_idx, y, attn_mask=attn_mask, loss_mask=loss_mask)
        if backward_scale is None:
            total = loss if total is None else total + loss
        else:
            # Backprop each repeat immediately to release its graph while preserving
            # the same averaged gradient as (sum(losses) / repeats).backward().
            # We downweight the loss by ADAPTIVE_LOSS_DOWNSCALE since the loss is only sum of losses at masked positions,
            # which is much smaller than the sequence length (2048).
            (loss * backward_scale * ADAPTIVE_LOSS_DOWNSCALE).backward()
            loss_detached = loss.detach()
            total = loss_detached if total is None else total + loss_detached
        return total

    def build_stage2_mtp_batch(self, x):
        B, T = x.shape
        device = x.device
        if self.stage2_num_groups <= 0 or self.stage2_group_size <= 0:
            masked_idx = x.clone()
            loss_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
            return masked_idx, loss_mask
        block_starts_list = []
        for _ in range(B):
            starts = self.sample_stage2_block_starts(T)
            block_starts_list.append(starts)
        if not block_starts_list or not block_starts_list[0]:
            raise RuntimeError("stage2 training requires at least one block")

        block_starts = torch.tensor(block_starts_list, dtype=torch.long, device=device)
        group_offsets = torch.arange(self.stage2_group_size, dtype=torch.long, device=device)
        block_positions = block_starts.unsqueeze(-1) + group_offsets.view(1, 1, -1)

        masked_idx = x.clone()
        masked_idx.scatter_(1, block_positions.reshape(B, -1), self.mask_token_id)

        loss_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        predictor_rows = (block_positions - 1).reshape(B, -1)
        loss_mask.scatter_(1, predictor_rows, torch.ones_like(predictor_rows, dtype=torch.bool))
        return masked_idx, loss_mask

    def compute_stage2_loss(self, model, x, y, backward_scale=None):
        if self.stage2_num_groups <= 0 or self.stage2_group_size <= 0:
            return torch.zeros((), dtype=torch.float32, device=x.device)
        masked_idx, loss_mask = self.build_stage2_mtp_batch(x)
        if self.stage2_full_seq:
            loss = model(masked_idx, y)
        else:
            attn_mask = get_causal_plus_right_window_mask(
                masked_idx.size(1), self.stage2_right_window_size, x.device
            )
            if attn_mask is None:
                loss = model(masked_idx, y, loss_mask=loss_mask)
            else:
                loss = model(masked_idx, y, attn_mask=attn_mask, loss_mask=loss_mask)
        if backward_scale is None:
            return loss
        (loss * backward_scale * ADAPTIVE_STAGE2_DOWNSCALE).backward()
        return loss.detach()

# =============================================================================
# Loss evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    model_device = _model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model_device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model_device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


def _local_eval_token_budget(total_tokens: int) -> int:
    _, rank, _, world_size = get_dist_info()
    base = total_tokens // world_size
    remainder = total_tokens % world_size
    return base + (1 if rank < remainder else 0)


def _iter_eval_token_budget(batches, total_tokens: int):
    remaining = _local_eval_token_budget(total_tokens)
    if remaining <= 0:
        return
    batch_iter = iter(batches)
    while remaining > 0:
        x, y, _ = next(batch_iter)
        B, T = x.shape
        batch_tokens = B * T
        if remaining >= batch_tokens:
            yield x, y
            remaining -= batch_tokens
            continue
        full_rows, tail_tokens = divmod(remaining, T)
        if full_rows > 0:
            yield x[:full_rows], y[:full_rows]
        if tail_tokens > 0:
            yield x[full_rows:full_rows + 1, :tail_tokens], y[full_rows:full_rows + 1, :tail_tokens]
        break


def _bucket_eval_length(required_length, T, crop_bucket):
    required_length = max(1, min(int(required_length), T))
    if crop_bucket is None or crop_bucket <= 0:
        return required_length
    return min(T, int(math.ceil(required_length / crop_bucket) * crop_bucket))


_bidirectional_eval_mask_cache = {}


def _build_bidirectional_eval_attn_mask(L, live_length, right_window_size, device):
    if right_window_size < 1:
        raise ValueError("bidirectional eval requires right_window_size >= 1")
    key = (str(device), int(L), int(live_length), int(right_window_size))
    mask = _bidirectional_eval_mask_cache.get(key)
    if mask is None or mask.device != device:
        mask = torch.zeros((1, 1, L, L), dtype=torch.bool, device=device)
        live_mask = get_causal_plus_right_window_mask(live_length, right_window_size, device)
        mask[:, :, :live_length, :live_length] = live_mask
        if live_length < L:
            pad_idx = torch.arange(live_length, L, device=device)
            mask[0, 0, pad_idx, pad_idx] = True
        _bidirectional_eval_mask_cache[key] = mask
    return mask


@torch.no_grad()
def _evaluate_bidirectional_batch(model, x, y, mask_token_id, ltr_length=16, eval_right_window_size=16, crop_bucket=256):
    device = _model_device(model)
    B, T = x.shape
    batch_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    batch_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    prefix_labels = min(ltr_length, T)
    if prefix_labels > 0:
        prefix_loss2d = model(
            x[:, :prefix_labels],
            y[:, :prefix_labels],
            loss_reduction='none',
        ).view(B, prefix_labels)
        prefix_valid = y[:, :prefix_labels] != -1
        batch_loss += prefix_loss2d[prefix_valid].sum().double()
        batch_tokens += prefix_valid.sum().to(torch.int64)
    if T <= ltr_length:
        return batch_loss, batch_tokens

    for t in range(ltr_length, T):
        live_length = min(T, t + 1 + eval_right_window_size)
        L = _bucket_eval_length(live_length, T, crop_bucket)
        input_idx = torch.full((B, L), mask_token_id, dtype=x.dtype, device=x.device)
        input_idx[:, :t + 1] = x[:, :t + 1]
        attn_mask = _build_bidirectional_eval_attn_mask(L, live_length, eval_right_window_size, x.device)
        logits = model(input_idx, attn_mask=attn_mask)
        row_loss = F.cross_entropy(logits[:, t], y[:, t], ignore_index=-1, reduction='none')
        valid = y[:, t] != -1
        batch_loss += row_loss[valid].sum().double()
        batch_tokens += valid.sum().to(torch.int64)
    return batch_loss, batch_tokens


@torch.no_grad()
def evaluate_bpb_token_budget(model, batches, total_tokens, token_bytes):
    model_device = _model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model_device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_tokens_seen = torch.tensor(0, dtype=torch.int64, device=model_device)
    for x, y in _iter_eval_token_budget(batches, total_tokens):
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens_seen += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_seen, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens_seen = total_loss.item(), total_tokens_seen.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens_seen if total_tokens_seen > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_bidirectional_loss_token_budget(model, batches, total_tokens, mask_token_id, ltr_length=16, eval_right_window_size=16, crop_bucket=256):
    model_device = _model_device(model)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=model_device)
    total_tokens_seen = torch.tensor(0, dtype=torch.int64, device=model_device)
    for x, y in _iter_eval_token_budget(batches, total_tokens):
        batch_loss, batch_tokens = _evaluate_bidirectional_batch(
            model,
            x,
            y,
            mask_token_id,
            ltr_length=ltr_length,
            eval_right_window_size=eval_right_window_size,
            crop_bucket=crop_bucket,
        )
        total_loss += batch_loss
        total_tokens_seen += batch_tokens
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_seen, op=dist.ReduceOp.SUM)
    total_loss, total_tokens_seen = total_loss.item(), total_tokens_seen.item()
    return total_loss / total_tokens_seen if total_tokens_seen > 0 else float('inf')


# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=7))
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# GPU info for MFU
gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name: gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

# FA3 status
if _fa3 is not None:
    print0("Using Flash Attention 3 (Hopper GPU detected) for pure causal batches; any custom attention mask, including right-looking masked passes and stage1 full-seq bidirectional masks, falls back to SDPA")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# Optional checkpoint load for eval-only final runs
checkpoint = load_final_checkpoint(args.load_final_checkpoint) if args.eval_only_final else None
checkpoint_args = checkpoint.get("args", {}) if checkpoint else {}
checkpoint_summary = checkpoint.get("training_summary", {}) if checkpoint else {}
metadata_args = checkpoint_args if checkpoint else vars(args)
loaded_run_name = checkpoint.get("run") if checkpoint else None
effective_eval_right_window_size = (
    int(args.eval_right_window_size)
    if args.eval_right_window_size is not None
    else int(metadata_args.get("stage2_right_window_size", STAGE2_RIGHT_WINDOW_SIZE))
)
final_eval_ltr_length = int(metadata_args.get("ltr_length", LTR_LENGTH))
if args.do_bidirectional_eval and effective_eval_right_window_size < 1:
    raise ValueError(
        "--do_bidirectional_eval requires an effective eval right window >= 1; "
        "provide --eval-right-window-size or train/load a checkpoint with stage2_right_window_size >= 1"
    )

# wandb
if args.run:
    run_name = args.run
elif args.eval_only_final and loaded_run_name:
    run_name = f"{loaded_run_name}_evalonly"
else:
    run_name = time.strftime("%Y%m%d_%H%M%S")
_wandb_kwargs = {"project": args.wandb_project, "name": run_name}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if (not master_process or wandb is None) else wandb.init(**_wandb_kwargs)
if master_process and wandb is not None:
    wandb_run.log_code(".")

# Load the original slowrun GPT-2 tokenizer and compute token_bytes for BPB evaluation.
encoder = tiktoken.get_encoding("gpt2")
base_vocab_size = encoder.n_vocab
mask_token_id = int(checkpoint["mask_token_id"]) if checkpoint else (args.mask_token_id if args.mask_token_id is not None else base_vocab_size)
if checkpoint:
    model_config = dict(checkpoint["model_config"])
    model_config.pop("mlp_dim", None)
    config = GPTConfig(**model_config)
else:
    vocab_size = max(base_vocab_size, mask_token_id + 1)
    config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout)
vocab_size = config.vocab_size

# Print hyperparameters
print0(f"--- Hyperparameters ---")
print0(f"  mode={'eval_only_final' if args.eval_only_final else 'train_and_eval'}")
print0(
    f"  n_layer={config.n_layer}, n_embd={config.n_embd}, mlp_dim={_compute_mlp_hidden_dim(config.n_embd)}, "
    f"n_head={config.n_head}, head_dim={config.n_embd // config.n_head}"
)
print0(f"  seq_len={config.sequence_len}, window_pattern={config.window_pattern}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
print0(f"  adam_lr={ADAM_LR}, lr_multiplier={args.lr_multiplier}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}, adam_eps={ADAM_EPS}")
print0(
    f"  warmup_ratio={WARMUP_RATIO}, scheduler=wsd, warmdown_ratio={WARMDOWN_RATIO}, "
    f"final_lr_frac={FINAL_LR_FRAC}, max_grad_norm={MAX_GRAD_NORM}"
)
print0(f"  wandb_project={args.wandb_project}, wandb_group={args.wandb_group}")
print0(f"  num_epochs={metadata_args.get('num_epochs', args.num_epochs)}, patience={args.patience}")
print0(f"  final_eval_size_requested={args.final_eval_size if args.final_eval_size is not None else EVAL_TOKENS}")
print0(f"  dropout={config.dropout}")
print0(
    f"  adaptive_stage0_epochs={metadata_args.get('adaptive_stage0_epochs', ADAPTIVE_STAGE0_EPOCHS)}, "
    f"adaptive_stage1_epochs={metadata_args.get('adaptive_stage1_epochs', ADAPTIVE_STAGE1_EPOCHS)}, "
    f"adaptive_loss_downscale={metadata_args.get('adaptive_loss_downscale', ADAPTIVE_LOSS_DOWNSCALE)}, "
    f"adaptive_stage2_downscale={metadata_args.get('adaptive_stage2_downscale', ADAPTIVE_STAGE2_DOWNSCALE)}, "
    f"ntp_loss_downscale={metadata_args.get('ntp_loss_downscale', NTP_LOSS_DOWNSCALE)}, "
    f"ntp_loss_stage2_downscale={metadata_args.get('ntp_loss_stage2_downscale', NTP_LOSS_STAGE2_DOWNSCALE)}"
)
print0(
    f"  stage1_group_size={metadata_args.get('stage1_group_size', STAGE1_GROUP_SIZE)}, "
    f"stage1_num_groups={metadata_args.get('stage1_num_groups', STAGE1_NUM_GROUPS)}, "
    f"stage1_total_masks={_stage1_total_masks_from_metadata(metadata_args)}, "
    f"stage1_mask_sampling={metadata_args.get('stage1_mask_sampling', STAGE1_MASK_SAMPLING)}, "
    f"stage1_mask_ratio_min={metadata_args.get('stage1_mask_ratio_min', STAGE1_MASK_RATIO_MIN)}, "
    f"stage1_mask_ratio_max={metadata_args.get('stage1_mask_ratio_max', STAGE1_MASK_RATIO_MAX)}, "
    f"stage1_right_window_size={metadata_args.get('stage1_right_window_size', STAGE1_RIGHT_WINDOW_SIZE)}, "
    f"stage1_full_seq={metadata_args.get('stage1_full_seq', STAGE1_FULL_SEQ)}, "
    f"stage1_full_seq_use_bidirectional={metadata_args.get('stage1_full_seq_use_bidirectional', STAGE1_FULL_SEQ_USE_BIDIRECTIONAL)}"
)
print0(
    f"  stage2_group_size={metadata_args.get('stage2_group_size', STAGE2_GROUP_SIZE)}, "
    f"stage2_num_groups={metadata_args.get('stage2_num_groups', STAGE2_NUM_GROUPS)}, "
    f"stage2_group_gap={metadata_args.get('stage2_group_gap', STAGE2_GROUP_GAP)}, "
    f"stage2_right_window_size={metadata_args.get('stage2_right_window_size', STAGE2_RIGHT_WINDOW_SIZE)}, "
    f"stage2_full_seq={metadata_args.get('stage2_full_seq', STAGE2_FULL_SEQ)}"
)
print0(f"  ltr_length={metadata_args.get('ltr_length', LTR_LENGTH)}")
print0(
    f"  do_bidirectional_eval={args.do_bidirectional_eval}, "
    f"eval_right_window_size={args.eval_right_window_size}, "
    f"effective_eval_right_window_size={effective_eval_right_window_size}, "
    f"eval_crop_bucket={EVAL_CROP_BUCKET}"
)
print0(
    f"  stage_schedule=(stage0<{metadata_args.get('adaptive_stage0_epochs', ADAPTIVE_STAGE0_EPOCHS)}, "
    f"stage1<{metadata_args.get('adaptive_stage0_epochs', ADAPTIVE_STAGE0_EPOCHS) + metadata_args.get('adaptive_stage1_epochs', ADAPTIVE_STAGE1_EPOCHS)}, "
    f"stage2 otherwise)"
)
print0(f"-----------------------")
print0(f"Base vocab size: {base_vocab_size:,} | model vocab size: {vocab_size:,} | mask_token_id={mask_token_id}")
if checkpoint:
    print0(f"Loaded final checkpoint metadata from {args.load_final_checkpoint}")

eot_id = encoder._special_tokens["<|endoftext|>"]
token_bytes_list = []
for token_id in range(vocab_size):
    if token_id == eot_id or token_id >= encoder.n_vocab:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(encoder.decode_single_token_bytes(token_id)))
token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

# Build model
with torch.device("meta"):
    orig_model = GPT(config)
orig_model.to_empty(device=device)
orig_model.init_weights()
if checkpoint:
    load_info = orig_model.load_state_dict(checkpoint["model_state"], strict=True)
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
        )
orig_model.apply_runtime_precision_fixup()

param_counts = sum(p.numel() for p in orig_model.parameters())
transformer_params = sum(p.numel() for p in orig_model.transformer.h.parameters())
lm_head_params = sum(p.numel() for p in orig_model.lm_head.parameters())
other_params = param_counts - transformer_params - lm_head_params
num_flops_per_token = orig_model.estimate_flops()
print0(f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Compile the full model, keeping an unwrapped reference for checkpoints.
model = torch.compile(orig_model, dynamic=False)

# Shared dataloaders / evaluation config
_train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "dclm_train.pt")
_val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "dclm_val.pt")
def build_val_loader():
    loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device)
    _assert_loader_vocab_compatible(loader, config.vocab_size - 1, "validation")
    return loader


def _resolve_eval_token_budget(loader, requested_tokens):
    target_tokens = EVAL_TOKENS if requested_tokens is None else requested_tokens
    return min(target_tokens, loader.total_tokens)


_val_loader_probe = build_val_loader()
default_eval_token_budget = _resolve_eval_token_budget(_val_loader_probe, None)
final_eval_token_budget = _resolve_eval_token_budget(_val_loader_probe, args.final_eval_size)
del _val_loader_probe

# Training/eval bookkeeping
step = int(checkpoint.get("step", 0)) if checkpoint else 0
current_epoch = int(checkpoint.get("current_epoch", 0)) if checkpoint else 0
val_loss = checkpoint_summary.get("last_val_loss")
min_val_bpb = checkpoint_summary.get("best_val_bpb", float("inf"))
min_val_loss = checkpoint_summary.get("best_val_loss", float("inf"))
epochs_without_improvement = 0
smooth_train_loss = 0.0
total_training_time = float(checkpoint_summary.get("total_training_time", 0.0))
final_train_loss = checkpoint_summary.get("final_train_loss", float("nan"))

result_payload = {
    "run": run_name,
    "adam_lr": metadata_args.get("adam_lr", args.adam_lr),
    "lr_multiplier": metadata_args.get("lr_multiplier", args.lr_multiplier),
    "warmup_ratio": metadata_args.get("warmup_ratio", args.warmup_ratio),
    "warmdown_ratio": metadata_args.get("warmdown_ratio", args.warmdown_ratio),
    "final_lr_frac": metadata_args.get("final_lr_frac", args.final_lr_frac),
    "max_grad_norm": metadata_args.get("max_grad_norm", args.max_grad_norm),
    "weight_decay": metadata_args.get("weight_decay", args.weight_decay),
    "num_epochs": metadata_args.get("num_epochs", args.num_epochs),
    "final_eval_size": final_eval_token_budget,
    "adaptive_stage0_epochs": metadata_args.get("adaptive_stage0_epochs", ADAPTIVE_STAGE0_EPOCHS),
    "adaptive_stage1_epochs": metadata_args.get("adaptive_stage1_epochs", ADAPTIVE_STAGE1_EPOCHS),
    "adaptive_loss_downscale": metadata_args.get("adaptive_loss_downscale", ADAPTIVE_LOSS_DOWNSCALE),
    "adaptive_stage2_downscale": metadata_args.get("adaptive_stage2_downscale", ADAPTIVE_STAGE2_DOWNSCALE),
    "stage1_group_size": metadata_args.get("stage1_group_size", STAGE1_GROUP_SIZE),
    "stage1_num_groups": metadata_args.get("stage1_num_groups", STAGE1_NUM_GROUPS),
    "stage1_total_masks": _stage1_total_masks_from_metadata(metadata_args),
    "stage1_mask_sampling": metadata_args.get("stage1_mask_sampling", STAGE1_MASK_SAMPLING),
    "stage1_mask_ratio_min": metadata_args.get("stage1_mask_ratio_min", STAGE1_MASK_RATIO_MIN),
    "stage1_mask_ratio_max": metadata_args.get("stage1_mask_ratio_max", STAGE1_MASK_RATIO_MAX),
    "stage1_right_window_size": metadata_args.get("stage1_right_window_size", STAGE1_RIGHT_WINDOW_SIZE),
    "stage1_full_seq": metadata_args.get("stage1_full_seq", STAGE1_FULL_SEQ),
    "stage1_full_seq_use_bidirectional": metadata_args.get("stage1_full_seq_use_bidirectional", STAGE1_FULL_SEQ_USE_BIDIRECTIONAL),
    "stage2_group_size": metadata_args.get("stage2_group_size", STAGE2_GROUP_SIZE),
    "stage2_num_groups": metadata_args.get("stage2_num_groups", STAGE2_NUM_GROUPS),
    "stage2_group_gap": metadata_args.get("stage2_group_gap", STAGE2_GROUP_GAP),
    "stage2_right_window_size": metadata_args.get("stage2_right_window_size", STAGE2_RIGHT_WINDOW_SIZE),
    "stage2_full_seq": metadata_args.get("stage2_full_seq", STAGE2_FULL_SEQ),
    "ltr_length": metadata_args.get("ltr_length", LTR_LENGTH),
    "do_bidirectional_eval": args.do_bidirectional_eval,
    "eval_right_window_size": args.eval_right_window_size,
    "effective_eval_right_window_size": effective_eval_right_window_size,
    "eval_crop_bucket": EVAL_CROP_BUCKET,
    "ntp_loss_downscale": metadata_args.get("ntp_loss_downscale", NTP_LOSS_DOWNSCALE),
    "ntp_loss_stage2_downscale": metadata_args.get("ntp_loss_stage2_downscale", NTP_LOSS_STAGE2_DOWNSCALE),
    "wandb_url": getattr(wandb_run, "url", None),
    "wandb_project": args.wandb_project,
    "wandb_group": args.wandb_group,
}
if args.eval_only_final:
    result_payload["load_final_checkpoint"] = args.load_final_checkpoint
if val_loss is not None:
    result_payload["val_loss"] = val_loss
if math.isfinite(min_val_loss):
    result_payload["best_val_loss"] = min_val_loss

optimizer = None
train_loader = None
adaptive_trainer = None
grad_accum_steps = None
num_iterations = None
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
if not args.eval_only_final:
    optimizer = model.setup_optimizer()
    train_loader = DataLoader(_train_path, args.device_batch_size, MAX_SEQ_LEN, device=device)
    _assert_loader_vocab_compatible(train_loader, config.vocab_size - 1, "training")
    TOKENS_PER_EPOCH = train_loader.total_tokens
    x, y, current_epoch = next(train_loader)
    adaptive_trainer = AdaptiveDreamTrainer(
        seq_len=MAX_SEQ_LEN,
        mask_token_id=mask_token_id,
        stage1_group_size=STAGE1_GROUP_SIZE,
        stage1_num_groups=STAGE1_NUM_GROUPS,
        stage1_mask_sampling=STAGE1_MASK_SAMPLING,
        stage1_mask_ratio_min=STAGE1_MASK_RATIO_MIN,
        stage1_mask_ratio_max=STAGE1_MASK_RATIO_MAX,
        stage2_group_size=STAGE2_GROUP_SIZE,
        stage2_num_groups=STAGE2_NUM_GROUPS,
        stage2_group_gap=STAGE2_GROUP_GAP,
        stage1_right_window_size=STAGE1_RIGHT_WINDOW_SIZE,
        stage2_right_window_size=STAGE2_RIGHT_WINDOW_SIZE,
        stage1_full_seq=STAGE1_FULL_SEQ,
        stage1_full_seq_use_bidirectional=STAGE1_FULL_SEQ_USE_BIDIRECTIONAL,
        stage2_full_seq=STAGE2_FULL_SEQ,
        stage0_epochs=ADAPTIVE_STAGE0_EPOCHS,
        total_epochs=args.num_epochs,
        ltr_length=LTR_LENGTH,
    )
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)  # estimate for LR schedule
    print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
    print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
else:
    print0("Eval-only final mode: skipping training loop and loading weights from checkpoint")
print0(f"Eval set: {default_eval_token_budget:,} tokens (requested {EVAL_TOKENS:,})")
print0(
    f"Final eval budget: {final_eval_token_budget:,} tokens "
    f"(requested {args.final_eval_size if args.final_eval_size is not None else EVAL_TOKENS:,})"
)

# Schedulers
def get_lr_scale(it):
    if num_iterations is None or num_iterations <= 0:
        return 1.0
    warmup = max(1, round(WARMUP_RATIO * num_iterations)) if WARMUP_RATIO > 0 else 0
    warmup = min(warmup, num_iterations)
    warmdown = max(1, round(WARMDOWN_RATIO * num_iterations)) if WARMDOWN_RATIO > 0 else 0
    warmdown = min(warmdown, num_iterations)
    if warmup > 0 and it < warmup:
        return (it + 1) / warmup
    if num_iterations <= warmup:
        return 1.0
    if warmdown <= 0 or it <= num_iterations - warmdown:
        return 1.0
    progress = (num_iterations - it) / warmdown
    progress = min(max(progress, 0.0), 1.0)
    return progress + (1.0 - progress) * FINAL_LR_FRAC

wall_clock_start = time.time()
if not args.eval_only_final:
    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb_token_budget(model, val_loader, default_eval_token_budget, token_bytes)
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    result_payload["val_loss"] = val_loss
    result_payload["best_val_loss"] = min_val_loss
    model.train()

    while current_epoch <= args.num_epochs:
        # Training step
        synchronize()
        t0 = time.time()
        micro_loss_sum = 0.0
        ntp_micro_loss_sum = 0.0
        ntp_micro_count = 0
        adaptive_stage1_micro_loss_sum = 0.0
        adaptive_stage1_micro_count = 0
        adaptive_stage2_micro_loss_sum = 0.0
        adaptive_stage2_micro_count = 0
        adaptive_micro_steps = 0
        stage0_micro_steps = 0
        stage1_micro_steps = 0
        stage2_micro_steps = 0
        for micro_step in range(grad_accum_steps):
            epoch_progress = (step * TOTAL_BATCH_SIZE + micro_step * tokens_per_fwdbwd) / TOKENS_PER_EPOCH
            if epoch_progress < ADAPTIVE_STAGE0_EPOCHS:
                stage = 0
                stage0_micro_steps += 1
                ntp_scale = NTP_LOSS_DOWNSCALE
            elif epoch_progress < ADAPTIVE_STAGE0_EPOCHS + ADAPTIVE_STAGE1_EPOCHS:
                stage = 1
                stage1_micro_steps += 1
                ntp_scale = NTP_LOSS_DOWNSCALE
            else:
                stage = 2
                stage2_micro_steps += 1
                ntp_scale = NTP_LOSS_STAGE2_DOWNSCALE

            raw_ntp_loss = None
            if ntp_scale > 0.0:
                with autocast_ctx:
                    ntp_loss = model(x, y)
                (ntp_loss * ntp_scale / grad_accum_steps).backward()
                raw_ntp_loss = ntp_loss.detach().item()
                ntp_micro_loss_sum += raw_ntp_loss
                ntp_micro_count += 1

            if stage == 1:
                adaptive_micro_steps += 1
                with autocast_ctx:
                    adaptive_loss = adaptive_trainer.compute_loss(
                        model, x, y, backward_scale=1.0 / grad_accum_steps
                    )
                raw_adaptive_loss = adaptive_loss.detach().item()
                if raw_ntp_loss is not None:
                    micro_loss_sum += raw_ntp_loss * ntp_scale
                micro_loss_sum += ADAPTIVE_LOSS_DOWNSCALE * raw_adaptive_loss
                adaptive_stage1_micro_loss_sum += raw_adaptive_loss
                adaptive_stage1_micro_count += 1
            elif stage == 2:
                stage2_aux_enabled = adaptive_trainer.stage2_num_groups > 0 and adaptive_trainer.stage2_group_size > 0
                if stage2_aux_enabled:
                    adaptive_micro_steps += 1
                    with autocast_ctx:
                        adaptive_loss = adaptive_trainer.compute_stage2_loss(
                            model, x, y, backward_scale=1.0 / grad_accum_steps
                        )
                    raw_adaptive_loss = adaptive_loss.detach().item()
                if raw_ntp_loss is not None:
                    micro_loss_sum += raw_ntp_loss * ntp_scale
                if stage2_aux_enabled:
                    micro_loss_sum += ADAPTIVE_STAGE2_DOWNSCALE * raw_adaptive_loss
                    adaptive_stage2_micro_loss_sum += raw_adaptive_loss
                    adaptive_stage2_micro_count += 1
            else:
                if raw_ntp_loss is not None:
                    micro_loss_sum += raw_ntp_loss * ntp_scale
            x, y, epoch = next(train_loader)

        # Update optimizer
        lrm = get_lr_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        grad_norm_stats = optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss_f = micro_loss_sum / grad_accum_steps
        raw_ntp_loss_avg = ntp_micro_loss_sum / ntp_micro_count if ntp_micro_count > 0 else None
        raw_stage1_adaptive_loss = (
            adaptive_stage1_micro_loss_sum / adaptive_stage1_micro_count
            if adaptive_stage1_micro_count > 0 else None
        )
        raw_stage2_mtp_loss = (
            adaptive_stage2_micro_loss_sum / adaptive_stage2_micro_count
            if adaptive_stage2_micro_count > 0 else None
        )
        total_adaptive_count = adaptive_stage1_micro_count + adaptive_stage2_micro_count
        raw_aux_loss = (
            (adaptive_stage1_micro_loss_sum + adaptive_stage2_micro_loss_sum) / total_adaptive_count
            if total_adaptive_count > 0 else None
        )
        synchronize()
        dt = time.time() - t0

        step += 1

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**step)
        pct = 100 * step / num_iterations
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
        if step > 3:
            total_training_time += dt
        steps_done = step - 3
        eta_str = f" | eta: {(num_iterations - step) * total_training_time / steps_done / 60:.1f}m" if steps_done > 0 else ""
        grad_norm_str = ""
        log_payload = {
            "step": step,
            "train/loss": debiased,
            "train/mfu": mfu,
            "train/adaptive_micro_frac": adaptive_micro_steps / grad_accum_steps,
            "train/stage0_frac": stage0_micro_steps / grad_accum_steps,
            "train/stage1_frac": stage1_micro_steps / grad_accum_steps,
            "train/stage2_frac": stage2_micro_steps / grad_accum_steps,
        }
        if raw_ntp_loss_avg is not None:
            log_payload["train/raw_loss_normal"] = raw_ntp_loss_avg
        if raw_aux_loss is not None:
            log_payload["train/raw_loss_aux"] = raw_aux_loss
        if raw_stage1_adaptive_loss is not None:
            log_payload["train/raw_loss_adaptive_stage1"] = raw_stage1_adaptive_loss
        if raw_stage2_mtp_loss is not None:
            log_payload["train/raw_loss_mtp_stage2"] = raw_stage2_mtp_loss
        if grad_norm_stats is not None:
            grad_norm_str = f" | grad_norm: {grad_norm_stats['grad_norm']:.4f}"
            log_payload.update({
                "train/grad_norm": grad_norm_stats["grad_norm"],
            })
        branch_str = (
            f" | adaptive_frac: {adaptive_micro_steps}/{grad_accum_steps}"
            f" | stage_counts(s0/s1/s2): {stage0_micro_steps}/{stage1_micro_steps}/{stage2_micro_steps}"
        )
        print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f}{grad_norm_str}{branch_str} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{eta_str}")
        wandb_run.log(log_payload)

        # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
        if ddp:
            epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
            dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
            epoch = epoch_tensor.item()

        # Epoch boundary: evaluate when the dataloader advances to a new epoch
        if epoch != current_epoch:
            model.eval()
            val_loader = build_val_loader()
            with autocast_ctx:
                val_bpb, val_loss = evaluate_bpb_token_budget(model, val_loader, default_eval_token_budget, token_bytes)
            print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
            wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss})
            result_payload["val_loss"] = val_loss
            # Early stopping
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
                min_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience >= 0 and epochs_without_improvement >= args.patience:
                    print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                    break
            result_payload["best_val_loss"] = min_val_loss
            model.train()
            current_epoch = epoch

        # GC management
        if step == 1:
            gc.collect(); gc.freeze(); gc.disable()

    final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
    checkpoint_summary = {
        "last_val_loss": val_loss,
        "best_val_bpb": min_val_bpb,
        "best_val_loss": min_val_loss,
        "final_train_loss": final_train_loss,
        "total_training_time": total_training_time,
    }
    result_payload["val_loss"] = val_loss
    result_payload["best_val_loss"] = min_val_loss
    if args.save_final_checkpoint:
        if master_process:
            save_final_checkpoint(
                args.save_final_checkpoint,
                orig_model,
                mask_token_id,
                run_name,
                step,
                current_epoch,
                extra_summary=checkpoint_summary,
            )
            result_payload["final_checkpoint_path"] = args.save_final_checkpoint
            persist_result_json(args.save_result, result_payload)
            print0(f"Saved final pre-eval checkpoint to {args.save_final_checkpoint}")
        if ddp:
            dist.barrier()
else:
    persist_result_json(args.save_result, result_payload)

# Final evaluation after training
model.eval()
orig_model.eval()
final_val_loader = build_val_loader()
with autocast_ctx:
    final_l2r_bpb, final_l2r_loss = evaluate_bpb_token_budget(
        model,
        final_val_loader,
        final_eval_token_budget,
        token_bytes,
    )
print0(f"Final Eval | L2R BPB: {final_l2r_bpb:.6f} | L2R Loss: {final_l2r_loss:.6f}")
wandb_run.log({"final_eval/l2r_bpb": final_l2r_bpb, "final_eval/l2r_loss": final_l2r_loss})
result_payload.update({"final_l2r_bpb": final_l2r_bpb, "final_l2r_loss": final_l2r_loss})
persist_result_json(args.save_result, result_payload)

final_bidirectional_loss = None
if args.do_bidirectional_eval:
    final_bidirectional_loader = build_val_loader()
    with autocast_ctx:
        final_bidirectional_loss = evaluate_bidirectional_loss_token_budget(
            orig_model,
            final_bidirectional_loader,
            final_eval_token_budget,
            mask_token_id,
            ltr_length=final_eval_ltr_length,
            eval_right_window_size=effective_eval_right_window_size,
            crop_bucket=EVAL_CROP_BUCKET,
        )
    print0(f"Final Eval | Bidirectional Loss: {final_bidirectional_loss:.6f}")
    wandb_run.log({"final_eval/bidirectional_loss": final_bidirectional_loss})
    result_payload.update({
        "final_bidirectional_loss": final_bidirectional_loss,
        "effective_eval_right_window_size": effective_eval_right_window_size,
        "eval_crop_bucket": EVAL_CROP_BUCKET,
    })
    persist_result_json(args.save_result, result_payload)

# Summary
wall_clock_time = time.time() - wall_clock_start
print0(f"Wall clock time: {wall_clock_time/60:.2f}m")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss
wandb_run.summary["final_l2r_loss"] = final_l2r_loss
if final_bidirectional_loss is not None:
    wandb_run.summary["final_bidirectional_loss"] = final_bidirectional_loss
result_payload.update({
    "final_train_loss": final_train_loss,
    "best_val_loss": min_val_loss,
    "final_l2r_loss": final_l2r_loss,
})
if final_bidirectional_loss is not None:
    result_payload["final_bidirectional_loss"] = final_bidirectional_loss
if val_loss is not None:
    result_payload["val_loss"] = val_loss
persist_result_json(args.save_result, result_payload)
if args.save_result and master_process:
    print0(f"Result saved to {args.save_result}")

total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
    
