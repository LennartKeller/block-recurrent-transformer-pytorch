import math
from random import random
from functools import wraps, partial
from collections import namedtuple
from packaging import version

from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, List, Tuple

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

def compact(arr):
    return [*filter(exists, arr)]

def and_reduce(arr: List[torch.Tensor]):
    if len(arr) == 0:
        return None
    head, *rest = arr
    for t in rest:
        head = head & t
    return head

print_once = once(print)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen):
        return self.compute_bias(qlen, klen)  # shape (1, num_heads, qlen, klen)
    
# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

        self.register_buffer('cached_freqs', None, persistent = False)
        self.register_buffer('cached_scales', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        device = self.device

        if exists(self.cached_freqs):
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        self.register_buffer('cached_freqs', freqs, persistent = False)
        self.register_buffer('cached_scales', scale, persistent = False)
        return freqs, scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, pos, scale = 1.):
    scale = default(scale, 1.)

    seq_len = t.shape[-2]

    assert pos.shape[-2] >= seq_len

    pos = pos[-seq_len:]

    if isinstance(scale, torch.Tensor):
        assert scale.shape[-2] >= seq_len
        scale = scale[-seq_len:]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

# maybe flash attention, if using pytorch 2.0

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# main class

class Attend(nn.Module):
    def __init__(
        self,
        causal = False,
        use_flash_attn = False,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.causal = causal
        self.ff_dropout = ff_dropout
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        masks = []

        if self.causal:
            i, j = q_len, k_len
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            masks.append(~causal_mask)

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = q.shape[0] // mask.shape[0])

            masks.append(mask)

        attn_mask = and_reduce(masks)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        # with torch.backends.cuda.sdp_kernel(**config._asdict()):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = attn_mask,
                dropout_p = self.ff_dropout
            )

        return out

    def forward(self, q, k, v, mask = None, use_flash_attn = None):
        use_flash_attn = default(use_flash_attn, self.use_flash_attn)

        b, n, device = q.shape[0], q.shape[-2], q.device

        q, ps = pack_one(q, '* h n d')
        k, _ = pack_one(k, '* n d')
        v, _ = pack_one(v, '* n d')

        if use_flash_attn:
            out = self.flash_attn(q, k, v, mask = mask)
            return unpack_one(out, ps, '* h n d')

        scale = q.shape[-1] ** -0.5

        k_einsum = 'b j d' if k.ndim == 3 else 'b h j d'
        v_einsum = 'b j d' if v.ndim == 3 else 'b h j d'

        # similarity

        sim = einsum(f"b h i d, {k_einsum} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = b)

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum(f"b h i j, {v_einsum} -> b h i d", attn, v)

        return unpack_one(out, ps, '* h n d')

# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(in_dim, out_dim, dropout_p=0.1):
    return nn.Sequential(
        LayerNorm(in_dim),
        nn.Dropout(dropout_p),
        nn.Linear(in_dim, out_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(out_dim, in_dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim_head,
        causal = False,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        use_flash_attn = False,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.causal = causal

        self.qk_rmsnorm = qk_rmsnorm
        self.qk_rmsnorm_scale = qk_rmsnorm_scale

        self.attend = Attend(causal = causal, use_flash_attn = use_flash_attn, ff_dropout = ff_dropout)

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(
        self,
        q, k, v,
        mask = None,
        rotary_pos_emb = None,
        xpos_scale = None,
        position_bias = None,
    ):

        scale = q.shape[-1] ** -0.5

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            scale = self.qk_rmsnorm_scale

        if self.qk_rmsnorm:
            q = q * self.q_scale
            k = k * self.k_scale

        # rotary positional embedding with xpos for length extrapolation

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb, xpos_scale)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, xpos_scale ** -1)
        
        if exists(position_bias):
            # Mask shape before : [n_batches, 1, qlen, klen]
            mask = position_bias + mask
            # Mask shape before : [n_batches, n_heads, qlen, klen]

        # attention

        out = self.attend(q, k, v, mask = mask)

        return out


class FixedStateGate(nn.Module):
    """
    Fixed gate to update to states.
    Simply wraps the mimics the original parameters and logic.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.state_out_to_gate = nn.Linear(dim, dim)
        self.learned_ema_bias = nn.Parameter(torch.randn(dim))
    
    def forward(self, orig_states: torch.Tensor, state_out: torch.Tensor) -> torch.Tensor:
        z = self.state_out_to_gate(state_out)
        learned_ema_decay = self.learned_ema_bias.sigmoid()
        new_states = learned_ema_decay * z + (1 - learned_ema_decay) * orig_states
        return new_states


class Lambda(nn.Module):
    
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class LSTMStyleGate(nn.Module):
    """
    LSTM-style gate as proposed in the original paper.
    Judging from brief experiments (i.e. train 1.5k steps w. bs 2),
    this gate performs considerably worse.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.state_out_to_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
        self.input_gate = nn.Sequential(
            nn.Linear(dim, dim),
            Lambda(lambda x: x - 1),
            nn.Sigmoid()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(dim, dim),
            Lambda(lambda x: x + 1),
            nn.Sigmoid()
        )

        self._init_weights(dim)
    
    def forward(self, orig_states: torch.Tensor, state_out: torch.Tensor) -> torch.Tensor:
        # State_out: projected current states, orig_states: states from prev. segment
        z = self.state_out_to_gate(state_out)
        # Constants should enforce the model to use the memory-
        input_mask = self.input_gate(state_out)
        forget_mask = self.forget_gate(state_out)
        new_states = (orig_states * forget_mask) + (z * input_mask)
        return new_states

    def _init_weights(self, dim: int):
        input_linear_layer = self.input_gate[0]
        forget_linear_layer = self.forget_gate[0]
        # Init biases
        bias_mean, bias_std = 1, 0.1
        nn.init.normal_(input_linear_layer.bias, mean=bias_mean, std=bias_std)
        nn.init.normal_(forget_linear_layer.bias, mean=bias_mean, std=bias_std)

        # projection weights
        weight_std = math.sqrt(0.1 / dim)
        nn.init.trunc_normal_(forget_linear_layer.weight, std=weight_std)
        nn.init.trunc_normal_(input_linear_layer.weight, std=weight_std)





class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        block_width,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        num_state_vectors = 0,
        use_flash_attn = False,
        gate_type = None,
        ff_dropout = 0.1
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        
        self.dropout = nn.Dropout(ff_dropout)

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        self.block_width = block_width
        self.is_recurrent_layer = num_state_vectors > 0

        self.to_out = nn.Linear(inner_dim * (2 if self.is_recurrent_layer else 1), dim, bias = False)

        if not self.is_recurrent_layer:
            return

        self.state_norm = LayerNorm(dim)

        self.q_to_state = nn.Linear(dim, inner_dim, bias = False)
        self.q_from_state = nn.Linear(dim, inner_dim, bias = False)

        self.state_to_q = nn.Linear(dim, inner_dim, bias = False)
        self.state_to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
        self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))

        self.to_state_out = nn.Linear(inner_dim * 2, dim, bias = False)

        self.to_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        self.state_self_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)
        self.from_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        # gating related parameters - using the fixed simple config

        # self.state_out_to_gate = nn.Linear(dim, dim)
        # self.learned_ema_beta = nn.Parameter(torch.randn(dim))
        if gate_type is None or gate_type.lower() == "fixed":
            self.state_gate = FixedStateGate(dim)
        elif gate_type.lower() == "lstm":
            self.state_gate = LSTMStyleGate(dim)
        else:
            raise ValueError(f"Invalid value {gate_type} for param 'gate_type'. Chose either 'fixed' or 'lstm'.")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        xpos_scale = None,
        position_bias = None,
        attn_mask = None,
        return_memories_and_states = None,
        xl_memories: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None
    ):
        batch, seq_len, _, width, device = *x.shape, self.block_width, x.device

        # first make sure to pad the sequence length to multiple of the block widths
        # for local attention

        if not divisible_by(seq_len, width):
            padding_to_width_multiple = math.ceil(seq_len / width) * width - seq_len
            x = pad_at_dim(x, (0, padding_to_width_multiple), dim = -2, value = 0)

        # pre normalization

        x = self.norm(x)

        # queries, keys, values and split out heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        split_head = partial(rearrange, pattern = 'b n (h d) -> b h n d', h = self.heads)
        q = split_head(q)

        # bucket the queries, keys, values by block width

        bq, bk, bv = map(lambda t: rearrange(t, 'b ... (w n) d -> b w ... n d', n = width), (q, k, v))

        # save the last key / values as memories for recurrence

        memories = None

        if return_memories_and_states:
            memories = torch.stack((bk[:, -1], bv[:, -1]))

        if exists(xl_memories):
            # if past memories are passed in, concat as the first bucket
            past_k, past_v = xl_memories
            past_k, past_v = map(lambda t: rearrange(t, 'b ... n d -> b 1 ... n d'), (past_k, past_v))
            bk = torch.cat((past_k, bk), dim = 1)
            bv = torch.cat((past_v, bv), dim = 1)
        else:
            # otherwise add padding
            bk = pad_at_dim(bk, (1, 0), value = 0., dim = 1)
            bv = pad_at_dim(bv, (1, 0), value = 0., dim = 1)

        # alter attention mask so that the first window looking back would either attend to nothing (in the case of padding), or everything (in the case of xl memories)

        if exists(attn_mask):
            attn_mask = repeat(attn_mask, 'i j -> w 1 i j', w = bq.shape[1])
            attn_mask[0, 0, :, :width] = exists(xl_memories)

        # local attention with look back of one bucket - in paper they did total receptive field of 2 * block_width, with 1 block_width worth of memories, seems like a more efficient transformer-xl design?

        bk = torch.cat((bk[:, :-1], bk[:, 1:]), dim = -2)
        bv = torch.cat((bv[:, :-1], bv[:, 1:]), dim = -2)

        # attention, but of course

        out = self.attn(
            bq, bk, bv,
            rotary_pos_emb = rotary_pos_emb,
            xpos_scale = xpos_scale,
            position_bias = position_bias,
            mask = attn_mask
        )

        # merge the heads as well as the buckets

        out = rearrange(out, 'b w h n d -> b (w n) (h d)')

        # in case there is padding during sampling, splice it out

        out = out[:, :seq_len]

        new_states = None

        # early return if not a recurrent layer

        if not self.is_recurrent_layer:
            out = self.to_out(out)
            out = self.dropout(out)
            return out, memories, new_states

        # if designated a recurrent layer, do all the state logic
        # it was hard moving this to a separate module, as the attention is closely intertwined between the current tokens and state tokens

        # process input in blocks

        x_blocks, k_blocks, v_blocks = map(lambda t: t[:, :seq_len].split(width, dim = -2), (x, k, v))

        # ready attended output of the input to the state, concatted block by block

        to_state_out = torch.empty((batch, 0, out.shape[-1]), device = device, dtype = out.dtype)

        # use initial state if no states were passed in

        if not exists(states):
            states = self.init_state

        for ind, (x_block, xk_block, xv_block) in enumerate(zip(x_blocks, k_blocks, v_blocks)):
            is_last = ind == (len(x_blocks) - 1)

            residual_states = states

            # pre norm state for attention

            states = self.state_norm(states)

            # add the positional ids, as stated in the paper critical for it to work

            states = states + self.state_pos_ids

            # get queries for cross attention, which they do not share, although they share key / values. another intriguing detail

            q_to_state = self.q_to_state(x_block)
            q_from_state = self.q_from_state(states)

            q_to_state, q_from_state = map(lambda t: rearrange(t, '... n (h d) -> ... h n d', h = self.heads), (q_to_state, q_from_state))

            # self attention qkv for states

            state_q, state_k, state_v = (self.state_to_q(states), *self.state_to_kv(states).chunk(2, dim = -1))

            state_q_einsum = 'n (h d)' if state_q.ndim == 2 else 'b n (h d)'
            state_q = repeat(state_q, f'{state_q_einsum} -> b h n d', h = self.heads, b = batch)

            # cross attend to the past states key values

            to_state_out_block = self.to_state_cross_attn(q_to_state, state_k, state_v)

            to_state_out_block = rearrange(to_state_out_block, 'b h n d -> b n (h d)')

            to_state_out = torch.cat((to_state_out, to_state_out_block), dim = -2)

            # if need to return states, or is not the last block, calculate state update

            if return_memories_and_states or not is_last:

                # states must also undergo self attention

                if q_from_state.ndim == 3:
                    q_from_state = repeat(q_from_state, '... -> b ...', b = batch)

                state_out = self.state_self_attn(state_q, state_k, state_v)

                from_state_out = self.from_state_cross_attn(q_from_state, xk_block, xv_block)

                state_out = torch.cat((state_out, from_state_out), dim = -1)
                state_out = rearrange(state_out, 'b h n d -> b n (h d)')

                state_out = self.to_state_out(state_out)

                states = self.state_gate(residual_states, state_out)

                # use the best performing configuration
                # fixed simple gate - nothing more than a learned EMA with some resemblance to highway networks

                # z = self.state_out_to_gate(state_out)
                # learned_ema_decay = self.learned_ema_beta.sigmoid()

                # # set new state with the learned EMA gating

                # states = learned_ema_decay * z + (1 - learned_ema_decay) * residual_states                

        # concat the output of cross attending to the state vectors

        out = torch.cat((out, to_state_out), dim = -1)

        if return_memories_and_states:
            new_states = states
        
        out = self.to_out(out)
        out = self.dropout(out)
        
        return out, memories, new_states

# classes

#@beartype
class BlockRecurrentTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        intermediate_dim = 4096,
        heads = 8,
        all_layers_qk_rmsnorm = False,
        max_seq_len = 1024,
        block_width = 512,
        xl_memories_layers: Optional[Tuple[int, ...]] = None,
        recurrent_layers: Optional[Tuple[int, ...]] = None,
        num_state_vectors = None,
        enhanced_recurrence = False,
        ignore_index = -100,
        use_flash_attn = False,
        gate_type = None,
        position_encoding_type = None,
        ff_dropout = 0.1
    ):
        super().__init__()
        num_state_vectors = default(num_state_vectors, block_width)
        xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        self.xl_memories_layers = set(xl_memories_layers)

        assert all([0 < layer <= depth for layer in xl_memories_layers])

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network
        self.recurrent_layers = set(recurrent_layers)

        assert all([0 < layer <= depth for layer in recurrent_layers]), ([layer for layer in recurrent_layers], depth)

        self.token_emb = nn.Embedding(num_tokens, dim)

        if position_encoding_type is None or position_encoding_type.lower() == "rotary":
            self.position_encoding = RotaryEmbedding(dim = dim_head)
        elif position_encoding_type.lower() == "rel_bias":
            self.position_encoding = RelativePositionBias(n_heads=heads)
        else:
            raise ValueError(f"Invalid value {position_encoding_type} for argument 'position_encoding_type'. Chose either 'rotary' or 'rel_bias'")
        self.position_encoding_type = position_encoding_type.lower() if position_encoding_type is not None else None

        self.dropout = nn.Dropout(ff_dropout)

        self.layers = nn.ModuleList([])

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers
            is_xl_layer = layer in self.xl_memories_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            # only layers with xl memories
            # or has recurrence in horizontal direction
            # use qk rmsnorm (in paper, they use cosine sim attention, but i think qk rmsnorm is more proven given Vit 22B paper)
            # one can also override to use all qk rmsnorm by setting all_layers_qk_rmsnorm = True

            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer or is_xl_layer

            self.layers.append(nn.ModuleList([
                AttentionBlock(
                    dim,
                    block_width = block_width,
                    dim_head = dim_head,
                    heads = heads,
                    qk_rmsnorm = qk_rmsnorm,
                    num_state_vectors = layer_num_state_vectors,
                    use_flash_attn = use_flash_attn,
                    gate_type=gate_type
                ),
                FeedForward(in_dim=dim, out_dim=intermediate_dim, dropout_p=ff_dropout)
            ]))

        self.to_embs = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim, bias = False),
            nn.Tanh()
        )

        self.max_seq_len = max_seq_len
        self.block_width = block_width

        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index

        self.enhanced_recurrence = enhanced_recurrence

        self.register_buffer('cached_attn_mask', None, persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_noncausal_attn_mask(self, width, device=None):
        # Hack to get multi-gpu setup to work
        if device is None:
            device = self.device
        attn_mask = torch.ones((width, 2* width), device=device, dtype=torch.long).bool()
        return attn_mask

    def forward(
        self,
        x,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        return_memories_and_states = None  # can force to either return memory + state or not. by default will only return when number of tokens == max_seq_len
    ):
        device = x.device

        # get sequence length i and j for dynamic pos bias

        assert x.shape[-1] <= self.max_seq_len

        w = self.block_width

        # token embedding

        x = self.token_emb(x)
        x = self.dropout(x)

        # dynamic pos bias

        attn_mask = self.get_noncausal_attn_mask(w, device=device)
        rotary_pos_emb, xpos_scale, position_bias = None, None, None
        if self.position_encoding_type is None or self.position_encoding_type == "rotary":
            rotary_pos_emb, xpos_scale = self.position_encoding(2 * w)
        else:
            position_bias = self.position_encoding(qlen=w, klen=2*w)

        # enhanced recurrence

        if self.enhanced_recurrence and len(xl_memories) > 1:
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # ready xl memories and states

        xl_memories = iter(xl_memories)
        states = iter(states)

        next_xl_memories = []
        next_states = []

        return_memories_and_states = default(return_memories_and_states, self.max_seq_len == x.shape[-2])

        # go through layers

        for ind, (attn, ff) in enumerate(self.layers):

            # determine if the layer requires transformer xl memories

            layer = ind + 1

            is_xl_layer     = layer in self.xl_memories_layers
            is_state_layer  = attn.is_recurrent_layer

            # whether to pass in xl memories

            attn_kwargs = dict(
                rotary_pos_emb = rotary_pos_emb,
                xpos_scale = xpos_scale,
                attn_mask = attn_mask,
                position_bias = position_bias,
                return_memories_and_states = return_memories_and_states
            )

            if is_xl_layer:
                attn_kwargs.update(xl_memories = next(xl_memories, None))

            if is_state_layer:
                attn_kwargs.update(states = next(states, None))

            # attention layer

            residual = x
            attn_branch_out, layer_xl_memories, layer_next_states = attn(x, **attn_kwargs)

            if return_memories_and_states:
                # save states if needed

                if exists(layer_next_states):
                    next_states.append(layer_next_states.detach())

                # save current xl memories if needed

                if is_xl_layer:
                    next_xl_memories.append(layer_xl_memories.detach())

            x = attn_branch_out + residual

            # feedforward layer

            x = ff(x) + x

        embs = self.to_embs(x)
        return embs, next_xl_memories, next_states
