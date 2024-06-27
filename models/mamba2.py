import math

import jax
import jax.nn as nn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random

from jax import vmap, jit
from einops import rearrange, repeat, einsum

from functools import partial
from dataclasses import dataclass
from typing import Tuple, NamedTuple
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker 


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    d_head: int = 128
    A_init_range: Tuple[int, int] = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        self.n_heads = self.d_inner // self.d_head

        # order z, x, B, C, dt
        self.d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.n_heads
        # x, B, C
        self.conv_dim = self.d_inner + 2 * self.d_state

        self.orig_vocab_size = self.vocab_size

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple


class LayerParams(NamedTuple):
    norm: Array
    norm_y: Array
    in_proj: Array
    in_proj_bias: Array | None 
    conv: Array
    conv_bias: Array | None 
    dt_bias: Array
    A_log: Array
    D: Array
    out_proj: Array
    out_proj_bias: Array | None


class Mamba2Params(NamedTuple):
    embedding: Array
    layers: LayerParams
    norm_f: Array


def initialize_params(key, args):
    truncated_normal_stddev = .87962566103423978
     
    A_min, A_max = args.A_init_range
    assert A_min > 0 and A_max >= A_min

    d_model_scale = 1 / (math.sqrt(args.d_model) * truncated_normal_stddev)
    d_inner_scale = 1 / (math.sqrt(args.d_inner) * truncated_normal_stddev)
    conv_dim_scale = 1 / (math.sqrt(args.conv_dim) * truncated_normal_stddev)

    embed_key, layers_key = random.split(key)
    layers_keys = random.split(layers_key, 5)

    embedding = random.truncated_normal(embed_key, -2, 2, (args.vocab_size, args.d_model)) * d_model_scale

    norm = jnp.ones((args.n_layer, args.d_model))

    norm_y = jnp.ones((args.n_layer, args.d_inner))
    
    in_proj = random.truncated_normal(layers_keys[0], -2, 2, (args.n_layer, args.d_model, args.d_in_proj)) * d_model_scale
    in_proj_bias = jnp.zeros((args.n_layer, args.d_in_proj)) if args.bias else None
    
    conv = random.truncated_normal(layers_keys[1], -2, 2, (args.n_layer, args.conv_dim, args.d_conv)) * conv_dim_scale
    conv_bias = jnp.zeros((args.n_layer, args.conv_dim)) if args.conv_bias else None
    
    dt = random.uniform(layers_keys[2], (args.n_layer, args.n_heads))
    dt = jnp.exp(
        dt * (math.log(args.dt_max) - math.log(args.dt_min)) + math.log(args.dt_min)
    ).clip(min=args.dt_init_floor)
    dt_bias = dt + jnp.log(-jnp.expm1(-dt))
    
    A_log = jnp.log(random.uniform(layers_keys[3], (args.n_layer, args.n_heads), minval=A_min, maxval=A_max))
    D = jnp.ones((args.n_layer, args.n_heads))
    
    out_proj = random.truncated_normal(layers_keys[4], -2, 2, (args.n_layer, args.d_inner, args.d_model)) * d_inner_scale
    out_proj_bias = jnp.zeros((args.n_layer, args.d_model)) if args.bias else None
    
    layers = LayerParams(
        norm=norm,
        norm_y=norm_y,
        in_proj=in_proj,
        in_proj_bias=in_proj_bias,
        conv=conv,
        conv_bias=conv_bias,
        dt_bias=dt_bias,
        A_log=A_log,
        D=D,
        out_proj=out_proj,
        out_proj_bias=out_proj_bias
    )

    norm_f = jnp.ones(args.d_model)

    return Mamba2Params(embedding=embedding, layers=layers, norm_f=norm_f)


def zero_or(x):
    return 0 if x is None else x


@jaxtyped(typechecker=typechecker)
def rms_norm_gated(
        w: Float[Array, 'd'],
        x: Float[Array, '*b d'],
        z: Float[Array, '*b d'] | None = None,
        eps=1e-5
    ) -> Float[Array, '*b d']:

    if z is not None:
        x = x * nn.silu(z)

    y = x.astype(jnp.float32)
    norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + eps)

    return w * norm.astype(x.dtype)


@jaxtyped(typechecker=typechecker)
def segsum(x: Float[Array, '*b l']) -> Float[Array, '*b l l']:
    """
    we need to compute the 1-SS matrix which for a sequence [ x_1, ..., x_l ]
    is a lower triangular matrix M of shape (n, n), with diagonal elements 1
    and for all 2 <= i <= l and 1 <= j < i, M_ij = x_1 * ... * x_{i - j}

    this causes numerical issues for large l, so we compute the cumulative sums
    by taking the log of x. this is assumed to be already done, before calling
    the function. we exp the output of this function to get the 1-SS matrix

    if x is the sequence we want to compute the 1-SS matrix for, we do

    M = jnp.exp(segsum(jnp.log(z)))
    """

    l = x.shape[-1]

    x = jnp.repeat(x[..., None], l, axis=-1) 
    mask = jnp.tril(jnp.ones((l, l), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)

    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((l, l), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    
    return x_segsum


@jaxtyped(typechecker=typechecker)
def ssd(
        dt: Float[Array, 'l nh'],
        x: Float[Array, 'l d'],
        A: Float[Array, 'nh'],
        B: Float[Array, 'l n'],
        C: Float[Array, 'l n'],
        D: Float[Array, 'nh']
    ) -> Float[Array, 'l d']:
    """
    tl;dr:

    we choose a structure for the SSM matrices that allow us to turn it into an attention
    analogue. the end result looks like removing the softmax in attention and using a special
    mask that allows for fast parallel training with quadratic complexity (like transformers)
    and fast inference with linear complexity (like RNNs)

    Args:
        dt: the step size for each position for each head 
        x: input tensor split into nh channel groups (heads)
        A: the state matrices. the matrix is a scalar for each head shared across positions
        B: the input matrices. the matrix is a vector for each position shared across heads
        C: the output matrices. the matrix is a vector for each position shared across heads
        D: the feedforward matrices. the matrix is a scalar for each head shared across positions
    
    consider the SSM for a single channel of an embedding sequence. it is a sequence to sequence 
    map (l,) -> (l,) using structured matrices A, B, and C of shapes (l,), (l, n), and (l, n)
    respectively where n is d_state. the 1-SS matrix M of shape (l, l) transforms the loop
    
    y = []
    h = jnp.zeros(n)
    for a_t, B_t, C_t in zip(A, B, C):
        h = a_t * h + B_t * x_t  # () * (n,) + (n,) * () = (n,)
        y.append(h @ C_t.T)  # (n,) @ (n,).T = ()
    y = jnp.array(y)  # (l,)

    into a single matrix multiplication

    y = M @ x  # (l, l) @ (l,) = (l,)

    we see below how. the tensors involved are
    
    x_t: ()
    a_t: ()
    B_t: (n,)  
    C_t: (n,)

    the recurrence as seen above is

    H_t = a_t * H_{t - 1} + B_t * x_t  # (n,)
    y_t = H_t @ C_t^T  # ()

    where a_t, B_t, and C_t are shared across channels and are discretized before computing the recurrence

    we can expand the recurrence to see a pattern

    H_{-1} = 0  # (n,)
    
    H_0 = a_0 * H_{-1} + B_0 * x_0
    ...
    H_{l - 1} = a_{l - 1} * H_{l - 2} + B_{l - 1} * x_{l - 1}  # l hidden states
    
    expanding from H_{l - 1}

    H_{l - 1} = a_{l - 1} * (a_{l - 2} * H_{l - 3} + B_{l - 2} * x_{l - 2}) + B_{l - 1} * x_{l - 1}
              = (a_{l - 1} * a_{l - 2}) * H_{l - 3} + a_{l - 1} * B_{l - 2} * x_{l - 2} + B_{l - 1} * x_{l - 1}

    expanding until we get to H_{-1}, we have 
              
    H_{l - 1} = (a_{l - 1} * ... * a_0) * H_{-1} + (a_{l - 1} * ... * a_1) * B_0 * x_0
                                                 + (a_{l - 1} * ... * a_2) * B_1 * x_1
                                                 ...
                                                 + (a_{l - 1} * a_{l - 2}) * B_{l - 3} * x_{l - 3}
                                                 + a_{l - 1} * B_{l - 2} * x_{l - 2}
                                                 + B_{l - 1} * x_{l - 1}
    
    then we do the read out
    
    y_l = H_{l - 1} @ C_{l - 1}^T

    since H_{-1} = 0 we have

    y_l = (a_{l - 1} * ... * a_1) * C_{l - 1} @ B_0^T * x_0 +
          (a_{l - 1} * ... * a_2) * C_{l - 1} @ B_1^T * x_1 +
          ...
          (a_{l - 1} * a_{l - 2}) * C_{l - 1} @ B_{l - 3}^T * x_{l - 3} +
          a_{l - 1} * C_{l - 1} @ B_{l - 2}^T * x_{l - 2} +
          C_{l - 1} @ B_{l - 1}^T * x_{l - 1}

    for example

    H_3 = a_3 * a_2 * a_1 * (C_3 @ B_0^T) * x_0 + 
          a_3 * a_2 * (C_3 @ B_1^T) * x_1 +
          a_3 * (C_3 @ B_2^T) * x_2 + 
          (C_3 @ B_3^T) * x_3

    if you look carefully, you can see that it is equivalent to a masked and weighted query-key dot product
    where C is analogous Q and B is analogous to K. the mask for y_t is 

                                0                                           t - 1    t        l - 1
                                |                                             |      |          |       
    MASK_{t - 1} = [ (a_1 * ... * a_{t - 1}), ..., a_{t - 2} * a_{t - 1}, a_{t - 1}, 1, 0, ..., 0 ]

    let L = [ MASK_0,
              MASK_1,
              ...,
              MASK_{l - 1} ]   # weighted mask
     
          = [ [ 1, 0, 0, ..., 0 ],
              [ a_1, 1, 0, ..., 0 ],
              [ a_1 * a_2, a_1, 1, ..., 0 ],
              ...
              [ (a_1 * ... * a_{l - 1}), ..., 1 ] ]  # (l, l)

    L is the 1-semiseparable (1-SS) matrix

    the SSM matrix is now

    M = L * (C @ B^T)  # (l, l) * ((l, n) @ (l, n).T) = (l, l)

    now given a sequence x of shape (l,)

    y = M @ x

    is the SSM output

    if x is of shape (l, dh) (multiple channels), the same equation applies
    """
    
    # split heads
    x = rearrange(x, 'l (nh dh) -> nh l dh', nh=A.shape[-1])

    # discretization
    xdt = einsum(x, dt, 'nh l dh, l nh -> nh l dh')  
    Adt = -jnp.exp(A) * dt
    
    L = jnp.exp(segsum(Adt.T))  # weighted mask
    M = L * (C @ B.T)  # scores = mask * (query @ key.T)
    y = M @ xdt + einsum(x, D, 'nh l dh, nh -> nh l dh')  # out = scores @ values 

    # combine heads
    return rearrange(y, 'nh l dh -> l (nh dh)')


def mamba2(args, params, tokens):

    def block(x, params):
        # (seq_len, d_model) -> (seq_len, d_inner + conv_dim + n_heads)
        zxBCdt = x @ params.in_proj + zero_or(params.in_proj_bias)

        # (seq_len, d_inner), (seq_len, conv_dim), (seq_len, n_heads)
        z, xBC, dt = jnp.split(zxBCdt, [args.d_inner, args.d_inner + args.conv_dim], -1)
        
        # (seq_len + d_conv - 1, conv_dim) -> (conv_dim, seq_len + d_conv - 1)
        xBC = jnp.concatenate([jnp.zeros((args.d_conv - 1, args.conv_dim)), xBC], 0).T
        # (conv_dim, seq_len + d_conv - 1) -> (conv_dim, seq_len) -> (seq_len, conv_dim)
        xBC = vmap(jnp.convolve, (0, 0, None))(xBC, params.conv, 'valid').T + zero_or(params.conv_bias)
        xBC = nn.silu(xBC)

        # (seq_len, conv_dim) -> (seq_len, d_inner), (seq_len, d_state), (seq_len, d_state)
        x, B, C = jnp.split(xBC, [args.d_inner, args.d_inner + args.d_state], -1)

        dt = nn.softplus(dt + params.dt_bias)  # (seq_len, n_heads)
        
        # SSM
        y = ssd(dt, x, params.A_log, B, C, params.D)
        
        # normalization
        y = rms_norm_gated(params.norm_y, y, z)

        # (l, d_inner) -> (l, d_model) 
        return y @ params.out_proj + zero_or(params.out_proj_bias)

    def f(x, params):
        return x + block(rms_norm_gated(params.norm, x, None), params), None

    x = params.embedding[tokens]

    x, _ = lax.scan(f, x, params.layers)

    logits = rms_norm_gated(params.norm_f, x, None) @ params.embedding.T

    return logits


def mamba2_step(args, valid_logits, params, token, cache):

    def block(x, params, conv_cache, ssm_state): 
        # (d_model,) -> (d_inner + conv_dim + n_heads,)
        zxBCdt = x @ params.in_proj + zero_or(params.in_proj_bias)
        
        # (d_inner,), (conv_dim,), (n_heads,)
        z, xBC, dt = jnp.split(zxBCdt, [args.d_inner, args.d_inner + args.conv_dim], -1)
        
        conv_input = jnp.concatenate([conv_cache, xBC[:, None]], -1)  # (conv_dim, d_conv)
        kernel = jnp.flip(params.conv, -1)  # (conv_dim, d_conv)
        xBC = nn.silu(jnp.vecdot(conv_input, kernel) + zero_or(params.conv_bias))
        
        x, B, C = jnp.split(xBC, [args.d_inner, args.d_inner + args.d_state], -1)

        # split heads
        x = rearrange(x, '(nh dh) -> nh dh', nh=args.n_heads)
        
        dt = nn.softplus(dt + params.dt_bias)

        # discretization
        xdt = einsum(x, dt, 'nh dh, nh -> nh dh')
        Adt_log = -jnp.exp(params.A_log) * dt
        Adt = jnp.exp(Adt_log)
        
        # SSM step
        ssm_state = einsum(Adt, ssm_state, 'nh, nh dh n -> nh dh n') + einsum(B, xdt, 'n, nh dh -> nh dh n')
        y = ssm_state @ C.T + einsum(x, params.D, 'nh dh, nh -> nh dh')
        
        # combine heads
        y = rearrange(y, 'nh dh -> (nh dh)') 
        
        y = rms_norm_gated(params.norm_y, y, z)

        y = y @ params.out_proj + zero_or(params.out_proj_bias)

        return y, (conv_input[:, 1:], ssm_state)

    def f(x, params_and_cache):
        params, cache = params_and_cache
        h, cache = block(rms_norm_gated(params.norm, x, None), params, *cache)
        return x + h, cache
    
    x = params.embedding[token]

    if cache is None:
        cache = (
            jnp.zeros((args.n_layer, args.conv_dim, args.d_conv - 1)),
            jnp.zeros((args.n_layer, args.n_heads, args.d_head, args.d_state))
        )

    x, cache = lax.scan(f, x, (params.layers, cache))

    logits = rms_norm_gated(params.norm_f, x, None) @ params.embedding.T

    return logits[:args.orig_vocab_size if valid_logits else args.vocab_size], cache
