import torch as tn
import numpy as np
from typing import Optional, Tuple, Union, List, Sequence, Any
import math
from pysp.torch_stats.pdist import TorchEncodedSequence

DINT = tn.int32
DFLOAT = tn.float64


def seed_tng(seed: int, device: Optional[tn.device] = None):
    return tn.Generator(device=device).manual_seed(int(seed))


def seed_sample(n: int, tng: tn.Generator):
    if n == 1:
        return tn.randint(0, 2**31, size=(n,), generator=tng, dtype=DINT)[0]
    else:
        return tn.randint(0, 2**31, size=(n,), generator=tng, dtype=DINT)


def zeros(size: Union[int, Tuple[int, ...]], device: Optional[tn.device] = None, dtype: Optional[tn.dtype] = DFLOAT):
    return tn.zeros(size, dtype=dtype, device=device)


def ones(size: Union[int, Tuple[int, ...]], device: Optional[tn.device] = None, dtype: Optional[tn.dtype] = DFLOAT):
    return tn.ones(size, dtype=dtype, device=device)


def int_vec(size: Union[int, Tuple[int, ...]], device: Optional[tn.device] = None):
    return tn.zeros(size, dtype=DINT, device=device)


def zeros_like(x: tn.Tensor):
    return tn.zeros_like(x, dtype=DFLOAT)


def tensor(x: Union[Union[List[int], List[float]], tn.Tensor, List[List[int]], List[List[float]], np.ndarray], device: Optional[tn.device] = None, dtype: Optional[tn.dtype] = DFLOAT):
    if isinstance(x, tn.Tensor):
        y = x.clone().detach().to(dtype)
        y.to(device)

        return y

    else:
        return tn.tensor(x, device=device, dtype=dtype)


def int_tensor(x: Union[Union[Sequence[int], List[List[int]], np.ndarray, tn.Tensor]], device: Optional[tn.device] = None, dtype: Optional[tn.dtype] = DINT):
    return tensor(x, device=device, dtype=dtype)


def choice(size: int, states: int, tng: tn.Generator, p: Optional[tn.Tensor] = None) -> tn.Tensor:

    device = tng.device
    if p is not None:
        alpha = p if p.device == device else p.to(device)
    else:
        alpha = ones(states, device=device)

    return tn.multinomial(alpha, num_samples=size, replacement=True, generator=tng)


def randint(size: int, low: int, high: int, tng: tn.Generator, dtype: Optional[tn.dtype] = DINT):
    return tn.randint(size=(size, ), low=low, high=high, generator=tng, dtype=dtype, device=tng.device)


def bincount_2d(x: tn.Tensor, w: tn.Tensor, minlength: int) -> tn.Tensor:
    """Bincount 2-d Tensor by index x.

    Args:
        x (Tensor): tensor of ints corresponding to id of row in w.
        w (Tensor): A 2-d (s by n) tensor with len(x) == n and s-states.
        minlength (int): Number of values to count by.

    Returns:
        Tensor with dim (minlength by s).

    """
    rv = zeros((w.shape[0], minlength), device=w.device)
    for i in range(w.shape[0]):
        rv += tn.bincount(x, w[i], minlength=minlength)

    return rv


def sample_dirichlet(alpha: tn.Tensor, size: int, tng: tn.Generator) -> tn.Tensor:
    """Sample from a Dirichlet distribution with shape alpha.

    Args:
        alpha (tn.Tensor): Shape parameter length equal to dim of simplex.
        size (int): Number of samples to draw.
        tng (Generator): Generator for sampling with device set.

    Returns:
        Tensor of shape (size X len(alpha)) on device same as tng.

    """
    k = alpha.shape[0]
    rv = tn._sample_dirichlet(alpha.expand((size, k)), generator=tng)
    rv = tensor(rv, device=tng.device)

    return rv



# def mixture_weights(
#         k: int,
#         tng: tn.Generator,
#         alpha: Optional[float] = None,
#         size: Optional[Union[int, tn.Tensor]] = None,
# ) -> tn.Tensor:
#
#     device = tng.device if tng.device is not None else tn.device('cpu')
#
#     alpha = 1 / k ** 2 if alpha is None else alpha
#     lam = 1 / alpha - 1
#     w = alpha / (math.exp(1) * (1 - alpha))
#     log_r = math.log(1.0 / (1.0 + w))
#
#     keep_idx = tn.arange(k * size, device=device) if size is not None else tn.arange(k, device=device)
#     rv = zeros(k * size, device=device) if size is not None else zeros(k * size, device=device)
#     sz = len(keep_idx)
#
#     while sz > 0:
#         log_u = tn.log(tn.rand(sz, generator=tng, dtype=rv.dtype, device=device))
#
#         z = zeros(sz, device=device)
#         r_cond = log_u <= log_r
#         nr_cond = ~r_cond
#
#         if tn.any(r_cond):
#             z[r_cond] += -log_u[r_cond] + log_r
#
#         if tn.any(nr_cond):
#             nr_cnt = tn.count_nonzero(nr_cond)
#             z[nr_cond] += tn.log(tn.rand(nr_cnt, generator=tng, dtype=rv.dtype, device=device)) / lam
#
#         log_h = -z - tn.exp(-z / alpha)
#
#         log_eta = tn.where(z >= 0, -z, math.log(w) + math.log(lam) + lam * z)
#         accept = log_h - log_eta >= tn.log(tn.rand(len(z), generator=tng, dtype=rv.dtype, device=device))
#
#         if tn.any(accept):
#             rv[keep_idx[accept]] += -z[accept] / alpha
#             keep_idx = keep_idx[~accept]
#             sz = len(keep_idx)
#
#     rv = tn.reshape(rv, (size, k)) if size is not None else tn.reshape(tn.log(rv), (1, k))
#     rv -= tn.logsumexp(rv, dim=1, keepdim=True)
#     tn.exp(rv, out=rv)
#
#     return rv