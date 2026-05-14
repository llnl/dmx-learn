"""Vector utilities backed by PyTorch tensors."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tn

DINT = tn.int32
_DEFAULT_FLOAT_DTYPE_STATE = {"value": tn.float64}


def _resolve_float_dtype(
    device: Optional[tn.device], dtype: Optional[tn.dtype]
) -> tn.dtype:
    if dtype is not None:
        return dtype
    if device is not None and device.type == "mps":
        return tn.float32
    return _DEFAULT_FLOAT_DTYPE_STATE["value"]


def resolve_device(device=None) -> tn.device:
    """Convert string or None to torch.device. Auto-detects CUDA > MPS > CPU if None."""
    if device is None:
        if tn.cuda.is_available():
            return tn.device("cuda:0")
        if tn.backends.mps.is_available():
            return tn.device("mps")

        return tn.device("cpu")

    return tn.device(device) if isinstance(device, str) else device


def float_dtype_for_device(device: tn.device) -> tn.dtype:
    """Return float32 for MPS (no float64 support), float64 for all others."""
    return tn.float32 if device.type == "mps" else tn.float64


def set_default_float_dtype(dtype: tn.dtype) -> None:
    _DEFAULT_FLOAT_DTYPE_STATE["value"] = dtype


def _sample_dirichlet_with_generator(
    alpha: tn.Tensor, generator: tn.Generator
) -> tn.Tensor:
    """Sample a Dirichlet tensor while honoring the caller's generator.

    PyTorch's public Dirichlet distribution API does not accept a
    `torch.Generator`. This internal helper keeps sampling reproducible for code
    that passes an explicit generator.
    """
    return tn._sample_dirichlet(  # pylint: disable=protected-access
        alpha, generator=generator
    )


def seed_tng(seed: int, device: Optional[tn.device] = None):
    return tn.Generator(device=device).manual_seed(int(seed))


def seed_sample(n: int, tng: tn.Generator):
    if n == 1:
        return tn.randint(0, 2**31, size=(n,), generator=tng, dtype=DINT)[0]

    return tn.randint(0, 2**31, size=(n,), generator=tng, dtype=DINT)


def zeros(
    size: Union[int, Tuple[int, ...]],
    device: Optional[tn.device] = None,
    dtype: Optional[tn.dtype] = None,
):
    return tn.zeros(size, dtype=_resolve_float_dtype(device, dtype), device=device)


def ones(
    size: Union[int, Tuple[int, ...]],
    device: Optional[tn.device] = None,
    dtype: Optional[tn.dtype] = None,
):
    return tn.ones(size, dtype=_resolve_float_dtype(device, dtype), device=device)


def int_vec(size: Union[int, Tuple[int, ...]], device: Optional[tn.device] = None):
    return tn.zeros(size, dtype=DINT, device=device)


def zeros_like(x: tn.Tensor):
    return tn.zeros_like(x, dtype=_resolve_float_dtype(x.device, None))


def tensor(
    x: Union[
        List[int],
        List[float],
        Sequence[int],
        Sequence[float],
        tn.Tensor,
        List[List[int]],
        List[List[float]],
        np.ndarray,
    ],
    device: Optional[tn.device] = None,
    dtype: Optional[tn.dtype] = None,
):
    target_device = (
        device
        if device is not None
        else (x.device if isinstance(x, tn.Tensor) else None)
    )
    dtype = _resolve_float_dtype(target_device, dtype)
    if isinstance(x, tn.Tensor):
        y = x.clone().detach().to(dtype)
        if device is not None:
            y = y.to(device)

        return y

    return tn.tensor(x, device=device, dtype=dtype)


def int_tensor(
    x: Union[List[int], List[List[int]], np.ndarray, tn.Tensor],
    device: Optional[tn.device] = None,
    dtype: Optional[tn.dtype] = DINT,
):
    return tensor(x, device=device, dtype=dtype)


def choice(
    size: int, states: int, tng: tn.Generator, p: Optional[tn.Tensor] = None
) -> tn.Tensor:

    device = tng.device
    if p is not None:
        alpha = p if p.device == device else p.to(device)
    else:
        alpha = ones(states, device=device)

    return tn.multinomial(alpha, num_samples=size, replacement=True, generator=tng)


def randint(
    size: int, low: int, high: int, tng: tn.Generator, dtype: Optional[tn.dtype] = DINT
):
    return tn.randint(
        size=(size,), low=low, high=high, generator=tng, dtype=dtype, device=tng.device
    )


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
    target_device = tng.device if tng.device is not None else alpha.device
    target_dtype = alpha.dtype

    if target_device.type == "mps" or alpha.device.type == "mps":
        alpha_cpu = alpha.detach().cpu().expand((size, k))
        rv = _sample_dirichlet_with_generator(
            alpha_cpu, generator=tn.Generator().manual_seed(tng.initial_seed())
        )
        rv = tensor(rv, device=target_device, dtype=target_dtype)
    else:
        alpha_dev = alpha.to(device=target_device)
        rv = _sample_dirichlet_with_generator(
            alpha_dev.expand((size, k)), generator=tng
        )
        rv = tensor(rv, device=target_device, dtype=target_dtype)

    return rv
