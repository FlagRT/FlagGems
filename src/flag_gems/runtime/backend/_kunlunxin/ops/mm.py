import logging
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def heur_split_k(args):
    return 1


def heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def heur_group_m(args):
    if args["BLOCK_M"] > args["BLOCK_N"]:
        return 1
    else:
        return (args["M"] + args["BLOCK_M"] - 1) // args["BLOCK_M"]


autotune_decorator = triton.autotune(
    configs=[],
    generate_configs="mm",
    key=["M", "N", "K"],
)


KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "1") == "1"

if not KLX_USE_AUTOTUNE:
    autotune_decorator = triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256}),
        ],
        key=["M", "N", "K"],
    )


@libentry()
@autotune_decorator
@triton.heuristics(
    {
        "SPLIT_K": heur_split_k,
        "EVEN_K": heur_even_k,
        "GROUP_M": heur_group_m,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        m_offset = (pid_m * BLOCK_M).to(tl.int32)
        n_offset = (pid_n * BLOCK_N).to(tl.int32)
        k_offset = (k * BLOCK_K * SPLIT_K + pid_z * BLOCK_K).to(tl.int32)
        a_block = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(m_offset, k_offset),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, n_offset),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
    acc = acc.to(C.dtype.element_ty)
    c_block = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=((pid_m * BLOCK_M).to(tl.int32), (pid_n * BLOCK_N).to(tl.int32)),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    if SPLIT_K == 1:
        tl.store(c_block, acc, boundary_check=(0, 1))
    else:
        tl.atomic_add(c_block, acc, boundary_check=(0, 1))


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    logger.debug("GEMS MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    dot_out_dtype = tl.float32
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype,
        )
    return c
