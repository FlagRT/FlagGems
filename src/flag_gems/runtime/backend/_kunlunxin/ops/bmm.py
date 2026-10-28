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


def heur_group_m(args):
    if args["TILE_M"] > args["TILE_N"]:
        return 1
    else:
        return (args["M"] + args["TILE_M"] - 1) // args["TILE_M"]


def heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0


def heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0


def heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0


# The XPU3 autotuner reports a zero-duration sample for block-pointer BMM on
# some large shapes. Keep the normal generated search, but provide a stable
# single configuration for the reproducible compatibility path.
_bmm_autotune = triton.autotune(
    configs=[],
    generate_configs="bmm",
    key=["M", "N", "K"],
)
if os.environ.get("KLX_USE_AUTOTUNE", "1") != "1":
    _bmm_autotune = triton.autotune(
        configs=[triton.Config({"TILE_M": 32, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 1}, num_warps=4, num_stages=2)],
        key=["M", "N", "K"],
    )

@libentry()
@_bmm_autotune
@triton.heuristics(
    {
        "GROUP_M": heur_group_m,
        "DIVISIBLE_M": heur_divisible_m,
        "DIVISIBLE_N": heur_divisible_n,
        "DIVISIBLE_K": heur_divisible_k,
    }
)
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_b = tle.program_id(2)
    A = A + pid_b * M * K
    B = B + pid_b * K * N
    O = O + pid_b * M * N

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tle.num_programs(0)
        gridy = tle.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    m_offset = (pid_m * TILE_M).to(tl.int32)
    n_offset = (pid_n * TILE_N).to(tl.int32)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_offset = (k * TILE_K).to(tl.int32)
        a_block = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(K, 1),
            offsets=(m_offset, k_offset),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        b_block = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(N, 1),
            offsets=(k_offset, n_offset),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, allow_tf32=False)

    o_block = tl.make_block_ptr(
        base=O,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m_offset, n_offset),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    out = acc.to(O.dtype.element_ty)
    tl.store(o_block, out, boundary_check=(0, 1))


def bmm(A, B):
    logger.debug("GEMS BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K)
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K)
    return out
