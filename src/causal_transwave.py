# =============================================================================
# CAUSAL TRANSWAVE — COMPLETE REPLACEMENT CELL
#
# Fresh experiment. No offline tokenization, BPE, SentencePiece, MLM or masking.
# Exact autoregressive byte likelihood:
#       NLL = -mean log p(x_t | x_<t)
#       PPL = exp(NLL)
#       BPB = NLL / ln(2)
#
# Causality:
#   * lifting is performed independently inside each completed 8-byte block;
#   * global Meyer mixing is one-sided causal linear convolution;
#   * local convolution is left-causal;
#   * the global block representation is shifted by one block;
#   * the 8 bytes of the current block are decoded autoregressively.
#
# G4:
#   * BF16 network, FP32/complex64 FFT path;
#   * activation checkpointing ON;
#   * batch probe includes forward + backward + grad clip + actual AdamW.step();
#   * choose the largest safe batch with 3 GiB reserved-memory headroom.
#
# Data:
#   existing /content/drive/MyDrive/spliNet/raw_text/<dataset>/*.txt only.
#   No download. No token shards. No re-tokenization.
#
# Runs independent models on:
#   arXiv, FineWeb-Edu, PG19, WikiText103
#
# Maximum exposure: 3 passes.
# Stop only if genuine validation PPL <= 3 (NLL <= ln 3).
# =============================================================================

from __future__ import annotations

import os
import gc
import json
import math
import time
import random
import shutil
import hashlib
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as torch_checkpoint


# =============================================================================
# HARDWARE
# =============================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU required."
    )


DEVICE = torch.device(
    "cuda"
)


AMP_DTYPE = torch.bfloat16


torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True


try:

    torch.set_float32_matmul_precision(
        "high"
    )

except Exception:

    pass


GPU_NAME = torch.cuda.get_device_name(
    0
)


GPU_TOTAL_GB = (
    torch.cuda
    .get_device_properties(
        0
    )
    .total_memory
    /
    1024**3
)


# =============================================================================
# FIXED ARCHITECTURE
# =============================================================================

ARCH = (
    "causal_transwave_blocklifting_meyer_byte_lm_g4_v2"
)


BYTE_VOCAB = 256

BYTE_DIM = 96

BYTES_PER_BLOCK = 8

N_SCALES = 8

MODEL_DIM = 768

MODEL_CONTEXT = 4096


RAW_BYTES_PER_SEQUENCE = (
    MODEL_CONTEXT
    *
    BYTES_PER_BLOCK
)


N_LAYERS = 12

N_GROUPS = 12


CHANNELS_PER_GROUP = (
    BYTE_DIM
    //
    N_GROUPS
)


N_MEYER_BANDS = 12

N_BAND_CONTROLS = 8


# Causal kernel in eight-byte block positions.
MEYER_KERNEL = 512


LOCAL_KERNEL = 7


# Keep V4 capacity initially.
FFN_DIM = 2816


BYTE_DECODER_DIM = 256


HEAD_CHUNK = 32768


assert (
    BYTE_DIM
    *
    BYTES_PER_BLOCK
    ==
    MODEL_DIM
)


assert (
    RAW_BYTES_PER_SEQUENCE
    ==
    32768
)


assert (
    BYTE_DIM
    %
    N_GROUPS
    ==
    0
)


# =============================================================================
# TRAINING
# =============================================================================

MAX_PASSES = 3


BASE_LR = 1.0e-4

MIN_LR = 1.0e-5


WEIGHT_DECAY = 0.0


ADAM_BETAS = (
    0.9,
    0.999,
)


ADAM_EPS = 1.0e-8


GRAD_CLIP = 1.0


TARGET_PPL = 3.0


TARGET_NLL = math.log(
    TARGET_PPL
)


VALIDATE_EVERY_STEPS = 50

SAVE_EVERY_STEPS = 25


# Probe includes actual AdamW state.
VRAM_HEADROOM_GB = 3.0


BATCH_CEILING = 512


MODEL_SEED = 20261010


ORDER_SEEDS = {

    "arxiv":
        8101,

    "fineweb_edu":
        8201,

    "pg19":
        8301,

    "wikitext103":
        8401,
}


DATASETS = (
    "arxiv",
    "fineweb_edu",
    "pg19",
    "wikitext103",
)


DISPLAY_NAMES = {

    "arxiv":
        "arXiv",

    "fineweb_edu":
        "FineWeb-Edu",

    "pg19":
        "PG19",

    "wikitext103":
        "WikiText103",
}


# =============================================================================
# PATHS
# =============================================================================

DRIVE_ROOT = Path(
    "/content/drive/MyDrive"
)


RAW_ROOT = (
    DRIVE_ROOT
    /
    "spliNet"
    /
    "raw_text"
)


PROJECT_ROOT = (
    DRIVE_ROOT
    /
    "TransWave_Meyer_Context"
)


CHECKPOINT_ROOT = (
    PROJECT_ROOT
    /
    "checkpoints"
    /
    ARCH
)


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    ARCH
)


LOCAL_RAW_ROOT = Path(
    "/content/causal_transwave_raw"
)


for p in (

    CHECKPOINT_ROOT,
    RESULT_ROOT,
    LOCAL_RAW_ROOT,

):

    p.mkdir(
        parents=True,
        exist_ok=True,
    )


CONFIG = {

    "architecture":
        ARCH,

    "objective":
        "exact_causal_next_byte_nll",

    "byte_vocab":
        BYTE_VOCAB,

    "byte_dim":
        BYTE_DIM,

    "bytes_per_block":
        BYTES_PER_BLOCK,

    "model_dim":
        MODEL_DIM,

    "model_context":
        MODEL_CONTEXT,

    "raw_bytes_per_sequence":
        RAW_BYTES_PER_SEQUENCE,

    "layers":
        N_LAYERS,

    "groups":
        N_GROUPS,

    "meyer_bands":
        N_MEYER_BANDS,

    "band_controls":
        N_BAND_CONTROLS,

    "meyer_kernel":
        MEYER_KERNEL,

    "local_kernel":
        LOCAL_KERNEL,

    "ffn_dim":
        FFN_DIM,

    "byte_decoder_dim":
        BYTE_DECODER_DIM,

    "maximum_passes":
        MAX_PASSES,
}


CONFIG_FINGERPRINT = hashlib.sha256(

    json.dumps(
        CONFIG,
        sort_keys=True,
    ).encode()

).hexdigest()


# =============================================================================
# UTILITIES
# =============================================================================

def cleanup_cuda():

    gc.collect()


    try:

        torch.cuda.synchronize()

    except Exception:

        pass


    torch.cuda.empty_cache()


    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


def seed_all(
    seed,
):

    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    torch.cuda.manual_seed_all(
        seed
    )


def atomic_torch_save(
    obj,
    path,
):

    path = Path(
        path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )


    tmp.unlink(
        missing_ok=True
    )


    torch.save(
        obj,
        tmp,
    )


    os.replace(
        tmp,
        path,
    )


def atomic_json(
    obj,
    path,
):

    path = Path(
        path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )


    tmp.unlink(
        missing_ok=True
    )


    tmp.write_text(

        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ),

        encoding="utf-8",
    )


    os.replace(
        tmp,
        path,
    )


# =============================================================================
# RAW BYTE DATA
# =============================================================================

def stage_raw_file(
    dataset,
    split,
):

    src = (
        RAW_ROOT
        /
        dataset
        /
        f"{split}.txt"
    )


    if not src.is_file():

        raise FileNotFoundError(

            "Existing raw file missing; "
            "this cell will not download/tokenize:\n"
            f"{src}"
        )


    dst_dir = (
        LOCAL_RAW_ROOT
        /
        dataset
    )


    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    dst = (
        dst_dir
        /
        f"{split}.txt"
    )


    if (
        dst.is_file()
        and
        dst.stat().st_size
        ==
        src.stat().st_size
    ):

        print(

            f"  {dataset}/{split}: "
            f"local raw copy present "
            f"({dst.stat().st_size / 1024**2:.1f} MiB)"
        )


        return dst


    print(

        f"  {dataset}/{split}: "
        f"copying raw bytes to local SSD "
        f"({src.stat().st_size / 1024**2:.1f} MiB)"
    )


    tmp = dst.with_suffix(
        ".txt.tmp"
    )


    tmp.unlink(
        missing_ok=True
    )


    shutil.copyfile(
        src,
        tmp,
    )


    if (
        tmp.stat().st_size
        !=
        src.stat().st_size
    ):

        tmp.unlink(
            missing_ok=True
        )


        raise RuntimeError(
            f"Incomplete raw copy for "
            f"{dataset}/{split}"
        )


    os.replace(
        tmp,
        dst,
    )


    return dst


def open_blocks(
    path,
):

    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
    )


    usable = (
        len(
            raw
        )
        //
        RAW_BYTES_PER_SEQUENCE
        *
        RAW_BYTES_PER_SEQUENCE
    )


    blocks = (
        raw[
            :usable
        ]
        .reshape(
            -1,
            RAW_BYTES_PER_SEQUENCE,
        )
    )


    return (
        raw,
        blocks,
    )


def make_cpu_batch(
    blocks,
    indices,
):

    arr = np.asarray(
        blocks[
            indices
        ],
        dtype=np.uint8,
    )


    arr = np.ascontiguousarray(
        arr
    )


    return (
        torch
        .from_numpy(
            arr
        )
        .pin_memory()
    )


# =============================================================================
# CAUSAL BLOCK-LOCAL LEARNED LIFTING
#
# Same Cell-3 LocalBlockDelta architecture, but applied independently to each
# completed eight-byte block. Therefore no convolution can cross from block b
# into block b+1.
# =============================================================================

class CTWLocalBlockDelta(
    nn.Module
):

    def __init__(
        self,
        dim=BYTE_DIM,
        groups=12,
        expansion=2,
    ):

        super().__init__()


        if (
            dim
            %
            groups
            !=
            0
        ):

            raise ValueError(
                "dim must be divisible by groups"
            )


        hidden = (
            dim
            *
            expansion
        )


        if (
            hidden
            %
            groups
            !=
            0
        ):

            raise ValueError(
                "expanded dimension must be divisible by groups"
            )


        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )


        self.mix_in = nn.Conv1d(
            dim,
            hidden,
            kernel_size=1,
            groups=groups,
            bias=True,
        )


        self.mix_out = nn.Conv1d(
            hidden,
            dim,
            kernel_size=1,
            groups=groups,
            bias=True,
        )


        # Haar-like initialization:
        # residual branch starts exactly at zero.

        nn.init.zeros_(
            self.mix_out.weight
        )


        nn.init.zeros_(
            self.mix_out.bias
        )


    def forward(
        self,
        x,
    ):

        y = x.transpose(
            1,
            2,
        )


        y = self.depthwise(
            y
        )


        y = F.silu(
            y
        )


        y = self.mix_in(
            y
        )


        y = F.silu(
            y
        )


        y = self.mix_out(
            y
        )


        return y.transpose(
            1,
            2,
        )


class CTWLiftingStage(
    nn.Module
):

    def __init__(
        self,
        dim=BYTE_DIM,
        groups=12,
        expansion=2,
    ):

        super().__init__()


        self.predict_delta = (
            CTWLocalBlockDelta(
                dim=dim,
                groups=groups,
                expansion=expansion,
            )
        )


        self.update_delta = (
            CTWLocalBlockDelta(
                dim=dim,
                groups=groups,
                expansion=expansion,
            )
        )


    def predict(
        self,
        even,
    ):

        return (
            even
            +
            self.predict_delta(
                even
            )
        )


    def update(
        self,
        detail,
    ):

        return (
            0.5
            *
            detail
            +
            self.update_delta(
                detail
            )
        )


    def forward(
        self,
        x,
    ):

        if (
            x.ndim != 3
            or
            x.shape[1] % 2 != 0
        ):

            raise ValueError(
                "Expected [N,T,C] with even T."
            )


        even = x[
            :,
            0::2,
            :
        ]


        odd = x[
            :,
            1::2,
            :
        ]


        detail = (
            odd
            -
            self.predict(
                even
            )
        )


        coarse = (
            even
            +
            self.update(
                detail
            )
        )


        return (
            coarse,
            detail,
        )


class CTWCrossScaleMixer(
    nn.Module
):

    def __init__(
        self,
        scales=N_SCALES,
        dim=BYTE_DIM,
        groups=N_GROUPS,
    ):

        super().__init__()


        if (
            dim
            %
            groups
            !=
            0
        ):

            raise ValueError(
                "dim must be divisible by groups"
            )


        self.scales = int(
            scales
        )


        self.dim = int(
            dim
        )


        self.groups = int(
            groups
        )


        self.channels_per_group = (
            self.dim
            //
            self.groups
        )


        # M_g = I + Delta_g

        self.scale_delta = nn.Parameter(

            torch.zeros(
                self.groups,
                self.scales,
                self.scales,
            )
        )


        # Content-dependent gain.

        self.gate_weight = nn.Parameter(

            torch.zeros(
                self.groups,
                self.scales,
                self.channels_per_group,
            )
        )


        self.gate_bias = nn.Parameter(

            torch.zeros(
                self.groups,
                self.scales,
            )
        )


    def forward(
        self,
        z,
    ):

        if (
            z.ndim != 4
            or
            z.shape[2]
            !=
            self.scales
            or
            z.shape[3]
            !=
            self.dim
        ):

            raise ValueError(

                f"Expected "
                f"[B,L,{self.scales},{self.dim}]"
            )


        b, l, s, _ = z.shape


        zg = z.reshape(
            b,
            l,
            s,
            self.groups,
            self.channels_per_group,
        )


        gate_logits = torch.einsum(
            "blsgc,gsc->blsg",
            zg,
            self.gate_weight,
        )


        gate_logits = (
            gate_logits
            /
            math.sqrt(
                self.channels_per_group
            )
            +
            self.gate_bias.transpose(
                0,
                1,
            )
        )


        # Zero logits -> gain exactly 1.

        gain = (
            2.0
            *
            torch.sigmoid(
                gate_logits
            )
        )


        zg = (
            zg
            *
            gain
            .to(
                zg.dtype
            )
            .unsqueeze(
                -1
            )
        )


        eye = torch.eye(
            self.scales,
            device=z.device,
            dtype=z.dtype,
        ).unsqueeze(
            0
        )


        mix = (
            eye
            +
            self.scale_delta.to(
                z.dtype
            )
        )


        mixed = torch.einsum(
            "blsgc,gts->bltgc",
            zg,
            mix,
        )


        return mixed.reshape(
            b,
            l,
            s,
            self.dim,
        )


class CTWBlockLiftingTokenizer(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.byte_embedding = nn.Embedding(
            BYTE_VOCAB,
            BYTE_DIM,
        )


        nn.init.normal_(

            self.byte_embedding.weight,

            mean=0.0,

            std=(
                BYTE_DIM
                **
                -0.5
            ),
        )


        self.stages = nn.ModuleList(

            [

                CTWLiftingStage(
                    dim=BYTE_DIM,
                    groups=N_GROUPS,
                    expansion=2,
                )

                for _ in range(
                    3
                )
            ]
        )


        self.cross_scale = (
            CTWCrossScaleMixer(
                scales=N_SCALES,
                dim=BYTE_DIM,
                groups=N_GROUPS,
            )
        )


    def forward(
        self,
        byte_ids,
    ):

        if (
            byte_ids.ndim
            !=
            2
        ):

            raise ValueError(
                "Expected byte IDs [B,8L]."
            )


        b, t = byte_ids.shape


        if (
            t
            %
            BYTES_PER_BLOCK
            !=
            0
        ):

            raise ValueError(
                "Byte length must be divisible by 8."
            )


        l = (
            t
            //
            BYTES_PER_BLOCK
        )


        # -------------------------------------------------------------
        # CRITICAL CAUSAL CHANGE:
        #
        # each completed eight-byte block becomes an independent item
        # before ANY lifting convolution is applied.
        # -------------------------------------------------------------

        blocks = (
            byte_ids
            .long()
            .reshape(
                b,
                l,
                BYTES_PER_BLOCK,
            )
        )


        x = self.byte_embedding(

            blocks.reshape(
                b
                *
                l,
                BYTES_PER_BLOCK,
            )
        )


        details = []


        coarse = x


        for stage in self.stages:

            (
                coarse,
                detail,
            ) = stage(
                coarse
            )


            details.append(
                detail
            )


        # D0=4, D1=2, D2=1, S3=1.

        packed = torch.cat(

            [
                details[0],
                details[1],
                details[2],
                coarse,
            ],

            dim=1,
        )


        if (
            packed.shape[1]
            !=
            N_SCALES
        ):

            raise RuntimeError(
                "Lifting scale packing mismatch."
            )


        z = packed.reshape(
            b,
            l,
            N_SCALES,
            BYTE_DIM,
        )


        return self.cross_scale(
            z
        )


# =============================================================================
# RMS NORM / SWIGLU
# =============================================================================

class CTWRMSNorm(
    nn.Module
):

    def __init__(
        self,
        dim,
        eps=1e-6,
    ):

        super().__init__()


        self.weight = nn.Parameter(
            torch.ones(
                dim
            )
        )


        self.eps = eps


    def forward(
        self,
        x,
    ):

        xf = x.float()


        y = (
            xf
            *
            torch.rsqrt(

                xf.pow(2).mean(
                    dim=-1,
                    keepdim=True,
                )
                +
                self.eps
            )
        )


        return (
            y.to(
                x.dtype
            )
            *
            self.weight.to(
                x.dtype
            )
        )


class CTWSwiGLU(
    nn.Module
):

    def __init__(
        self,
        dim,
        hidden,
    ):

        super().__init__()


        self.gate = nn.Linear(
            dim,
            hidden,
        )


        self.value = nn.Linear(
            dim,
            hidden,
        )


        self.out = nn.Linear(
            hidden,
            dim,
        )


        for m in (
            self.gate,
            self.value,
            self.out,
        ):

            nn.init.xavier_uniform_(
                m.weight
            )


            nn.init.zeros_(
                m.bias
            )


    def forward(
        self,
        x,
    ):

        return self.out(

            F.silu(
                self.gate(
                    x
                )
            )

            *

            self.value(
                x
            )
        )


# =============================================================================
# MEYER BASIS
# =============================================================================

def ctw_meyer_nu(
    x,
):

    x = x.clamp(
        0.0,
        1.0,
    )


    return (
        x**4
        *
        (
            35.0
            -
            84.0 * x
            +
            70.0 * x**2
            -
            20.0 * x**3
        )
    )


def ctw_build_meyer_basis(
    kernel_size,
    bands,
    controls,
):

    nf = (
        kernel_size
        //
        2
        +
        1
    )


    f = torch.linspace(
        0.0,
        1.0,
        nf,
        dtype=torch.float32,
    )


    edges = torch.linspace(
        0.0,
        1.0,
        bands + 1,
        dtype=torch.float32,
    )


    windows = []


    for j in range(
        bands
    ):

        left = float(
            edges[j]
        )


        right = float(
            edges[j + 1]
        )


        delta = (
            0.35
            *
            (
                right
                -
                left
            )
        )


        if j == 0:

            left_gate = torch.ones_like(
                f
            )

        else:

            a = (
                left
                -
                delta
            )


            b = (
                left
                +
                delta
            )


            u = (
                f
                -
                a
            ) / max(
                b - a,
                1e-8,
            )


            left_gate = torch.sin(

                0.5
                *
                math.pi
                *
                ctw_meyer_nu(
                    u
                )
            )


            left_gate = torch.where(
                f <= a,
                torch.zeros_like(
                    f
                ),
                left_gate,
            )


            left_gate = torch.where(
                f >= b,
                torch.ones_like(
                    f
                ),
                left_gate,
            )


        if j == bands - 1:

            right_gate = torch.ones_like(
                f
            )

        else:

            a = (
                right
                -
                delta
            )


            b = (
                right
                +
                delta
            )


            u = (
                f
                -
                a
            ) / max(
                b - a,
                1e-8,
            )


            right_gate = torch.cos(

                0.5
                *
                math.pi
                *
                ctw_meyer_nu(
                    u
                )
            )


            right_gate = torch.where(
                f <= a,
                torch.ones_like(
                    f
                ),
                right_gate,
            )


            right_gate = torch.where(
                f >= b,
                torch.zeros_like(
                    f
                ),
                right_gate,
            )


        windows.append(
            left_gate
            *
            right_gate
        )


    windows = torch.stack(
        windows,
        dim=0,
    )


    windows = (
        windows
        /
        torch.sqrt(

            windows.pow(2).sum(
                dim=0,
                keepdim=True,
            )
            +
            1e-12
        )
    )


    degree = (
        controls
        -
        1
    )


    basis = []


    for j in range(
        bands
    ):

        left = edges[j]


        right = edges[j + 1]


        u = (
            (
                f
                -
                left
            )
            /
            (
                right
                -
                left
                +
                1e-12
            )
        ).clamp(
            0.0,
            1.0,
        )


        row = []


        for m in range(
            controls
        ):

            bern = (

                math.comb(
                    degree,
                    m,
                )

                *

                u.pow(
                    m
                )

                *

                (
                    1.0
                    -
                    u
                ).pow(
                    degree
                    -
                    m
                )
            )


            row.append(
                windows[j]
                *
                bern
            )


        basis.append(

            torch.stack(
                row,
                dim=0,
            )
        )


    return torch.stack(
        basis,
        dim=0,
    )


# =============================================================================
# SAME-TIME FULL 8x8 SCALE MIXING
# =============================================================================

class CTWScaleMixer(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        eye = torch.eye(
            N_SCALES
        )


        self.weight = nn.Parameter(

            eye
            .unsqueeze(
                0
            )
            .repeat(
                N_GROUPS,
                1,
                1,
            )
        )


    def forward(
        self,
        x,
    ):

        b, l, s, c = x.shape


        xg = x.reshape(
            b,
            l,
            s,
            N_GROUPS,
            CHANNELS_PER_GROUP,
        )


        y = torch.einsum(
            "blsgc,gts->bltgc",
            xg,
            self.weight,
        )


        return y.reshape(
            b,
            l,
            s,
            c,
        )


# =============================================================================
# STRICTLY CAUSAL MEYER FIR
#
# h[k] is a nonnegative lag:
#
# y[t] = SUM_{k=0}^{K-1} h[k] x[t-k]
#
# FFT only implements this zero-padded LINEAR convolution efficiently.
# =============================================================================

class CTWCausalMeyerFIR(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        basis = ctw_build_meyer_basis(
            MEYER_KERNEL,
            N_MEYER_BANDS,
            N_BAND_CONTROLS,
        )


        self.register_buffer(
            "basis",
            basis,
            persistent=False,
        )


        imag_mask = torch.ones(
            MEYER_KERNEL
            //
            2
            +
            1,
            dtype=torch.float32,
        )


        imag_mask[0] = 0.0


        if (
            MEYER_KERNEL
            %
            2
            ==
            0
        ):

            imag_mask[-1] = 0.0


        self.register_buffer(
            "imag_mask",
            imag_mask,
            persistent=False,
        )


        shape = (
            N_GROUPS,
            N_SCALES,
            N_MEYER_BANDS,
            N_BAND_CONTROLS,
        )


        self.coeff_re = nn.Parameter(
            torch.empty(
                shape
            )
        )


        self.coeff_im = nn.Parameter(
            torch.empty(
                shape
            )
        )


        nn.init.normal_(
            self.coeff_re,
            mean=0.0,
            std=1e-3,
        )


        nn.init.normal_(
            self.coeff_im,
            mean=0.0,
            std=1e-3,
        )


    def causal_kernel(
        self,
    ):

        with torch.autocast(
            "cuda",
            enabled=False,
        ):

            basis = self.basis.float()


            re = torch.einsum(
                "gsbm,bmf->gsf",
                self.coeff_re.float(),
                basis,
            )


            im = torch.einsum(
                "gsbm,bmf->gsf",
                self.coeff_im.float(),
                basis,
            )


            im = (
                im
                *
                self.imag_mask[
                    None,
                    None,
                    :
                ]
            )


            spectrum = torch.complex(
                re,
                im,
            )


            h = torch.fft.irfft(
                spectrum,
                n=MEYER_KERNEL,
                dim=-1,
            )


        return h


    def forward(
        self,
        x,
    ):

        # [B,L,S,96]

        b, l, s, c = x.shape


        # [B,G,S,Cg,L]

        xg = (
            x
            .reshape(
                b,
                l,
                s,
                N_GROUPS,
                CHANNELS_PER_GROUP,
            )
            .permute(
                0,
                3,
                2,
                4,
                1,
            )
            .contiguous()
        )


        linear_length = (
            l
            +
            MEYER_KERNEL
            -
            1
        )


        nfft = (
            1
            <<
            (
                linear_length
                -
                1
            ).bit_length()
        )


        with torch.autocast(
            "cuda",
            enabled=False,
        ):

            xf = xg.float()


            h = self.causal_kernel()


            X = torch.fft.rfft(
                xf,
                n=nfft,
                dim=-1,
            )


            H = torch.fft.rfft(
                h,
                n=nfft,
                dim=-1,
            )


            Y = torch.fft.irfft(

                X
                *
                H[
                    None,
                    :,
                    :,
                    None,
                    :
                ],

                n=nfft,
                dim=-1,
            )


            # First L outputs of LINEAR convolution.

            Y = Y[
                ...,
                :l
            ]


        return (

            Y
            .permute(
                0,
                4,
                2,
                1,
                3,
            )
            .contiguous()
            .reshape(
                b,
                l,
                s,
                c,
            )
            .to(
                x.dtype
            )
        )


# =============================================================================
# STRICTLY LEFT-CAUSAL LOCAL CONVOLUTION
# =============================================================================

class CTWCausalLocalConv(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.conv = nn.Conv1d(
            MODEL_DIM,
            MODEL_DIM,
            kernel_size=LOCAL_KERNEL,
            groups=MODEL_DIM,
            bias=True,
        )


        nn.init.normal_(
            self.conv.weight,
            mean=0.0,
            std=0.02,
        )


        nn.init.zeros_(
            self.conv.bias
        )


    def forward(
        self,
        x,
    ):

        xt = x.transpose(
            1,
            2,
        )


        xt = F.pad(
            xt,
            (
                LOCAL_KERNEL
                -
                1,
                0,
            ),
        )


        return (
            self.conv(
                xt
            )
            .transpose(
                1,
                2,
            )
        )


# =============================================================================
# ONE CAUSAL TRANSWAVE LAYER
# =============================================================================

class CTWLayer(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.norm1 = CTWRMSNorm(
            MODEL_DIM
        )


        self.input_gate = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )


        self.output_gate = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )


        nn.init.zeros_(
            self.input_gate.weight
        )


        nn.init.constant_(
            self.input_gate.bias,
            2.0,
        )


        nn.init.zeros_(
            self.output_gate.weight
        )


        nn.init.constant_(
            self.output_gate.bias,
            2.0,
        )


        self.scale_in = (
            CTWScaleMixer()
        )


        self.meyer = (
            CTWCausalMeyerFIR()
        )


        self.scale_out = (
            CTWScaleMixer()
        )


        self.local = (
            CTWCausalLocalConv()
        )


        self.alpha_meyer = nn.Parameter(

            torch.tensor(
                0.10,
                dtype=torch.float32,
            )
        )


        self.alpha_local = nn.Parameter(

            torch.tensor(
                0.10,
                dtype=torch.float32,
            )
        )


        self.norm2 = CTWRMSNorm(
            MODEL_DIM
        )


        self.ffn = CTWSwiGLU(
            MODEL_DIM,
            FFN_DIM,
        )


    def forward(
        self,
        x,
    ):

        b, l, s, c = x.shape


        flat = x.reshape(
            b,
            l,
            MODEL_DIM,
        )


        n = self.norm1(
            flat
        )


        gate_in = torch.sigmoid(

            self.input_gate(
                n
            )

        ).reshape(
            b,
            l,
            s,
            c,
        )


        gate_out = torch.sigmoid(

            self.output_gate(
                n
            )

        ).reshape(
            b,
            l,
            s,
            c,
        )


        spectral = self.meyer(

            self.scale_in(
                x
                *
                gate_in
            )
        )


        spectral = self.scale_out(
            spectral
        )


        local = self.local(
            n
        ).reshape(
            b,
            l,
            s,
            c,
        )


        x = (

            x

            +

            self.alpha_meyer.to(
                x.dtype
            )
            *
            spectral
            *
            gate_out

            +

            self.alpha_local.to(
                x.dtype
            )
            *
            local
        )


        flat = x.reshape(
            b,
            l,
            MODEL_DIM,
        )


        flat = (

            flat

            +

            self.ffn(
                self.norm2(
                    flat
                )
            )
        )


        return flat.reshape(
            b,
            l,
            s,
            c,
        )


# =============================================================================
# EXACT EIGHT-BYTE AUTOREGRESSIVE DECODER
# =============================================================================

class CTWByteDecoder(
    nn.Module
):

    def __init__(
        self,
        shared_embedding,
    ):

        super().__init__()


        self.shared_embedding = (
            shared_embedding
        )


        self.context_proj = nn.Linear(
            MODEL_DIM,
            BYTE_DECODER_DIM,
        )


        self.gru = nn.GRUCell(
            BYTE_DIM,
            BYTE_DECODER_DIM,
        )


        self.norm = CTWRMSNorm(
            BYTE_DECODER_DIM
        )


        self.to_byte_space = nn.Linear(
            BYTE_DECODER_DIM,
            BYTE_DIM,
        )


        self.bos = nn.Parameter(
            torch.zeros(
                BYTE_DIM
            )
        )


        self.byte_position = nn.Parameter(

            torch.zeros(
                BYTES_PER_BLOCK,
                BYTE_DIM,
            )
        )


        nn.init.normal_(
            self.bos,
            mean=0.0,
            std=0.02,
        )


        nn.init.normal_(
            self.byte_position,
            mean=0.0,
            std=0.01,
        )


    def nll(
        self,
        context,
        target_blocks,
    ):

        b, l, _ = context.shape


        n = (
            b
            *
            l
        )


        hidden = torch.tanh(

            self.context_proj(
                context
            )

        ).reshape(
            n,
            BYTE_DECODER_DIM,
        )


        previous = (
            self.bos[
                None,
                :
            ]
            .expand(
                n,
                -1,
            )
        )


        total_loss = None


        for r in range(
            BYTES_PER_BLOCK
        ):

            decoder_input = (

                previous

                +

                self.byte_position[
                    r
                ][
                    None,
                    :
                ]
            )


            hidden = self.gru(
                decoder_input,
                hidden,
            )


            z = self.to_byte_space(
                self.norm(
                    hidden
                )
            )


            target = (
                target_blocks[
                    :,
                    :,
                    r
                ]
                .reshape(
                    -1
                )
                .long()
            )


            position_loss = None


            for start in range(
                0,
                n,
                HEAD_CHUNK,
            ):

                end = min(
                    start
                    +
                    HEAD_CHUNK,
                    n,
                )


                logits = F.linear(
                    z[
                        start:end
                    ],
                    self.shared_embedding.weight,
                )


                piece = F.cross_entropy(
                    logits.float(),
                    target[
                        start:end
                    ],
                    reduction="sum",
                )


                position_loss = (
                    piece
                    if position_loss is None
                    else
                    position_loss
                    +
                    piece
                )


            total_loss = (
                position_loss
                if total_loss is None
                else
                total_loss
                +
                position_loss
            )


            # Exact teacher forcing:
            # byte r+1 may condition on true byte r.

            previous = self.shared_embedding(
                target
            )


        return (
            total_loss
            /
            float(
                n
                *
                BYTES_PER_BLOCK
            )
        )


# =============================================================================
# COMPLETE CAUSAL MODEL
# =============================================================================

class CTWModel(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.tokenizer = (
            CTWBlockLiftingTokenizer()
        )


        self.position = nn.Parameter(

            torch.zeros(
                1,
                MODEL_CONTEXT,
                MODEL_DIM,
            )
        )


        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.01,
        )


        self.layers = nn.ModuleList(

            [

                CTWLayer()

                for _ in range(
                    N_LAYERS
                )
            ]
        )


        self.final_norm = CTWRMSNorm(
            MODEL_DIM
        )


        self.bos_context = nn.Parameter(
            torch.zeros(
                MODEL_DIM
            )
        )


        nn.init.normal_(
            self.bos_context,
            mean=0.0,
            std=0.02,
        )


        self.decoder = CTWByteDecoder(
            self.tokenizer.byte_embedding
        )


    def encode_completed_blocks(
        self,
        raw_bytes,
        checkpoint_layers,
    ):

        if (
            raw_bytes.ndim
            !=
            2
        ):

            raise ValueError(
                "Expected raw byte IDs [B,T]."
            )


        b, t = raw_bytes.shape


        if (
            t
            %
            BYTES_PER_BLOCK
            !=
            0
        ):

            raise ValueError(
                "Raw byte length must be divisible by 8."
            )


        l = (
            t
            //
            BYTES_PER_BLOCK
        )


        if (
            l
            >
            MODEL_CONTEXT
        ):

            raise ValueError(

                f"Context {l} exceeds "
                f"MODEL_CONTEXT={MODEL_CONTEXT}."
            )


        x = self.tokenizer(
            raw_bytes.long()
        )


        pos = (
            self.position[
                :,
                :l,
                :
            ]
            .reshape(
                1,
                l,
                N_SCALES,
                BYTE_DIM,
            )
        )


        x = (
            x
            +
            pos.to(
                x.dtype
            )
        )


        for layer in self.layers:

            if (
                checkpoint_layers
                and
                self.training
            ):

                x = torch_checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

            else:

                x = layer(
                    x
                )


        h = self.final_norm(

            x.reshape(
                b,
                l,
                MODEL_DIM,
            )
        )


        blocks = (
            raw_bytes
            .long()
            .reshape(
                b,
                l,
                BYTES_PER_BLOCK,
            )
        )


        return (
            h,
            blocks,
        )


    def contexts(
        self,
        raw_bytes,
        checkpoint_layers=False,
    ):

        (
            h,
            blocks,
        ) = self.encode_completed_blocks(
            raw_bytes,
            checkpoint_layers,
        )


        b, l, _ = h.shape


        context = torch.empty_like(
            h
        )


        # First block has no preceding block.

        context[
            :,
            0,
            :
        ] = self.bos_context.to(
            h.dtype
        )


        # -------------------------------------------------------------
        # CRITICAL CAUSAL SHIFT
        #
        # prediction for block b receives h[b-1], never h[b].
        # -------------------------------------------------------------

        if l > 1:

            context[
                :,
                1:,
                :
            ] = h[
                :,
                :-1,
                :
            ]


        return (
            context,
            blocks,
        )


    def nll(
        self,
        raw_bytes,
        checkpoint_layers=True,
    ):

        (
            context,
            blocks,
        ) = self.contexts(
            raw_bytes,
            checkpoint_layers,
        )


        return self.decoder.nll(
            context,
            blocks,
        )


# =============================================================================
# PARAMETERS / OPTIMIZER
# =============================================================================

def count_parameters(
    model,
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    return (
        total,
        trainable,
    )


def make_optimizer(
    model,
):

    kwargs = dict(

        lr=BASE_LR,

        betas=ADAM_BETAS,

        eps=ADAM_EPS,

        weight_decay=WEIGHT_DECAY,
    )


    try:

        opt = torch.optim.AdamW(
            model.parameters(),
            fused=True,
            **kwargs,
        )


        return (
            opt,
            True,
        )


    except Exception:

        opt = torch.optim.AdamW(
            model.parameters(),
            **kwargs,
        )


        return (
            opt,
            False,
        )


# =============================================================================
# STRICT CAUSALITY TEST
# =============================================================================

@torch.no_grad()
def causality_test(
    model,
):

    print(
        "\nSTRICT CAUSALITY TEST"
    )


    model.eval()


    n_blocks = 64

    cut = 30


    a = torch.randint(
        0,
        BYTE_VOCAB,
        (
            1,
            n_blocks
            *
            BYTES_PER_BLOCK,
        ),
        device=DEVICE,
        dtype=torch.long,
    )


    b = a.clone()


    # Change block cut+1 and everything after it.

    mutation_start = (
        cut
        +
        1
    ) * BYTES_PER_BLOCK


    b[
        :,
        mutation_start:
    ] = torch.randint(
        0,
        BYTE_VOCAB,
        b[
            :,
            mutation_start:
        ].shape,
        device=DEVICE,
        dtype=torch.long,
    )


    ca, _ = model.contexts(
        a,
        checkpoint_layers=False,
    )


    cb, _ = model.contexts(
        b,
        checkpoint_layers=False,
    )


    # Context for block cut+1 must still be identical.

    diff = float(

        (
            ca[
                :,
                :
                cut
                +
                2,
                :
            ]
            -
            cb[
                :,
                :
                cut
                +
                2,
                :
            ]
        )
        .abs()
        .max()
    )


    print(

        f"max future->past difference : "
        f"{diff:.3e}"
    )


    if (
        diff
        >
        2e-4
    ):

        raise RuntimeError(
            "CAUSALITY TEST FAILED."
        )


    print(
        "causality                    : PASS"
    )


    model.train()


# =============================================================================
# COMPLETE-STEP G4 VRAM PROBE
# =============================================================================

def full_step_probe(
    batch,
):

    cleanup_cuda()


    seed_all(
        MODEL_SEED
    )


    model = (
        CTWModel()
        .to(
            DEVICE
        )
    )


    optimizer, fused = make_optimizer(
        model
    )


    model.train()


    torch.cuda.reset_peak_memory_stats()


    try:

        dummy = torch.randint(
            0,
            BYTE_VOCAB,
            (
                batch,
                RAW_BYTES_PER_SEQUENCE,
            ),
            device=DEVICE,
            dtype=torch.long,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            "cuda",
            dtype=AMP_DTYPE,
        ):

            loss = model.nll(
                dummy,
                checkpoint_layers=True,
            )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )


        # Important:
        # allocate/exercise actual AdamW moments.

        optimizer.step()


        torch.cuda.synchronize()


        peak_alloc = (
            torch.cuda
            .max_memory_allocated()
            /
            1024**3
        )


        peak_reserved = (
            torch.cuda
            .max_memory_reserved()
            /
            1024**3
        )


        result = {

            "fits":
                True,

            "batch":
                batch,

            "peak_alloc":
                peak_alloc,

            "peak_reserved":
                peak_reserved,

            "free":
                GPU_TOTAL_GB
                -
                peak_reserved,

            "nll":
                float(
                    loss.detach()
                ),

            "fused":
                fused,
        }


        del (
            dummy,
            loss,
        )


    except torch.cuda.OutOfMemoryError:

        result = {

            "fits":
                False,

            "batch":
                batch,
        }


    except RuntimeError as exc:

        if (
            "out of memory"
            in
            str(
                exc
            ).lower()
        ):

            result = {

                "fits":
                    False,

                "batch":
                    batch,
            }

        else:

            raise


    finally:

        try:

            optimizer.zero_grad(
                set_to_none=True
            )

        except Exception:

            pass


        del (
            optimizer,
            model,
        )


        cleanup_cuda()


    return result


def choose_batch_size():

    print(
        "=" * 120
    )


    print(
        "G4 COMPLETE-STEP VRAM AUTOTUNER"
    )


    print(
        f"GPU                     : "
        f"{GPU_NAME}"
    )


    print(
        f"VRAM                    : "
        f"{GPU_TOTAL_GB:.2f} GiB"
    )


    print(
        f"required headroom       : "
        f"{VRAM_HEADROOM_GB:.2f} GiB"
    )


    print(
        "probe                   : "
        "forward + backward + AdamW.step()"
    )


    print(
        "activation checkpointing: ON"
    )


    print(
        "=" * 120
    )


    seed_all(
        MODEL_SEED
    )


    test_model = (
        CTWModel()
        .to(
            DEVICE
        )
    )


    (
        total,
        trainable,
    ) = count_parameters(
        test_model
    )


    print(
        f"parameters              : "
        f"{total:,}"
    )


    print(
        f"trainable               : "
        f"{trainable:,}"
    )


    causality_test(
        test_model
    )


    del test_model


    cleanup_cuda()


    coarse = (

        8,
        16,
        24,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
    )


    best_safe = None

    first_unsafe = None


    for batch in coarse:

        if (
            batch
            >
            BATCH_CEILING
        ):

            break


        r = full_step_probe(
            batch
        )


        if not r[
            "fits"
        ]:

            print(
                f"B={batch:<3d} OOM"
            )


            first_unsafe = batch


            break


        print(

            f"B={batch:<3d} FIT | "

            f"alloc="
            f"{r['peak_alloc']:6.2f} GiB | "

            f"reserved="
            f"{r['peak_reserved']:6.2f} GiB | "

            f"free="
            f"{r['free']:6.2f} GiB | "

            f"NLL="
            f"{r['nll']:.4f}"
        )


        if (
            r[
                "free"
            ]
            >=
            VRAM_HEADROOM_GB
        ):

            best_safe = batch


        else:

            first_unsafe = batch


            print(

                f"B={batch}: "
                f"fits but violates "
                f"{VRAM_HEADROOM_GB:.1f} GiB headroom."
            )


            break


    if best_safe is None:

        for batch in (
            6,
            4,
            2,
            1,
        ):

            r = full_step_probe(
                batch
            )


            if not r[
                "fits"
            ]:

                print(
                    f"B={batch:<3d} OOM"
                )


                continue


            print(

                f"B={batch:<3d} FIT | "

                f"alloc="
                f"{r['peak_alloc']:6.2f} GiB | "

                f"reserved="
                f"{r['peak_reserved']:6.2f} GiB | "

                f"free="
                f"{r['free']:6.2f} GiB"
            )


            if (
                r[
                    "free"
                ]
                >=
                VRAM_HEADROOM_GB
            ):

                best_safe = batch


                break


    if best_safe is None:

        raise RuntimeError(
            "No batch fits a complete AdamW step."
        )


    # Refine between safe and unsafe in multiples of 4.

    if (
        first_unsafe is not None
        and
        first_unsafe
        >
        best_safe
        +
        4
    ):

        low = best_safe

        high = first_unsafe


        while (
            high
            -
            low
            >
            4
        ):

            mid = (
                (
                    low
                    +
                    high
                )
                //
                8
                *
                4
            )


            if mid <= low:

                mid = (
                    low
                    +
                    4
                )


            if mid >= high:

                break


            r = full_step_probe(
                mid
            )


            safe = (
                r[
                    "fits"
                ]
                and
                r[
                    "free"
                ]
                >=
                VRAM_HEADROOM_GB
            )


            if safe:

                print(

                    f"B={mid:<3d} SAFE | "

                    f"alloc="
                    f"{r['peak_alloc']:6.2f} GiB | "

                    f"reserved="
                    f"{r['peak_reserved']:6.2f} GiB | "

                    f"free="
                    f"{r['free']:6.2f} GiB"
                )


                low = mid


                best_safe = mid


            else:

                if r[
                    "fits"
                ]:

                    print(

                        f"B={mid:<3d} TOO TIGHT | "

                        f"reserved="
                        f"{r['peak_reserved']:6.2f} GiB | "

                        f"free="
                        f"{r['free']:6.2f} GiB"
                    )

                else:

                    print(
                        f"B={mid:<3d} OOM"
                    )


                high = mid


    print(
        "=" * 120
    )


    print(
        f"SELECTED PHYSICAL BATCH : "
        f"{best_safe}"
    )


    print(
        "selection includes actual "
        "AdamW state/step memory"
    )


    print(
        "=" * 120
    )


    return best_safe


# =============================================================================
# LEARNING RATE / DATA ORDER
# =============================================================================

def learning_rate(
    update,
    total_updates,
):

    warmup = max(
        10,
        int(
            0.01
            *
            total_updates
        ),
    )


    if update < warmup:

        return (
            BASE_LR
            *
            float(
                update
                +
                1
            )
            /
            float(
                warmup
            )
        )


    progress = (
        float(
            update
            -
            warmup
        )
        /
        float(
            max(
                1,
                total_updates
                -
                warmup,
            )
        )
    )


    progress = max(
        0.0,
        min(
            1.0,
            progress,
        ),
    )


    cosine = (
        0.5
        *
        (
            1.0
            +
            math.cos(
                math.pi
                *
                progress
            )
        )
    )


    return (
        MIN_LR
        +
        (
            BASE_LR
            -
            MIN_LR
        )
        *
        cosine
    )


def dataset_indices(
    n_train,
    examples_seen,
    batch_size,
    seed,
):

    cursor = int(
        examples_seen
    )


    remaining = int(
        batch_size
    )


    pieces = []


    while remaining > 0:

        pass_id = (
            cursor
            //
            n_train
        )


        offset = (
            cursor
            %
            n_train
        )


        permutation = (

            np.random
            .default_rng(
                seed
                +
                pass_id
            )
            .permutation(
                n_train
            )
        )


        take = min(
            remaining,
            n_train
            -
            offset,
        )


        pieces.append(

            permutation[
                offset:
                offset
                +
                take
            ]
        )


        cursor += take


        remaining -= take


    return np.concatenate(
        pieces
    )


# =============================================================================
# GENUINE CAUSAL VALIDATION
# =============================================================================

@torch.no_grad()
def validate_model(
    model,
    blocks,
    batch_size,
):

    model.eval()


    total_loss = 0.0

    total_bytes = 0


    for start in range(
        0,
        len(
            blocks
        ),
        batch_size,
    ):

        count = min(
            batch_size,
            len(
                blocks
            )
            -
            start,
        )


        indices = np.arange(
            start,
            start
            +
            count,
            dtype=np.int64,
        )


        cpu = make_cpu_batch(
            blocks,
            indices,
        )


        ids = cpu.to(
            DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )


        with torch.autocast(
            "cuda",
            dtype=AMP_DTYPE,
        ):

            nll = model.nll(
                ids,
                checkpoint_layers=False,
            )


        nbytes = (
            count
            *
            RAW_BYTES_PER_SEQUENCE
        )


        total_loss += (
            float(
                nll.detach()
            )
            *
            nbytes
        )


        total_bytes += nbytes


        del (
            indices,
            cpu,
            ids,
            nll,
        )


    mean_nll = (
        total_loss
        /
        total_bytes
    )


    ppl = math.exp(
        min(
            mean_nll,
            20.0,
        )
    )


    bpb = (
        mean_nll
        /
        math.log(
            2.0
        )
    )


    model.train()


    return (
        mean_nll,
        ppl,
        bpb,
        total_bytes,
    )


# =============================================================================
# TRAIN ONE DATASET
# =============================================================================

def train_dataset(
    dataset,
    default_physical_batch,
):

    name = DISPLAY_NAMES[
        dataset
    ]


    print(
        "\n"
        +
        "#" * 120
    )


    print(
        f"# CAUSAL TRANSWAVE — {name}"
    )


    print(
        "#" * 120
    )


    train_path = stage_raw_file(
        dataset,
        "train",
    )


    valid_path = stage_raw_file(
        dataset,
        "validation",
    )


    (
        train_mem,
        train_blocks,
    ) = open_blocks(
        train_path
    )


    (
        valid_mem,
        valid_blocks,
    ) = open_blocks(
        valid_path
    )


    n_train = int(
        len(
            train_blocks
        )
    )


    n_valid = int(
        len(
            valid_blocks
        )
    )


    if (
        n_train < 1
        or
        n_valid < 1
    ):

        raise RuntimeError(
            f"{dataset}: empty train/validation stream."
        )


    run_root = (
        CHECKPOINT_ROOT
        /
        dataset
    )


    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    latest_path = (
        run_root
        /
        "training_latest.pt"
    )


    best_path = (
        run_root
        /
        "best_model.pt"
    )


    final_path = (
        run_root
        /
        "trained_model_final.pt"
    )


    result_path = (
        RESULT_ROOT
        /
        f"{dataset}.json"
    )


    run_fingerprint = hashlib.sha256(

        (
            CONFIG_FINGERPRINT
            +
            "|"
            +
            dataset
        ).encode()

    ).hexdigest()


    # A completed 3-pass run is already complete,
    # regardless of whether target was reached.

    if result_path.is_file():

        try:

            old = json.loads(

                result_path.read_text(
                    encoding="utf-8"
                )
            )


            if (
                old.get(
                    "fingerprint"
                )
                ==
                run_fingerprint
                and
                old.get(
                    "status"
                )
                ==
                "complete"
            ):

                print(
                    f"{name}: "
                    f"completed result already exists."
                )


                print(

                    f"NLL="
                    f"{old['best_validation_nll']:.6f} | "

                    f"PPL="
                    f"{old['best_validation_ppl']:.6f} | "

                    f"BPB="
                    f"{old['best_validation_bpb']:.6f}"
                )


                del (
                    train_mem,
                    train_blocks,
                    valid_mem,
                    valid_blocks,
                )


                cleanup_cuda()


                return old


        except Exception:

            pass


    cleanup_cuda()


    seed_all(
        MODEL_SEED
    )


    model = (
        CTWModel()
        .to(
            DEVICE
        )
    )


    optimizer, fused_adam = (
        make_optimizer(
            model
        )
    )


    (
        total_params,
        trainable_params,
    ) = count_parameters(
        model
    )


    examples_seen = 0

    update = 0

    best_nll = float(
        "inf"
    )

    history = []

    prior_seconds = 0.0

    stop_reason = None


    physical_batch = int(
        default_physical_batch
    )


    # -------------------------------------------------------------------------
    # RESUME
    # -------------------------------------------------------------------------

    if latest_path.is_file():

        state = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )


        if (
            state.get(
                "fingerprint"
            )
            !=
            run_fingerprint
        ):

            raise RuntimeError(
                f"{dataset}: "
                f"checkpoint fingerprint mismatch."
            )


        model.load_state_dict(
            state[
                "model_state"
            ],
            strict=True,
        )


        optimizer.load_state_dict(
            state[
                "optimizer_state"
            ]
        )


        for item in optimizer.state.values():

            for key, value in list(
                item.items()
            ):

                if torch.is_tensor(
                    value
                ):

                    item[
                        key
                    ] = value.to(
                        DEVICE
                    )


        examples_seen = int(
            state.get(
                "examples_seen",
                0,
            )
        )


        update = int(
            state.get(
                "update",
                0,
            )
        )


        best_nll = float(
            state.get(
                "best_nll",
                float(
                    "inf"
                ),
            )
        )


        history = list(
            state.get(
                "history",
                [],
            )
        )


        prior_seconds = float(
            state.get(
                "training_seconds",
                0.0,
            )
        )


        physical_batch = int(
            state.get(
                "physical_batch",
                physical_batch,
            )
        )


        del state


        print(

            f"RESUME | "

            f"update="
            f"{update:,} | "

            f"examples="
            f"{examples_seen:,} | "

            f"B="
            f"{physical_batch}"
        )


    max_examples = (
        MAX_PASSES
        *
        n_train
    )


    max_updates = max(

        1,

        math.ceil(
            max_examples
            /
            physical_batch
        ),
    )


    print(
        f"train sequences          : "
        f"{n_train:,}"
    )


    print(
        f"validation sequences     : "
        f"{n_valid:,}"
    )


    print(
        f"bytes / sequence         : "
        f"{RAW_BYTES_PER_SEQUENCE:,}"
    )


    print(
        f"physical batch           : "
        f"{physical_batch}"
    )


    print(
        f"parameters               : "
        f"{total_params:,}"
    )


    print(
        f"trainable parameters     : "
        f"{trainable_params:,}"
    )


    print(
        f"fused AdamW              : "
        f"{fused_adam}"
    )


    print(
        f"maximum passes           : "
        f"{MAX_PASSES}"
    )


    print(
        f"maximum updates          : "
        f"{max_updates:,}"
    )


    session_start = (
        time.perf_counter()
    )


    def elapsed():

        return (
            prior_seconds
            +
            time.perf_counter()
            -
            session_start
        )


    def save_latest():

        atomic_torch_save(

            {
                "fingerprint":
                    run_fingerprint,

                "config":
                    CONFIG,

                "dataset":
                    dataset,

                "physical_batch":
                    physical_batch,

                "examples_seen":
                    examples_seen,

                "update":
                    update,

                "best_nll":
                    best_nll,

                "history":
                    history,

                "training_seconds":
                    elapsed(),

                "model_state":
                    model.state_dict(),

                "optimizer_state":
                    optimizer.state_dict(),
            },

            latest_path,
        )


    def save_best(
        val_nll,
        val_ppl,
        val_bpb,
    ):

        atomic_torch_save(

            {
                "fingerprint":
                    run_fingerprint,

                "config":
                    CONFIG,

                "dataset":
                    dataset,

                "physical_batch":
                    physical_batch,

                "examples_seen":
                    examples_seen,

                "update":
                    update,

                "validation_nll":
                    float(
                        val_nll
                    ),

                "validation_ppl":
                    float(
                        val_ppl
                    ),

                "validation_bpb":
                    float(
                        val_bpb
                    ),

                "model_state":
                    model.state_dict(),
            },

            best_path,
        )


    # -------------------------------------------------------------------------
    # IMMEDIATE VALIDATION AFTER RESUME
    # -------------------------------------------------------------------------

    if update > 0:

        print(

            f"\n{name}: "
            f"immediate genuine causal validation"
        )


        (
            val_nll,
            val_ppl,
            val_bpb,
            val_bytes,
        ) = validate_model(
            model,
            valid_blocks,
            physical_batch,
        )


        print(

            f"VALID | "

            f"NLL="
            f"{val_nll:.6f} | "

            f"PPL="
            f"{val_ppl:.6f} | "

            f"BPB="
            f"{val_bpb:.6f} | "

            f"bytes="
            f"{val_bytes:,}"
        )


        history.append(

            {
                "update":
                    update,

                "examples_seen":
                    examples_seen,

                "nll":
                    float(
                        val_nll
                    ),

                "ppl":
                    float(
                        val_ppl
                    ),

                "bpb":
                    float(
                        val_bpb
                    ),

                "reason":
                    "resume_validation",
            }
        )


        if val_nll < best_nll:

            best_nll = float(
                val_nll
            )


            save_best(
                val_nll,
                val_ppl,
                val_bpb,
            )


        if (
            val_nll
            <=
            TARGET_NLL
        ):

            stop_reason = (
                "validation_ppl_3_reached"
            )


    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    model.train()


    torch.cuda.reset_peak_memory_stats()


    last_train_nll = None

    last_train_ppl = None

    last_train_bpb = None


    last_completed_pass = (
        examples_seen
        //
        n_train
    )


    try:

        while (
            stop_reason is None
            and
            examples_seen
            <
            max_examples
        ):

            this_batch = min(
                physical_batch,
                max_examples
                -
                examples_seen,
            )


            indices = dataset_indices(
                n_train,
                examples_seen,
                this_batch,
                ORDER_SEEDS[
                    dataset
                ],
            )


            cpu = make_cpu_batch(
                train_blocks,
                indices,
            )


            ids = cpu.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=True,
            )


            lr = learning_rate(
                update,
                max_updates,
            )


            for group in optimizer.param_groups:

                group[
                    "lr"
                ] = lr


            optimizer.zero_grad(
                set_to_none=True
            )


            step_start = (
                time.perf_counter()
            )


            with torch.autocast(
                "cuda",
                dtype=AMP_DTYPE,
            ):

                loss = model.nll(
                    ids,
                    checkpoint_layers=True,
                )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )


            optimizer.step()


            torch.cuda.synchronize()


            update += 1


            examples_seen += int(
                this_batch
            )


            last_train_nll = float(
                loss.detach()
            )


            last_train_ppl = math.exp(
                min(
                    last_train_nll,
                    20.0,
                )
            )


            last_train_bpb = (
                last_train_nll
                /
                math.log(
                    2.0
                )
            )


            peak_alloc = (
                torch.cuda
                .max_memory_allocated()
                /
                1024**3
            )


            peak_reserved = (
                torch.cuda
                .max_memory_reserved()
                /
                1024**3
            )


            pass_fraction = (
                float(
                    examples_seen
                )
                /
                float(
                    n_train
                )
            )


            step_seconds = (
                time.perf_counter()
                -
                step_start
            )


            bytes_per_second = (
                this_batch
                *
                RAW_BYTES_PER_SEQUENCE
                /
                max(
                    step_seconds,
                    1e-9,
                )
            )


            print(

                f"{dataset:<12} | "

                f"step="
                f"{update:5d} | "

                f"pass="
                f"{pass_fraction:5.3f}/"
                f"{MAX_PASSES} | "

                f"B="
                f"{this_batch:<3d} | "

                f"NLL="
                f"{last_train_nll:7.4f} | "

                f"PPL="
                f"{last_train_ppl:8.3f} | "

                f"BPB="
                f"{last_train_bpb:6.3f} | "

                f"lr="
                f"{lr:.2e} | "

                f"bytes/s="
                f"{bytes_per_second:,.0f} | "

                f"VRAM="
                f"{peak_alloc:.2f}/"
                f"{peak_reserved:.2f} GiB"
            )


            del (
                indices,
                cpu,
                ids,
                loss,
            )


            if (
                update
                %
                SAVE_EVERY_STEPS
                ==
                0
            ):

                save_latest()


            completed_pass = (
                examples_seen
                //
                n_train
            )


            crossed_pass = (
                completed_pass
                >
                last_completed_pass
            )


            if crossed_pass:

                last_completed_pass = (
                    completed_pass
                )


            if (
                update
                %
                VALIDATE_EVERY_STEPS
                ==
                0

                or

                crossed_pass
            ):

                print(

                    f"\n{name}: "
                    f"FULL GENUINE CAUSAL VALIDATION "
                    f"@ update {update}"
                )


                (
                    val_nll,
                    val_ppl,
                    val_bpb,
                    val_bytes,
                ) = validate_model(
                    model,
                    valid_blocks,
                    physical_batch,
                )


                print(

                    f"VALID | "

                    f"NLL="
                    f"{val_nll:.6f} | "

                    f"PPL="
                    f"{val_ppl:.6f} | "

                    f"BPB="
                    f"{val_bpb:.6f} | "

                    f"bytes="
                    f"{val_bytes:,}"
                )


                history.append(

                    {
                        "update":
                            update,

                        "examples_seen":
                            examples_seen,

                        "training_nll":
                            last_train_nll,

                        "training_ppl":
                            last_train_ppl,

                        "training_bpb":
                            last_train_bpb,

                        "nll":
                            float(
                                val_nll
                            ),

                        "ppl":
                            float(
                                val_ppl
                            ),

                        "bpb":
                            float(
                                val_bpb
                            ),

                        "reason":
                            (
                                "pass_boundary"
                                if crossed_pass
                                else
                                "periodic"
                            ),
                    }
                )


                if (
                    val_nll
                    <
                    best_nll
                ):

                    best_nll = float(
                        val_nll
                    )


                    save_best(
                        val_nll,
                        val_ppl,
                        val_bpb,
                    )


                    print(
                        "NEW BEST GENUINE CAUSAL MODEL"
                    )


                save_latest()


                if (
                    val_nll
                    <=
                    TARGET_NLL
                ):

                    stop_reason = (
                        "validation_ppl_3_reached"
                    )


                    print(
                        "=" * 100
                    )


                    print(
                        f"{name}: "
                        f"GENUINE PPL <= 3 REACHED"
                    )


                    print(

                        f"NLL="
                        f"{val_nll:.6f} | "

                        f"PPL="
                        f"{val_ppl:.6f} | "

                        f"BPB="
                        f"{val_bpb:.6f}"
                    )


                    print(
                        "=" * 100
                    )


                    break


    except BaseException:

        try:

            save_latest()


            print(
                f"{name}: "
                f"emergency checkpoint saved."
            )


        except Exception as save_error:

            print(

                f"{name}: "
                f"emergency save failed: "
                f"{save_error}"
            )


        raise


    # -------------------------------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------------------------------

    if stop_reason is None:

        stop_reason = (
            "three_pass_limit"
        )


        print(

            f"\n{name}: "
            f"FINAL GENUINE CAUSAL VALIDATION"
        )


        (
            val_nll,
            val_ppl,
            val_bpb,
            val_bytes,
        ) = validate_model(
            model,
            valid_blocks,
            physical_batch,
        )


        print(

            f"FINAL | "

            f"NLL="
            f"{val_nll:.6f} | "

            f"PPL="
            f"{val_ppl:.6f} | "

            f"BPB="
            f"{val_bpb:.6f} | "

            f"bytes="
            f"{val_bytes:,}"
        )


        history.append(

            {
                "update":
                    update,

                "examples_seen":
                    examples_seen,

                "training_nll":
                    last_train_nll,

                "training_ppl":
                    last_train_ppl,

                "training_bpb":
                    last_train_bpb,

                "nll":
                    float(
                        val_nll
                    ),

                "ppl":
                    float(
                        val_ppl
                    ),

                "bpb":
                    float(
                        val_bpb
                    ),

                "reason":
                    "final_validation",
            }
        )


        if (
            val_nll
            <
            best_nll
        ):

            best_nll = float(
                val_nll
            )


            save_best(
                val_nll,
                val_ppl,
                val_bpb,
            )


        if (
            val_nll
            <=
            TARGET_NLL
        ):

            stop_reason = (
                "validation_ppl_3_reached"
            )


    if not history:

        raise RuntimeError(
            f"{dataset}: "
            f"no validation result generated."
        )


    best = min(

        history,

        key=lambda row:
            row[
                "nll"
            ],
    )


    target_reached = bool(
        best[
            "nll"
        ]
        <=
        TARGET_NLL
    )


    result = {

        "status":
            "complete",

        "fingerprint":
            run_fingerprint,

        "architecture":
            ARCH,

        "dataset":
            dataset,

        "display_name":
            name,

        "objective":
            "exact_causal_next_byte_nll",

        "tokenization":
            "NONE_raw_utf8_bytes_learned_block_lifting",

        "future_leakage":
            False,

        "physical_batch":
            physical_batch,

        "maximum_passes":
            MAX_PASSES,

        "updates":
            update,

        "examples_seen":
            examples_seen,

        "effective_passes":
            float(
                examples_seen
            )
            /
            float(
                n_train
            ),

        "train_sequences":
            n_train,

        "validation_sequences":
            n_valid,

        "raw_context_bytes":
            RAW_BYTES_PER_SEQUENCE,

        "model_context_blocks":
            MODEL_CONTEXT,

        "parameter_count":
            total_params,

        "trainable_parameter_count":
            trainable_params,

        "target_nll":
            TARGET_NLL,

        "target_ppl":
            TARGET_PPL,

        "target_reached":
            target_reached,

        "best_validation_nll":
            float(
                best[
                    "nll"
                ]
            ),

        "best_validation_ppl":
            float(
                best[
                    "ppl"
                ]
            ),

        "best_validation_bpb":
            float(
                best[
                    "bpb"
                ]
            ),

        "best_validation_update":
            int(
                best[
                    "update"
                ]
            ),

        "last_training_nll":
            last_train_nll,

        "last_training_ppl":
            last_train_ppl,

        "last_training_bpb":
            last_train_bpb,

        "stop_reason":
            stop_reason,

        "training_seconds":
            elapsed(),

        "peak_allocated_vram_gb":
            float(
                torch.cuda
                .max_memory_allocated()
                /
                1024**3
            ),

        "peak_reserved_vram_gb":
            float(
                torch.cuda
                .max_memory_reserved()
                /
                1024**3
            ),

        "validation_history":
            history,
    }


    atomic_torch_save(

        {
            **result,

            "model_state":
                model.state_dict(),
        },

        final_path,
    )


    atomic_json(
        result,
        result_path,
    )


    latest_path.unlink(
        missing_ok=True
    )


    print(
        "=" * 120
    )


    print(
        f"{name} FINISHED"
    )


    print(

        f"best genuine NLL : "
        f"{result['best_validation_nll']:.6f}"
    )


    print(

        f"best genuine PPL : "
        f"{result['best_validation_ppl']:.6f}"
    )


    print(

        f"best BPB         : "
        f"{result['best_validation_bpb']:.6f}"
    )


    print(
        f"target reached   : "
        f"{target_reached}"
    )


    print(
        f"stop reason      : "
        f"{stop_reason}"
    )


    print(

        f"peak VRAM        : "

        f"{result['peak_allocated_vram_gb']:.2f}/"

        f"{result['peak_reserved_vram_gb']:.2f} GiB"
    )


    print(
        f"model            : "
        f"{final_path}"
    )


    print(
        f"result           : "
        f"{result_path}"
    )


    print(
        "=" * 120
    )


    del (
        model,
        optimizer,
        train_mem,
        train_blocks,
        valid_mem,
        valid_blocks,
    )


    cleanup_cuda()


    return result


# =============================================================================
# START
# =============================================================================

print(
    "=" * 120
)


print(
    "CAUSAL TRANSWAVE — REAL AUTOREGRESSIVE BYTE LANGUAGE MODEL"
)


print(
    f"GPU                     : "
    f"{GPU_NAME}"
)


print(
    f"VRAM                    : "
    f"{GPU_TOTAL_GB:.2f} GiB"
)


print(
    "offline tokenization    : NONE"
)


print(
    "BPE / SentencePiece     : NONE"
)


print(
    "MLM / MASK corruption   : NONE"
)


print(
    "objective               : exact p(x_t | x_<t)"
)


print(
    "lifting                 : "
    "learned 3-level block-local Haar-initialized lifting"
)


print(
    "Meyer temporal operator : "
    "strictly causal zero-padded FFT linear FIR"
)


print(
    "local convolution       : "
    "strictly left-causal"
)


print(
    "within-block decoder    : "
    "exact 8-byte autoregressive factorization"
)


print(

    f"raw context             : "
    f"{RAW_BYTES_PER_SEQUENCE:,} bytes"
)


print(

    f"model context           : "
    f"{MODEL_CONTEXT:,} eight-byte blocks"
)


print(
    f"layers                  : "
    f"{N_LAYERS}"
)


print(
    f"dimension               : "
    f"{MODEL_DIM}"
)


print(
    f"Meyer bands             : "
    f"{N_MEYER_BANDS}"
)


print(
    f"controls / band         : "
    f"{N_BAND_CONTROLS}"
)


print(

    f"Meyer causal FIR length : "
    f"{MEYER_KERNEL} block positions"
)


print(

    f"REAL TARGET NLL         : "
    f"{TARGET_NLL:.9f}"
)


print(

    f"REAL TARGET PPL         : "
    f"{TARGET_PPL:.6f}"
)


print(
    "=" * 120
)


# =============================================================================
# MAXIMIZE G4 VRAM
# =============================================================================

PHYSICAL_BATCH = choose_batch_size()


# =============================================================================
# FOUR INDEPENDENT CORPUS RUNS
# =============================================================================

ALL_RESULTS = []


for dataset in DATASETS:

    ALL_RESULTS.append(

        train_dataset(
            dataset,
            PHYSICAL_BATCH,
        )
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n"
    +
    "=" * 120
)


print(
    "CAUSAL TRANSWAVE — GENUINE HELD-OUT AUTOREGRESSIVE RESULTS"
)


print(
    "=" * 120
)


print(

    f"{'DATASET':<18}"

    f"{'NLL':>14}"

    f"{'PPL':>14}"

    f"{'BPB':>14}"

    f"{'PASSES':>12}"

    f"{'TARGET':>12}"
)


print(
    "-" * 84
)


for row in ALL_RESULTS:

    print(

        f"{row['display_name']:<18}"

        f"{row['best_validation_nll']:>14.6f}"

        f"{row['best_validation_ppl']:>14.6f}"

        f"{row['best_validation_bpb']:>14.6f}"

        f"{row['effective_passes']:>12.3f}"

        f"{('YES' if row['target_reached'] else 'NO'):>12}"
    )


print(
    "-" * 84
)


print(

    f"Stopping target: "
    f"genuine NLL <= "
    f"{TARGET_NLL:.9f} | "

    f"genuine PPL <= "
    f"{TARGET_PPL:.6f}"
)


print(

    "PPL = exp(exact held-out causal byte NLL). "
    "It is not an MLM reconstruction score."
)


print(
    "=" * 120
)