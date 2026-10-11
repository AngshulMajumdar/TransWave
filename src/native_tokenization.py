# CELL 1 — NATIVE TOKENIZATION, NO GATED REPOSITORIES
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path

import numpy as np

from transformers import AutoTokenizer


# =============================================================================
# PATHS
# =============================================================================

RAW_ROOT = Path(
    "/content/drive/MyDrive/spliNet/raw_text"
)

ROOT = Path(
    "/content/drive/MyDrive/"
    "TransWave_Meyer_Context/"
    "commercial_transformer_baselines_v2"
)

TOKEN_ROOT = (
    ROOT
    /
    "native_tokens"
)

TOKEN_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# DATASETS
# =============================================================================

DATASETS = (
    "arxiv",
    "fineweb_edu",
    "pg19",
    "wikitext103",
)

SPLITS = (
    "train",
    "validation",
)


# =============================================================================
# PUBLIC REPOSITORIES
#
# We use these only for:
#   tokenizer
#   architecture config
#
# NO pretrained weights are loaded.
# =============================================================================

MODEL_SPECS = {

    "llama": {

        "display":
            "LLaMA-3.2",

        "family":
            "meta-llama/Llama-3.2-1B",

        "repo":
            "unsloth/Llama-3.2-1B",
    },

    "gemma": {

        "display":
            "Gemma-2",

        "family":
            "google/gemma-2-2b",

        "repo":
            "unsloth/gemma-2-2b",
    },

    "qwen": {

        "display":
            "Qwen2.5",

        "family":
            "Qwen/Qwen2.5-1.5B",

        "repo":
            "Qwen/Qwen2.5-1.5B",
    },
}


# =============================================================================
# STREAMING TOKENIZATION
# =============================================================================

CHARS_PER_CHUNK = 1_000_000

TOKEN_DTYPE = np.uint32


def tokenizer_signature(
    tokenizer,
    repo,
):

    payload = {

        "repo":
            repo,

        "tokenizer_class":
            tokenizer.__class__.__name__,

        "vocab_size":
            len(
                tokenizer
            ),

        "bos_token_id":
            tokenizer.bos_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "pad_token_id":
            tokenizer.pad_token_id,

        "add_bos_token":
            bool(
                getattr(
                    tokenizer,
                    "add_bos_token",
                    False,
                )
            ),

        "add_eos_token":
            bool(
                getattr(
                    tokenizer,
                    "add_eos_token",
                    False,
                )
            ),
    }


    return hashlib.sha256(

        json.dumps(
            payload,
            sort_keys=True,
        ).encode()

    ).hexdigest()


def native_prefix_suffix(
    tokenizer,
):

    prefix = []

    suffix = []


    if (
        bool(
            getattr(
                tokenizer,
                "add_bos_token",
                False,
            )
        )
        and
        tokenizer.bos_token_id
        is not None
    ):

        prefix.append(
            int(
                tokenizer.bos_token_id
            )
        )


    if (
        bool(
            getattr(
                tokenizer,
                "add_eos_token",
                False,
            )
        )
        and
        tokenizer.eos_token_id
        is not None
    ):

        suffix.append(
            int(
                tokenizer.eos_token_id
            )
        )


    return (
        prefix,
        suffix,
    )


def tokenize_file(
    model_key,
    dataset,
    split,
    tokenizer,
    repo,
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
            src
        )


    out_dir = (
        TOKEN_ROOT
        /
        model_key
        /
        dataset
    )


    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    bin_path = (
        out_dir
        /
        f"{split}.uint32.bin"
    )


    meta_path = (
        out_dir
        /
        f"{split}.json"
    )


    signature = tokenizer_signature(
        tokenizer,
        repo,
    )


    source_bytes = (
        src.stat().st_size
    )


    # -------------------------------------------------------------------------
    # RESTART SAFE
    # -------------------------------------------------------------------------

    if (
        bin_path.is_file()
        and
        meta_path.is_file()
    ):

        try:

            old = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )


            expected_size = (
                int(
                    old[
                        "token_count"
                    ]
                )
                *
                np.dtype(
                    TOKEN_DTYPE
                ).itemsize
            )


            if (
                old.get(
                    "status"
                )
                ==
                "complete"

                and

                old.get(
                    "tokenizer_signature"
                )
                ==
                signature

                and

                int(
                    old.get(
                        "source_bytes",
                        -1,
                    )
                )
                ==
                source_bytes

                and

                bin_path.stat().st_size
                ==
                expected_size
            ):

                print(

                    f"{model_key:<6} | "
                    f"{dataset:<12} | "
                    f"{split:<10} | "
                    f"CACHED | "
                    f"{old['token_count']:,} tokens"
                )


                return old


        except Exception:

            pass


    tmp_bin = bin_path.with_suffix(
        ".uint32.bin.tmp"
    )


    tmp_meta = meta_path.with_suffix(
        ".json.tmp"
    )


    tmp_bin.unlink(
        missing_ok=True
    )


    tmp_meta.unlink(
        missing_ok=True
    )


    prefix, suffix = (
        native_prefix_suffix(
            tokenizer
        )
    )


    token_count = 0


    print()

    print(
        f"{MODEL_SPECS[model_key]['display']} | "
        f"{dataset} | {split}"
    )


    print(
        f"  tokenizer : {repo}"
    )


    print(
        f"  raw       : "
        f"{source_bytes / 1024**2:.1f} MiB"
    )


    print(
        f"  BOS       : "
        f"{prefix if prefix else 'none'}"
    )


    print(
        f"  EOS       : "
        f"{suffix if suffix else 'none'}"
    )


    old_model_max_length = (
        tokenizer.model_max_length
    )


    tokenizer.model_max_length = (
        10**12
    )


    try:

        with open(
            src,
            "r",
            encoding="utf-8",
            errors="strict",
        ) as fin, open(
            tmp_bin,
            "wb",
        ) as fout:


            if prefix:

                a = np.asarray(
                    prefix,
                    dtype=TOKEN_DTYPE,
                )


                a.tofile(
                    fout
                )


                token_count += int(
                    a.size
                )


            chars_done = 0


            while True:

                text = fin.read(
                    CHARS_PER_CHUNK
                )


                if not text:

                    break


                ids = tokenizer(

                    text,

                    add_special_tokens=False,

                    truncation=False,

                    return_attention_mask=False,

                )[
                    "input_ids"
                ]


                a = np.asarray(
                    ids,
                    dtype=TOKEN_DTYPE,
                )


                if a.size:

                    if (
                        int(
                            a.max()
                        )
                        >=
                        2**32
                    ):

                        raise RuntimeError(
                            "Token ID exceeds uint32."
                        )


                    a.tofile(
                        fout
                    )


                    token_count += int(
                        a.size
                    )


                chars_done += len(
                    text
                )


                if (
                    chars_done
                    %
                    20_000_000
                    <
                    CHARS_PER_CHUNK
                ):

                    print(

                        f"  ~"
                        f"{chars_done / 1e6:.0f}M chars | "
                        f"{token_count:,} tokens"
                    )


            if suffix:

                a = np.asarray(
                    suffix,
                    dtype=TOKEN_DTYPE,
                )


                a.tofile(
                    fout
                )


                token_count += int(
                    a.size
                )


    finally:

        tokenizer.model_max_length = (
            old_model_max_length
        )


    os.replace(
        tmp_bin,
        bin_path,
    )


    meta = {

        "status":
            "complete",

        "model_key":
            model_key,

        "display":
            MODEL_SPECS[
                model_key
            ][
                "display"
            ],

        "family":
            MODEL_SPECS[
                model_key
            ][
                "family"
            ],

        "tokenizer_repo":
            repo,

        "tokenizer_class":
            tokenizer.__class__.__name__,

        "tokenizer_signature":
            signature,

        "vocab_size":
            len(
                tokenizer
            ),

        "source_path":
            str(
                src
            ),

        "source_bytes":
            source_bytes,

        "token_count":
            token_count,

        "dtype":
            "uint32",

        "bos_token_id":
            tokenizer.bos_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "pad_token_id":
            tokenizer.pad_token_id,

        "add_bos_token":
            bool(
                getattr(
                    tokenizer,
                    "add_bos_token",
                    False,
                )
            ),

        "add_eos_token":
            bool(
                getattr(
                    tokenizer,
                    "add_eos_token",
                    False,
                )
            ),
    }


    tmp_meta.write_text(

        json.dumps(
            meta,
            indent=2,
            sort_keys=True,
        ),

        encoding="utf-8",
    )


    os.replace(
        tmp_meta,
        meta_path,
    )


    print(

        f"  DONE      : "
        f"{token_count:,} tokens | "

        f"{token_count/source_bytes:.4f} "
        f"tokens/raw-byte"
    )


    return meta


# =============================================================================
# LOAD AND VERIFY THE THREE TOKENIZERS FIRST
#
# Fail before processing 12 huge streams if any repository is unavailable.
# =============================================================================

TOKENIZERS = {}


print(
    "=" * 110
)


print(
    "VERIFYING ALL THREE PUBLIC NATIVE TOKENIZERS"
)


print(
    "=" * 110
)


for model_key, spec in MODEL_SPECS.items():

    print(

        f"{spec['display']:<12} | "
        f"{spec['repo']}"
    )


    tok = AutoTokenizer.from_pretrained(
        spec[
            "repo"
        ],
        use_fast=True,
    )


    TOKENIZERS[
        model_key
    ] = tok


    probe = tok(
        "The quick brown fox jumps over the lazy dog.",
        add_special_tokens=False,
    )[
        "input_ids"
    ]


    print(

        f"  class="
        f"{tok.__class__.__name__} | "

        f"vocab="
        f"{len(tok):,} | "

        f"probe tokens="
        f"{len(probe)}"
    )


print(
    "=" * 110
)


print(
    "ALL THREE TOKENIZERS ACCESSIBLE"
)


print(
    "=" * 110
)


# =============================================================================
# TOKENIZE 3 x 4 x 2 STREAMS
# =============================================================================

TOKEN_META = {}


for model_key, spec in MODEL_SPECS.items():

    tokenizer = TOKENIZERS[
        model_key
    ]


    TOKEN_META[
        model_key
    ] = {}


    print()

    print(
        "=" * 110
    )


    print(

        f"NATIVE TOKENIZATION: "
        f"{spec['display']} | "
        f"{spec['repo']}"
    )


    print(
        "=" * 110
    )


    for dataset in DATASETS:

        TOKEN_META[
            model_key
        ][
            dataset
        ] = {}


        for split in SPLITS:

            TOKEN_META[
                model_key
            ][
                dataset
            ][
                split
            ] = tokenize_file(

                model_key,

                dataset,

                split,

                tokenizer,

                spec[
                    "repo"
                ],
            )


# =============================================================================
# SUMMARY
# =============================================================================

print()

print(
    "=" * 110
)


print(
    "NATIVE TOKENIZATION COMPLETE"
)


print(
    "=" * 110
)


for model_key in MODEL_SPECS:

    for dataset in DATASETS:

        train_n = (

            TOKEN_META[
                model_key
            ][
                dataset
            ][
                "train"
            ][
                "token_count"
            ]
        )


        valid_n = (

            TOKEN_META[
                model_key
            ][
                dataset
            ][
                "validation"
            ][
                "token_count"
            ]
        )


        print(

            f"{model_key:<6} | "
            f"{dataset:<12} | "

            f"train="
            f"{train_n:,} | "

            f"validation="
            f"{valid_n:,}"
        )