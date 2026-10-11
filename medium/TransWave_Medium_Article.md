# TransWave: Causal Language Modeling Through Learned Multiscale Structure

Transformer language models made attention the default mechanism for long-range interaction. State-space and long-convolution models then showed that attention is not the only viable route. TransWave asks a different question: can language be modeled efficiently by making multiscale structure explicit from the first byte onward?

TransWave is a causal byte-level language model built around learned lifting and Meyer-shaped sequence mixing. It does not use a BPE or SentencePiece vocabulary. Raw UTF-8 bytes are grouped into completed eight-byte blocks, transformed into a learned multiscale representation, and processed by a stack of causal TransWave layers. Prediction remains fully autoregressive.

## From bytes to scales

The tokenizer is not an offline preprocessing algorithm. It is a trainable part of the network.

Each completed block of eight bytes is embedded and processed through three lifting stages. The resulting detail and coarse components are packed into eight scale slots with 96 channels each. Together they form a 768-dimensional representation.

The lifting operations are block-local. That restriction matters: a causal language model cannot allow the analysis transform itself to look across the boundary into the block being predicted.

## A causal Meyer operator

A standard FFT mixer is bidirectional because every output can depend on the full sequence. TransWave instead uses the frequency-domain parameterization to construct a one-sided finite impulse response filter. Zero-padded FFTs implement linear convolution efficiently, and only current and past lags survive.

Each layer combines this causal Meyer path with learned 8×8 cross-scale mixing, content gates, a left-padded local convolution and a SwiGLU feed-forward block.

After twelve layers, the block representation is shifted by one completed block. An intra-block autoregressive decoder then predicts the next eight bytes sequentially. The validation objective is therefore the ordinary causal likelihood

`-mean log p(x_t | x_<t)`

and perplexity is `exp(NLL)`.

## Four corpora

The current causal experiment was run independently on FineWeb-Edu, arXiv, PG19 and WikiText-103 for three corpus passes.

| Dataset | NLL | PPL | BPB |
|---|---:|---:|---:|
| FineWeb-Edu | 2.523234 | 12.468853 | 3.640257 |
| arXiv | 2.222877 | 9.233854 | 3.206933 |
| PG19 | 2.452323 | 11.615302 | 3.537955 |
| WikiText-103 | 2.486348 | 12.017311 | 3.587042 |

The model has 96,130,040 trainable parameters. On the high-memory G4 runtime used for the experiment, a complete forward/backward/AdamW probe selected physical batch 176 while reserving about 89.99 GiB of the 94.97 GiB runtime.

## Earlier efficient-sequence baselines

The repository also preserves the earlier Mamba-2, Hyena and RetNet experiment. TransWave's reported NLL/PPL is numerically lower than all three on each of the four corpora. Those older models were trained under the earlier tokenized baseline protocol, however, so their token-level NLL/PPL is not a tokenizer-independent unit. That distinction is why the next experiment uses BPB for the cross-tokenizer comparison.

## The next benchmark

The next comparison focuses on three Transformer families that matter in deployed language modeling: LLaMA-3.2, Gemma-2 and Qwen2.5.

Each family keeps its native tokenizer. The same underlying raw train/validation text is tokenized separately for each model, and each architecture is initialized from scratch. All three are run on all four corpora.

Native-token NLL and perplexity remain useful within each tokenizer. Across tokenizers, the comparison will use NLL per raw byte and bits per byte.

The native streams are already prepared. LLaMA-3.2 uses a 128,256-token vocabulary, Gemma-2 uses 256,000, and Qwen2.5 uses 151,665. The resulting training streams range from roughly 114 million to 147 million native tokens depending on corpus and tokenizer.

## Why multiscale modeling is interesting

The central idea is not simply to replace attention with another mixer. TransWave changes the representation on which global sequence modeling acts. Fine detail and coarse structure are present explicitly and can interact through learned scale mixing.

This creates a model with a different inductive bias from a flat causal Transformer, an SSM, or a long convolution. The current results justify testing that bias against the strongest Transformer baselines rather than only against alternative efficient sequence models.

The repository contains the exact experiment notebook, extracted causal TransWave code, result tables, six technical illustrations, a static GitHub Pages site, and the native tokenizer preparation for the upcoming Transformer benchmark.
