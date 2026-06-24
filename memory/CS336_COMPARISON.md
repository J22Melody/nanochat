# nanochat ↔ Stanford CS336: A Comparison

> A side-by-side of this nanochat walkthrough and **[Stanford CS336 — Language Modeling from
> Scratch](https://cs336.stanford.edu/)** (Hashimoto & Liang), which the author is taking
> concurrently. They cover **almost the same pipeline** — tokenizer → architecture → training →
> scaling → data → alignment — but from opposite directions: CS336 has you **implement** each
> piece from scratch with the *canonical* design choices; nanochat is a **complete, runnable**
> repo using *speedrun-optimized* choices that you **read and run**. Reading them together is
> high-leverage: CS336 teaches you to build the standard version, nanochat shows you the
> production-grade one.
>
> *Researched from the CS336 course site and assignment pages (June 2026). Architecture
> specifics for Assignment 1 are from the public repo/handout summaries and secondary write-ups —
> high confidence but not from reading the PDF handout directly; see Sources.*

## TL;DR — the relationship

| | **CS336** | **nanochat** |
|---|---|---|
| What it is | a **course** (5 assignments + ~19 lectures) | a **working repo** (Karpathy) + this annotated walkthrough |
| Your job | **implement** each component, pass unit tests | **read + run** a finished system end-to-end |
| Architecture | the **canonical/standard** transformer | a **speedrun-optimized** transformer (modded-nanogpt lineage) |
| Output | understanding + a minimal LM you built | a real chatbot you trained for ~$100 |
| Compute | TinyStories/OpenWebText; a scaling **API** (A3) | ClimbMix-400B; real 8×H100 speedrun |
| Breadth | broader **systems** (kernels, MoE, parallelism, inference) | one cohesive **end-to-end** pipeline |

They are **complements, not substitutes**: CS336 is *bottom-up implementation*; nanochat is *top-down reading of an optimized whole*.

## The pipeline, mapped three ways

| Stage | CS336 | nanochat | Our chapters |
|---|---|---|---|
| Tokenization (BPE) | Lecture 1 + **A1** (BPE part) | `rustbpe` + tiktoken | [03](03_pretokenization.md)–[07](07_tokenization_deep_dives.md) |
| Architecture & hyperparams | Lecture 3 + **A1** (model) | `gpt.py`, `GPTConfig` | [08](08_model_config_and_scaling.md)–[10](10_embedding_stage.md), + upcoming |
| Optimizer & training loop | **A1** (AdamW, loop) | Muon+AdamW, `base_train.py` | upcoming |
| Systems / kernels | **A2** (FlashAttention2 in Triton, distributed) | FA3 + DDP + fp8 (uses, doesn't implement) | (light) |
| Scaling laws | Lecture 9/11 + **A3** (fit via API) | hard-coded Chinchilla/Power-Lines/muP | [08](08_model_config_and_scaling.md) |
| Inference | Lecture 10 | `engine.py`, KV cache | upcoming |
| Evaluation | Lecture 12 | CORE/DCLM, bits-per-byte | [06](06_encoding_and_evaluation.md) (BPB) |
| Data curation | Lecture 13/14 + **A4** (Common Crawl) | ClimbMix (pre-prepared) | [02](02_data_preparation.md) |
| Alignment: SFT + RL | Lecture 15-17 + **A5** | `chat_sft.py`, `chat_rl.py` (GSM8K) | upcoming |

## Assignment-by-assignment

- **A1 — Basics** *(heaviest overlap with this walkthrough).* Implement a BPE tokenizer, a
  Transformer LM, cross-entropy, **AdamW**, and a training loop (cosine LR + warmup, gradient
  clipping, checkpointing, MFU), trained on **TinyStories / OpenWebText**. This is essentially
  "build nanochat's `tokenizer.py` + `gpt.py` + `base_train.py`, standard-flavored." → our
  [Ch03–10](08_model_config_and_scaling.md) and the upcoming model/training chapters.

- **A2 — Systems.** Profile/benchmark, write **FlashAttention2 in Triton**, build a
  memory-efficient **distributed** trainer. nanochat *uses* Flash Attention 3 and DDP/fp8 but
  doesn't ask you to write kernels — so **A2 is the one area CS336 goes deeper than nanochat**.

- **A3 — Scaling.** Fit a scaling law by querying a training **API** (you lack the compute to
  sweep yourself). nanochat instead **bakes in** published scaling laws (Chinchilla horizon,
  Power-Lines batch, muP LR) and *applies* them — exactly our [Ch08](08_model_config_and_scaling.md).
  CS336 makes you *derive* the curve; nanochat shows you *consuming* the result.

- **A4 — Data.** Turn raw **Common Crawl** into pretraining data via filtering + dedup. nanochat
  ships a **pre-curated** dataset (ClimbMix-400B), so it *uses* the output of an A4-style pipeline
  rather than building one — our [Ch02](02_data_preparation.md) covers the data we consume.

- **A5 — Alignment & Reasoning RL.** SFT + RL to make an LM reason on **math** (optional safety/DPO).
  Directly parallels nanochat's `chat_sft.py` / `chat_rl.py` on **GSM8K** — both do SFT then a
  GRPO-style RL for reasoning.

## Architecture diff: standard (CS336 A1) vs optimized (nanochat)

CS336 A1 teaches a clean, modern-**standard** transformer; nanochat layers on **modded-nanogpt
speedrun tricks**. Same skeleton, different parts:

| Component | CS336 A1 (standard) | nanochat (optimized) |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) — **same**, + **QK-norm** on Q/K |
| Position | RoPE | RoPE — **same** ([Ch10](10_embedding_stage.md) pointer) |
| Attention | multi-head, causal | **GQA** + sliding window + value-embeds (ResFormer) + logit softcap |
| FFN / MLP | **SwiGLU** | **ReLU²** (squared ReLU) |
| Optimizer | **AdamW** only | **Muon** (matrices) + **AdamW** (embeddings/scalars) |
| Token mixing | attention only | + **smear** (token-shift) at the input |
| Residual stream | plain `x + f(x)` | + learnable `resid_lambdas`, `x0` re-injection, `backout` |
| Precision | (bf16) | bf16 + optional **fp8** training |

So a great exercise: after CS336 A1, read nanochat's `gpt.py` and **diff** it — every extra line
(`smear`, `value_embeds`, `resid_lambdas`, ReLU², Muon, QK-norm, softcap) is a "what did the
speedrun add to the textbook model, and why?" question. Several are catalogued in our
[Ch09](09_weight_initialization.md)/[Ch10](10_embedding_stage.md) with web-verified provenance.

## Three differences in philosophy

1. **Implement vs. read.** CS336's value is in *writing and debugging* each component against unit
   tests; nanochat's is in seeing a *complete, optimized, working* system and the glue between
   stages. The first builds muscle; the second builds the map.
2. **Canonical vs. speedrun.** CS336 picks the *teachable standard* (SwiGLU, AdamW, plain
   residuals). nanochat picks whatever *wins the speedrun* (ReLU², Muon, smear, value embeds) —
   less canonical, more cutting-edge, and a window into where the frontier actually is.
3. **Systems depth.** CS336 dedicates a whole assignment + several lectures to **kernels (Triton),
   parallelism, MoE, inference systems**. nanochat uses these (FA3, DDP, fp8) as black boxes. For
   GPU-systems internals, CS336 is the deeper resource.

## How to use them together (suggested study loop)

1. **CS336 A1** → build the standard tokenizer/transformer/optimizer yourself.
2. **This walkthrough + nanochat `gpt.py`** → read the optimized version; diff against what you
   built; for each extra trick, ask "why does the speedrun do this?"
3. **CS336 A3 + our [Ch08](08_model_config_and_scaling.md)** → derive scaling laws (A3), then see
   them *applied* to size a real run (Ch08).
4. **CS336 A2** → go deep on the systems nanochat treats as black boxes (FlashAttention, distributed).
5. **CS336 A5 + nanochat SFT/RL** → compare your from-scratch RLHF/RL-reasoning to nanochat's
   `chat_rl.py` on GSM8K.

> **One line:** CS336 teaches you to **build the standard LM and its systems from scratch**;
> nanochat shows you a **complete, speedrun-optimized LM you can read and actually run**. Same
> pipeline, opposite directions — use CS336 to earn the understanding, nanochat to see the whole,
> optimized machine in motion.

## Sources

- [CS336 course site](https://cs336.stanford.edu/) · [Spring 2025 archive](https://cs336.stanford.edu/spring2025/)
- [Assignment 1 (basics) repo](https://github.com/stanford-cs336/assignment1-basics) · [DeepWiki overview](https://deepwiki.com/stanford-cs336/assignment1-basics/1-overview)
- [stanford-cs336 GitHub org](https://github.com/stanford-cs336/) (assignment repos 1–5)
- Student write-ups: [Andy Timm review](https://andytimm.github.io/posts/cs336/cs336_review.html), [bearbearyu1223 study notes](https://bearbearyu1223.github.io/cs336/2025/09/13/cs336-build-a-transformer-language-model.html)
- nanochat architecture confirmed from [`gpt.py`](../nanochat/gpt.py) directly.
