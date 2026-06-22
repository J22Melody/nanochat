# nanochat: An Annotated Walkthrough

A chapter-by-chapter tutorial built while reading and running Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat) on an HPC/SLURM cluster. The aim is to
refresh the modern LLM tech stack — pretraining, post-training, tokenization, inference —
through real code, with the insightful Q&A preserved as callout boxes.

Intended to be shareable with students and colleagues.

## Chapters

| # | Chapter | Status |
|---|---|---|
| 00 | [Overview of nanochat](00_overview.md) | ✅ |
| 01 | [Environment Setup](01_environment_setup.md) — uv, scratch vs. home quota, SLURM | ✅ |
| 02 | [Data Preparation](02_data_preparation.md) — ClimbMix, shards, two data loaders | ✅ |
| 03 | [Pre-tokenization](03_pretokenization.md) — the GPT-4 split regex, the 7 rules | ✅ |
| 04 | [BPE and the Byte Vocabulary](04_bpe_and_byte_vocabulary.md) — 256 bytes, vocab math, special tokens | ✅ |
| 05 | [The BPE Merge Loop](05_bpe_merge_loop.md) — training algorithm, dedup table, heap/patch efficiency, scale | ✅ |
| 06 | [Encoding and Evaluating the Tokenizer](06_encoding_and_evaluation.md) — merge replay, the 3 saved artifacts, compression ratio, bits-per-byte | ✅ |
| 07 | [Tokenization Deep-Dives](07_tokenization_deep_dives.md) — multilingual, morphology, SuperBPE, parity-aware BPE, byte/pixel models | ✅ |
| 08 | [Sizing the Model: Config, Parameters & Scaling Laws](08_model_config_and_scaling.md) — one `depth` knob, meta-device init, param counts, Chinchilla/Power-Lines/muP hyperparameters | ✅ |

## Coming next (day 2+)

- The transformer architecture ([`nanochat/gpt.py`](../nanochat/gpt.py)): embeddings, attention, blocks, forward pass
- Pretraining loop ([`base_train.py`](../scripts/base_train.py)), optimizer & LR schedule (Muon + AdamW)
- Evaluation (CORE/DCLM, bits-per-byte)
- Post-training: SFT, RL
- Inference & serving (KV cache, sampling)

## Conventions

- Code references link into the repo with line anchors (e.g.
  [`../nanochat/tokenizer.py#L30`](../nanochat/tokenizer.py#L30)) — work in-IDE and on GitHub.
- `> **Q:** / **A:**` callouts preserve the discussion that produced each insight.
- Numbers (sizes, token counts, timings) are measured from an actual run, not estimated.
- Research-frontier claims in Ch07 are verified against cited sources.

The full method and style guide is in **[AUTHORING_PRINCIPLES.md](AUTHORING_PRINCIPLES.md)** —
read it before adding or editing a chapter.
