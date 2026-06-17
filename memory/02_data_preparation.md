# Chapter 2 — Data Preparation

> What the pretraining data actually *is*, how it's stored, and how it's streamed —
> including the important detail that BPE training and LLM training read the **same
> source** through **different loaders**.

## The dataset: ClimbMix-400B

nanochat pretrains on **ClimbMix-400B**
([huggingface.co/datasets/karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)),
a large **English web-text** corpus. (The repo switched from
[FineWeb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) → ClimbMix
in March 2026; ClimbMix derives from the [ClimbLab](https://arxiv.org/abs/2504.13161)
NVIDIA work on clustered/classified web data.) Defined in
[`nanochat/dataset.py`](../nanochat/dataset.py):

- **Source:** [`BASE_URL`](../nanochat/dataset.py#L23) on HuggingFace
- **Format:** [Parquet](https://parquet.apache.org/) shards, `shard_00000.parquet` …
  `shard_06542.parquet` (6543 total)
- **Stored at:** `$NANOCHAT_BASE_DIR/base_data_climbmix/` (see [Ch01](01_environment_setup.md))

"climbmix" = a curated *mixture* of filtered/quality-classified web data. "400b" = ~400B
tokens in the full set. **"shuffle"** = documents are pre-shuffled across shards, so
reading the first N shards already gives an unbiased random sample (important — you can
train on a prefix of shards without bias).

### What a shard looks like (measured)

| Property | Value |
|---|---|
| One shard | ~92 MB on disk, **86,016 documents**, **~253M chars** |
| Schema | a single column: **`text`** (string). That's it — no labels, no metadata |
| Doc length | mean ~2,939 chars, median ~2,452, min 5, max ~309K |
| One row group | 1024 docs ≈ ~2.9M chars ≈ **~3 MB** |

The documents are plain English web prose — news, explainers, how-tos. **No Q&A, no chat
structure.** That conversational format only appears later in SFT, which is why the chat
special tokens are reserved now but unused (see [Ch04](04_bpe_and_byte_vocabulary.md#the-special-tokens-9)).

> **Q:** Why web text instead of clean/curated data?
>
> **A:** For pretraining, **scale and diversity beat cleanliness.** You want broad world
> knowledge and language coverage from the next-token objective. You *align* behavior
> later with SFT/RL on much smaller, high-quality data. The single `text` field means
> the model only ever learns next-token prediction — no supervision beyond "what comes
> next."

## Downloading shards

```bash
python -m nanochat.dataset -n 8     # download first 8 shards (~800 MB) — enough for tokenizer
python -m nanochat.dataset -n 170   # ~170 shards (~20 GB) — enough for GPT-2-grade pretraining
```

Implementation detail worth knowing: the **validation shard is always the last shard**
(`shard_06542`) and is *always* downloaded in addition to the `-n` train shards. So
`train` = all shards except the last; `val` = the last shard. This split convention is
shared by both data loaders below.

## The crucial distinction: two loaders, one source

BPE training and LLM pretraining read the **same parquet shards** but through **different
functions** for different purposes.

### Loader A — for BPE training: `parquets_iter_batched` (simple)

[`dataset.py:67`](../nanochat/dataset.py#L67):

```python
def parquets_iter_batched(split, start=0, step=1):
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]  # last = val
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()   # list of strings
            yield texts                              # yield a BATCH of doc strings
```

It yields **raw text strings**, one row group at a time. No tokens, no BOS, no fixed
length — the tokenizer doesn't exist yet (it's being *trained* on this text).
[`tok_train.py:28`](../scripts/tok_train.py#L28) wraps it to crop each doc to 10k chars
and stop at 2B chars total.

### Loader B — for LLM pretraining: `tokenizing_..._bos_bestfit` (complex)

[`nanochat/dataloader.py:74`](../nanochat/dataloader.py#L74) does far more, because the
model needs fixed-shape token tensors:

1. Reads the same text, immediately **tokenizes** it, prepending `<|bos|>` to every doc.
2. **Packs documents into fixed-length rows of `T+1` tokens** with a *best-fit* algorithm
   (every row starts with BOS; place the largest doc that fits; crop to fill when nothing
   fits → **100% utilization, no padding, ~35% of tokens cropped** at T=2048).
3. Builds the **shifted `(inputs, targets)`** pair: `inputs = row[:-1]`, `targets = row[1:]`
   — that *is* the next-token-prediction objective.
4. **DDP sharding**: each GPU reads different row groups (`rg_idx = ddp_rank; += world_size`).
5. **Infinite / multi-epoch**, resumable, with pinned-memory + a single host→device copy.

| | BPE training (A) | LLM pretraining (B) |
|---|---|---|
| Yields | raw **text strings** | token tensors `(B,T)` inputs+targets |
| Tokenizer | being *created* | already trained, used to *encode* |
| BOS | none | every row starts with `<|bos|>` |
| Length | crop to 10k chars | best-fit pack to exactly `T+1` tokens |
| Output | variable | fixed `(B, T)` |

## Memory: it's a lazy stream, not one big list

> **Q:** It returns a "list of text strings" — is the whole 2B chars in memory?
>
> **A:** No. The functions are **generators** (`yield`). At any instant only **one row
> group (~3 MB, 1024 docs)** is materialized as a Python list; it's yielded, then becomes
> eligible for GC before the next is read. The shard (253 MB) and full set (~GBs) are
> never fully loaded.
>
> **Caveat:** the data *loader* is memory-light, but the **BPE trainer itself** does
> accumulate the corpus internally (as pre-tokenized **word-frequency counts**, not the
> raw 2B-char string) so it can repeatedly find the most-frequent pair. Identical words
> collapse to one counted entry, so it's far smaller than 2B chars — but it's "the whole
> corpus's statistics," not a 3 MB window. This streaming pattern is what lets you train
> on 400B tokens without ever loading a shard fully.
