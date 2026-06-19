# Chapter 0 — Overview of nanochat

> A walkthrough of Andrej Karpathy's **nanochat** ([github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)),
> a minimal but *complete* full-stack ChatGPT clone. The goal of this tutorial series is
> to refresh the modern LLM tech stack — pretraining, post-training, tokenization,
> inference — by reading and running real code, end to end.
>
> Related lineage: [nanoGPT](https://github.com/karpathy/nanoGPT) (the pretraining core),
> [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (the speedrun culture +
> Muon optimizer), and [minbpe](https://github.com/karpathy/minbpe) (the tokenizer ideas).

## Why nanochat?

Most "learn LLMs" resources show you *one* slice (a transformer block, or an RLHF loss).
nanochat is valuable because it is the **entire pipeline** in readable, framework-free
code: tokenizer → pretraining → midtraining → SFT → RL → inference/eval. Every stage
maps to a small, self-contained file you can actually read in an afternoon.

## The pipeline at a glance

The single source of truth is [`runs/speedrun.sh`](../runs/speedrun.sh) — the whole
"blank GPU box → chatbot you can talk to" pipeline in ~98 lines, designed for an
**8×H100 node, ~3 hours, ~$100** of compute.

The conceptual stages:

| # | Stage | Entry script | Core module | What it teaches | Tutorial ch. |
|---|---|---|---|---|---|
| 1 | **Tokenization** | [`tok_train.py`](../scripts/tok_train.py) | [`tokenizer.py`](../nanochat/tokenizer.py) | BPE, byte vocab, special tokens, merge loop, encoding, eval | [03](03_pretokenization.md), [04](04_bpe_and_byte_vocabulary.md), [05](05_bpe_merge_loop.md), [06](06_encoding_and_evaluation.md), [07](07_tokenization_deep_dives.md) |
| 2 | **Pretraining** | [`base_train.py`](../scripts/base_train.py) | [`gpt.py`](../nanochat/gpt.py), [`optim.py`](../nanochat/optim.py), [`dataloader.py`](../nanochat/dataloader.py) | transformer, attention, next-token loss, LR schedule, DDP | TBC |
| 3 | **Base eval** | [`base_eval.py`](../scripts/base_eval.py) | [`core_eval.py`](../nanochat/core_eval.py), [`loss_eval.py`](../nanochat/loss_eval.py) | perplexity, CORE/DCLM score, bits-per-byte | TBC |
| 4 | **SFT** | [`chat_sft.py`](../scripts/chat_sft.py) | [`tasks/`](../tasks/), [`engine.py`](../nanochat/engine.py) | instruction tuning, chat templates, loss masking | TBC |
| 5 | **RL** | [`chat_rl.py`](../scripts/chat_rl.py) | [`engine.py`](../nanochat/engine.py), [`tasks/gsm8k.py`](../tasks/gsm8k.py) | GRPO-style RL, reward, policy gradient | TBC |
| 6 | **Inference** | [`chat_cli.py`](../scripts/chat_cli.py), [`chat_web.py`](../scripts/chat_web.py) | [`engine.py`](../nanochat/engine.py) | KV cache, sampling, serving | TBC |

## What the speedrun actually does (read [`runs/speedrun.sh`](../runs/speedrun.sh))

1. **Env setup** — installs `uv`, creates `.venv`, syncs deps, resets a markdown report.
2. **Tokenizer** — downloads ~2B chars of text, trains a BPE tokenizer (vocab 32768),
   evaluates compression. (Kicks off downloading the *full* ~170 shards in the
   background while the tokenizer trains — overlapping I/O with compute.)
3. **Pretraining** — `torchrun --nproc_per_node=8` trains a depth-24 GPT from scratch
   ([speedrun.sh#L73](../runs/speedrun.sh#L73)). Notable: `--target-param-data-ratio=8`
   deliberately *undertrains* (8 tokens/param vs the compute-optimal ~10.5 ratio from the
   [Chinchilla paper](https://arxiv.org/abs/2203.15556)) — just enough to beat GPT-2,
   saving compute. Uses `--fp8` for speed.
4. **SFT** — downloads synthetic "identity" conversations, finetunes the base model to
   use chat special tokens, tool use, multiple choice.
5. **Talk to it** — `chat_cli` (terminal) or `chat_web` (ChatGPT-style UI).
6. **Report** — stitches all sections into `report.md`, a scorecard of the run.

> **Note:** the *default* speedrun is tokenizer → pretrain → SFT only. **RL and
> midtraining exist in the repo but are skipped** by the default speedrun. The clean
> mental model: tokenize → pretrain → SFT → eval.

## Hardware: what you actually need

- **Reference speedrun:** 8×H100 (80GB), ~3h.
- **Single GPU:** drop `torchrun` — the code auto-switches to gradient accumulation and
  produces ~identical results, but ~8× slower. For <80GB cards, lower
  `--device-batch-size` (32 → 16 → 8 → …) to avoid OOM.
- **For a learning walkthrough** (what this series does): one H100/A100 80GB and a
  *smaller* `--depth` finishes in a couple hours and exercises every stage.

## How to read this series

Go in pipeline order. For each stage: read the **script** (the what/why), then the
**module** it calls (the how). The two highest-leverage files for interviews are
[`nanochat/gpt.py`](../nanochat/gpt.py) (the model) and
[`nanochat/optim.py`](../nanochat/optim.py) (optimizer + LR schedule, uses the
[Muon optimizer](https://kellerjordan.github.io/posts/muon/) alongside AdamW — a great
"tell me something modern" talking point).

---

*Chapters 1–7 cover environment, data, and the full tokenization story — pre-tokenization,
the byte vocabulary, the merge loop, encoding + evaluation, and the research frontier. Later
chapters will continue into the model and training.*
