# Chapter 7 — Tokenization Deep-Dives

> Where the tokenizer story gets interesting: multilingual fairness, linguistic
> typology, and the research frontier (cross-word tokenization, fairness-aware BPE,
> and tokenizer-free byte/pixel models). All references below were verified against
> current sources (see footer).
>
> Builds on [Ch03](03_pretokenization.md) (the regex walls), [Ch04](04_bpe_and_byte_vocabulary.md)
> (the byte vocabulary), [Ch05](05_bpe_merge_loop.md) (the merge loop) and
> [Ch06](06_encoding_and_evaluation.md) (encoding + evaluation). This chapter is the
> "frontier" companion to the core pipeline — safe to skim on a first read.

## Multilingual: the "tokenization tax"

nanochat's tokenizer is trained on **English** ([ClimbMix](02_data_preparation.md)), so
non-English scripts pay a heavy cost. Measured tokens-per-character on the same sentence
("the quick brown fox"):

| Script | tokens/char | vs English |
|---|---|---|
| English | **0.21** | 1× (baseline) |
| Russian | 1.26 | ~6× |
| Arabic | 1.53 | ~7× |
| Hindi | 1.75 | ~8× |
| Japanese | 2.25 | ~11× |
| Korean | 2.38 | ~11× |
| **Chinese** | **2.71** | **~13×** |
| Emoji | 3.00 | ~14× |

A Chinese sentence costs ~13× more tokens → 13× the API cost and 13× faster context
exhaustion. This was a real fairness complaint about early GPT models.

### The failure happens in TWO distinct layers

1. **Pre-tokenization (the regex)** — actually *mostly fine*, because `\p{L}` is any
   Unicode letter. Russian/Arabic/Korean chunk cleanly on spaces, just like English. But:
   - **Scriptio continua** (Chinese/Japanese have no spaces) → the whole sentence becomes
     **one giant chunk**. The regex offers BPE no internal structure.
   - **Combining marks** (Hindi/Devanagari) → vowel signs and the nukta `़` are `\p{M}`
     (marks), *not* `\p{L}`, so rule 2 stops at them and **shatters** single visual
     characters into multiple chunks.
2. **BPE merging** — the *bigger* culprit. Trained on English, it never learned merges for
   other scripts, so CJK characters fall back to **raw UTF-8 bytes** (~3 bytes/char, few
   merged). This is fixable: train BPE on multilingual data.

> **Q:** So it all depends on the training data? Real LLMs train multilingually and BPE
> learns CJK tokens?
>
> **A:** Exactly. The regex sets *what's possible*; the **corpus decides what merges are
> actually learned**. The fix is corpus + budget:
> - **Multilingual corpus** in tokenizer training → CJK character/phrase merges form.
> - **Bigger vocab** to afford coverage without crowding out English:
>   GPT-2 ~50K, [Llama-3](https://arxiv.org/abs/2407.21783) ~128K, GPT-4o ~200K,
>   [Qwen](https://arxiv.org/abs/2412.15115) ~152K,
>   [DeepSeek](https://arxiv.org/abs/2401.02954) ~100–128K (vs nanochat's 32K). Bigger
>   vocab = better multilingual efficiency, but larger embedding/output matrices and more
>   params — the tradeoff knob.
> - **Mixture ratios** matter: train on 90/10 English/Chinese and Chinese stays
>   under-served. Token fairness is literally tuned by sampling ratios.

### The asymmetry: who finds word boundaries?

A key insight discovered while reasoning through this:

| | English | Chinese |
|---|---|---|
| word boundaries | marked (spaces) | unmarked (no spaces) |
| pre-tok role | walls off words | passes through (one big chunk) |
| BPE's main job | strip/learn affixes *within* words | *discover* word/compound boundaries |
| max token span | ~1 word (walled) | unbounded — can span a phrase |

> Because Chinese chunks have **no internal walls**, BPE is free to merge across what we'd
> call word boundaries. In a Chinese-trained tokenizer, frequent phrases like
> `中华人民共和国` *can become a single token*. English can't do this — pre-tokenization
> already walled words apart. So: **English** = regex segments, BPE refines;
> **Chinese** = regex passes through, BPE segments *and* refines.

### Do Qwen/DeepSeek add CJK rules to the regex?

Mostly **no**. Hard-coding a Chinese word-segmenter
([jieba](https://github.com/fxsjy/jieba)-style) into pre-tokenization would (a) be
lossy/ambiguous (`研究生命` = `研究/生命`? `研究生/命`?) and (b) propagate its errors into
every token. Better to let **BPE learn boundaries statistically from data** —
which is exactly Chinese's strength (no internal walls). So the regex stays mostly
script-agnostic; the work is done by corpus mixture + vocab size. The *common* regex
refinements are **digit-splitting** (→ `\p{N}{1}` for arithmetic) and **code/whitespace**
handling — **not** CJK segmentation.

### Morphology / linguistic typology

Chinese is **isolating** (analytic): words don't inflect. No plurals (狗 = dog/dogs), no
conjugation (去 = go/goes/went), no case. So you avoid English's explosion of
`dog/dogs/dog's`, `run/runs/running/ran` — one stable written form, no inflectional
variants to spend vocab on. Chinese instead trades morphology for **compounding**
(电+脑 = computer, 火+车 = train). So BPE's job shifts: English learns `root+affix` merges;
Chinese learns `char+char→compound` merges. The morphological simplicity is a genuine
tailwind that partly offsets the no-spaces / byte-overhead headwinds.

## Research frontier — verified references

### SuperBPE: cross-word ("superword") tokenization

**"SuperBPE: Space Travel for Language Models"** ([COLM 2025, arXiv 2503.13423](https://arxiv.org/abs/2503.13423)).
Your intuition — "SuperBPE ≈ remove pre-tokenization" — is right, with a refinement: it's
a **two-stage pretokenization curriculum**, not a full removal.
- **Stage 1:** normal BPE *with* pre-tokenization → learn subwords (no merges across
  whitespace).
- **Stage 2:** resume from that vocab but **lift the pre-tokenization restriction**, so
  token pairs *bridging whitespace* can merge → **superwords** (tokens spanning >1 word,
  e.g. `␣of␣the`).

Why staged (and not removed from the start): learning subwords first avoids wasting vocab
on redundant punctuation/spacing variants. **Results:** +4.0% absolute average over BPE
across 30 tasks (+8.2% on MMLU) **and 27% less inference compute** (fewer tokens).

### Parity-aware BPE: fairness-aware merge selection

**"Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization"**
([ACL 2026, arXiv 2508.04796](https://arxiv.org/abs/2508.04796); Foroutan et al.,
swiss-ai, [code](https://github.com/swiss-ai/parity-aware-bpe)). My earlier guess at the
mechanism was correct, and here are the verified specifics:
- Standard BPE merges the **globally** most-frequent pair — dominated by the majority
  language, leaving low-resource languages with longer tokenizations (compute + cost
  inequity).
- Parity-aware BPE instead, **at every merge step, maximizes the compression gain of the
  currently *worst-compressed* language** — trading a little global compression for
  cross-lingual fairness.
- **Results:** <1% drop in overall compression, but Gini coefficient (a fairness measure)
  drops **0.064 → 0.011**; downstream model performance unaffected, token counts and
  perplexities more equitable across languages.

### Tokenizer-free: byte, character, and pixel models

The most radical direction — **escape the tokenizer entirely**. Your intuition ("encode by
pixels and the heavy tokenization reasoning goes away") is partly right: the **fairness and
brittleness problems do largely vanish** (bytes/pixels are language-neutral, no vocab, no
OOV, no glitch tokens, robust to typos). **But the cost doesn't vanish — it moves:**
sequence length explodes (a byte/patch carries far less info than a BPE token), so
attention's quadratic cost blows up, and the model must relearn character/word structure
from scratch. Hence every approach adds **downsampling or hierarchy**.

Verified models:

| Model | Type | Key idea |
|---|---|---|
| **ByT5** (Google, TACL 2021) | byte-level | UTF-8 bytes, no tokenizer; robust to noise/spelling; scaled >10B params |
| **CANINE** (Google, 2021) | character-level | first tokenization-free pretrained encoder; char input + downsampling + deep transformer |
| **Charformer** (Google, 2021) | learned char→subword | **GBST** (gradient-based subword tokenization): learns a *soft, differentiable* tokenization end-to-end; on par with ByT5 but 2× more memory-efficient, 10–93% faster |
| **MEGABYTE** (Meta, 2023) | byte-level | multiscale/hierarchical modeling to handle long byte sequences |
| **SpaceByte** (2024, arXiv 2404.14408) | byte-level | inserts "patch" boundaries (often at spaces) to close the gap with subword models |
| **MrT5** (2024, arXiv 2410.20771) | byte-level | dynamic token merging for efficient byte-level LMs |
| **PIXEL** (2022, arXiv 2207.06991) | **pixel/visual** | renders text as images → fixed-size patches → ViT encoder; reconstructs masked patches (no vocab at all) |

**PIXEL's notable result:** it *outperforms* BERT on scripts **not seen in pretraining**
(the fairness win you predicted) but is slightly weaker on Latin scripts, and is more
robust to noisy input. As of early 2026 these are promising but not yet dominant for
frontier LLMs — the efficiency math still favors BPE.

---

## Sources

- SuperBPE — [arXiv 2503.13423](https://arxiv.org/abs/2503.13423) · [project page](https://superbpe.github.io/) (COLM 2025)
- Parity-Aware BPE — [arXiv 2508.04796](https://arxiv.org/abs/2508.04796) · [code](https://github.com/swiss-ai/parity-aware-bpe) (ACL 2026)
- ByT5 — [TACL / MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00461/110049/) · CANINE & Charformer — [Charformer arXiv 2106.12672](https://arxiv.org/pdf/2106.12672)
- PIXEL — [arXiv 2207.06991](https://arxiv.org/pdf/2207.06991) · Text rendering — [arXiv 2311.00522](https://arxiv.org/abs/2311.00522)
- MEGABYTE / SpaceByte / MrT5 — [SpaceByte arXiv 2404.14408](https://arxiv.org/pdf/2404.14408) · [MrT5 arXiv 2410.20771](https://arxiv.org/pdf/2410.20771)
