# Chapter 4 — BPE and the Byte Vocabulary

> How the vocabulary is built: it starts from **256 raw bytes** (not characters),
> grows via learned **merges**, and reserves a handful of **special tokens**. The
> arithmetic lands exactly on 32768. *(The detailed merge-loop walkthrough is
> [Ch05](05_bpe_merge_loop.md).)*
>
> Prereq: [Ch03](03_pretokenization.md) (the walls BPE merges within). Foundational
> reading: [Sennrich et al. 2016](https://arxiv.org/abs/1508.07909) (BPE for NMT, the
> origin), [GPT-2 / Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
> (byte-level BPE), and Karpathy's [minbpe](https://github.com/karpathy/minbpe).

## The numbers from our run

Training [`python -m scripts.tok_train`](../scripts/tok_train.py) on ~2B chars produced
(in ~2.5 min on 2 cores):

```
Finished training: 32503 merges completed
vocab_size: 32,768
```

And the vocab arithmetic — a beautiful sanity check:

```
   256   base byte tokens   (every possible byte value)
+ 32503   learned BPE merges (256 → 32759)
+     9   special tokens     (appended after training)
─────────
  32768   = 2^15  ✓ exactly vocab_size
```

That's *why* it's 32503 merges, not a round number: the trainer computes
`merges = vocab_size − 256 − num_special` so the total hits 32768 exactly.

## What goes into the trainer

From [`nanochat/tokenizer.py:171`](../nanochat/tokenizer.py#L171),
`RustBPETokenizer.train_from_iterator`:

```python
tokenizer = rustbpe.Tokenizer()                              # fresh, empty
vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)     # 32768 - 9 = 32759
assert vocab_size_no_special >= 256                          # must fit the 256 bytes
tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
```

Three inputs: the **text stream**, the **target vocab (32759)**, and the **split pattern**
(Chapter 3). Note the **9 special tokens are NOT trained** — they're reserved out of the
budget now and bolted on afterward at fixed ids past the learned ones, matched by exact
string (never split into bytes).

nanochat uses three tokenizer implementations for different jobs: **`rustbpe`** (a fast
Rust trainer bundled with [nanochat](https://github.com/karpathy/nanochat), installed as
a pip dependency), **[tiktoken](https://github.com/openai/tiktoken)** (fast inference
encoder, built from the learned merge ranks), and a
**[HuggingFace tokenizers](https://github.com/huggingface/tokenizers)** wrapper
([`tokenizer.py:39`](../nanochat/tokenizer.py#L39), reference/alternate). Same algorithm,
different speed/role.

## The initial vocab = the 256 byte values (NOT characters)

> **Q:** I thought BPE starts from characters or ASCII letters?
>
> **A:** No — it starts from **bytes**. A byte is just a number 0–255, and there are only
> 256 possible values, *ever*. The base vocab is one token per byte value:

| Range | Count | What they are |
|---|---|---|
| 0–31, 127 | 32 | **control bytes** — NUL, TAB(9), newline/LF(10), CR(13), ESC, DEL |
| 32–126 | 95 | **printable ASCII** — space, digits, `A-Z`, `a-z`, punctuation |
| 128–255 | 128 | **high bytes** — *not characters alone*; only meaningful as pieces of multi-byte UTF-8 |
| **256** | | the complete byte space |

The key fact: **a non-ASCII character is multiple bytes**
([UTF-8](https://en.wikipedia.org/wiki/UTF-8)):

```
'A'  → 1 byte:  [65]
'é'  → 2 bytes: [195, 169]
'你' → 3 bytes: [228, 189, 160]
'🌍' → 4 bytes: [240, 159, 140, 141]
```

So one character ≠ one base token. `你` is **3 byte-tokens**. (That's why `你好` encoded to
6 byte-tokens earlier — none frequent enough in English to merge.)

**Why bytes, not characters?** There are ~150,000 Unicode characters — too many for a base
vocab, and you'd *still* hit unknowns. But there are only 256 byte values. A 256-byte base
vocab can represent **any text in any language with zero unknowns** — there is **no `<unk>`
token, ever**. Worst case, a rare character stays as raw bytes; common ones get merged.

**Why English felt "character-like":** for ASCII, byte == character one-to-one (bytes
32–126 *are* the printable ASCII chars). So English tokenization *looks* character-level —
but that's a special case that breaks the instant you hit é, 你, or 🌍.

> Mental model: BPE does **not** start from characters — it *arrives* at them.
> ```
> 256 byte tokens  --merge frequent adjacent pairs-->  " t", "the", "world", ...
> (the alphabet; ASCII looks char-like, 128-255 are UTF-8 fragments)
> ```

## The special tokens (9)

From [`nanochat/tokenizer.py:13`](../nanochat/tokenizer.py#L13):

```python
SPECIAL_TOKENS = [
    "<|bos|>",             # Beginning of Sequence — delimits documents (pretraining)
    "<|user_start|>", "<|user_end|>",            # chat: user turn (SFT)
    "<|assistant_start|>", "<|assistant_end|>",  # chat: assistant turn (SFT)
    "<|python_start|>", "<|python_end|>",        # tool use: assistant calls Python REPL
    "<|output_start|>", "<|output_end|>",        # tool use: REPL output back to assistant
]
```

- **`<|bos|>`** is the only one used in **pretraining** — prepended to every document.
- The other 8 are used only in **SFT/chat** to render conversations + tool calls. Reserved
  now (stable ids) but the model learns their meaning during finetuning.

> **Q:** No end-of-sentence or end-of-document token?
>
> **A:** Correct, and intentional. The only doc-level delimiter is `<|bos|>` at the
> *start*. Documents are packed back-to-back: `<|bos|> docA <|bos|> docB ...` — so the
> *next* `<|bos|>` *is* the end signal; a separate end token would be redundant. (Same
> logic as GPT-2's `<|endoftext|>`, which despite the name is almost always *prepended*.)
> There's **no sentence token at all** — sentence boundaries are learned implicitly from
> `.`/`\n` patterns. The chat exception: SFT *does* add `<|assistant_end|>` etc., because
> at inference the model must signal "I'm done" so generation can stop.

## Tokenization in action (encoding examples)

| Input | Tokens | Lesson |
|---|---|---|
| `hello world` | `['hell','o','␣world']` | common word = 1 token (with leading space); `hello` wasn't frequent enough |
| `␣the cat sat` | `['␣the','␣cat','␣sat']` | 1 token/common-word; `␣the` has a very low id (learned early = very frequent) |
| `I'm happy, you're not.` | `I `'m`␣happy`,`␣you`'re`␣not`.` | contractions + punctuation split off (pre-tok) |
| `The year was 2024.` | `…'␣','20','24','.'` | `2024` → `20`+`24` (the `\p{N}{1,2}` rule) |
| `unbelievable` | `['un','belie','vable']` | rare/long word → subword pieces |
| `你好 🌍` | 8 byte-tokens for 4 chars | byte fallback; no `<unk>`, just inefficient |

Takeaways: **lower token id ≈ higher frequency** (merged earlier); common word = 1 token;
rare word = many subwords; non-English = raw bytes (many tokens). Token count ≈ inverse
familiarity. The non-English inefficiency is the subject of [Ch07](07_tokenization_deep_dives.md#multilingual-the-tokenization-tax).

## The `token_bytes` cache → bits-per-byte (BPB)

[`tok_train.py:72-91`](../scripts/tok_train.py#L79) also caches, per token id, **how many
UTF-8 bytes** that token represents (specials = 0). This lets eval report
**[bits-per-byte](https://en.wikipedia.org/wiki/Cross_entropy)** instead of raw per-token
loss. BPB is **invariant to vocab size**, so you can fairly compare tokenizers/models with
different vocabularies — it's one of nanochat's primary pretraining metrics (alongside the
[DCLM](https://arxiv.org/abs/2406.11794) CORE score).

---

*Continued in [Ch05](05_bpe_merge_loop.md): the BPE **merge loop** itself — watching a
single chunk collapse from raw bytes → intermediate merges → final token ids — then
[Ch06](06_encoding_and_evaluation.md) on encoding and how `mergeable_ranks` drive fast
inference.*
