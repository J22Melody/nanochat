# Chapter 3 — Pre-tokenization

> Before BPE merges anything, text is first chopped into chunks by a regex. This
> "pre-tokenization" step draws the **walls** that BPE is forbidden to merge across. It
> is the single most underappreciated part of a tokenizer.
>
> Prereq: [Ch02](02_data_preparation.md) (the text stream). Next: [Ch04](04_bpe_and_byte_vocabulary.md)
> (what BPE does *within* each wall). Background: Karpathy's
> [minbpe](https://github.com/karpathy/minbpe) and his
> ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) video.

## The pattern

From [`nanochat/tokenizer.py:30`](../nanochat/tokenizer.py#L30) (`SPLIT_PATTERN`):

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

This is the [GPT-4 split pattern](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
(the `cl100k_base` pattern) with **one tweak** (see below). It is a single big
**alternation** (`A|B|C|…`) applied with `findall`: scan left→right, at each position try
the alternatives **in order**, take the **first** that matches, consume it, repeat. Every
character lands in exactly one chunk. (Requires the
[`regex`](https://pypi.org/project/regex/) module, not stdlib `re`, for `\p{L}`/`\p{N}`
Unicode classes and possessive quantifiers.)

## The 7 rules, in priority order

(`␣` = a literal space.)

| # | Rule | Matches | Example | Chunk grabbed |
|---|---|---|---|---|
| **1** | `'(?i:[sdmt]\|ll\|ve\|re)` | contraction suffix: `'s 't 'm 'd 'll 've 're` | `you're` | `'re` |
| **2** | `[^\r\n\p{L}\p{N}]?+\p{L}+` | a word: letters + **one optional leading non-letter/digit** | `hello␣world` | `␣world` |
| **3** | `\p{N}{1,2}` | **1–2 digits** | `2024` | `20`, then `24` |
| **4** | ` ?[^\s\p{L}\p{N}]++[\r\n]*` | symbol/punct run (optional leading space, optional trailing newlines) | `cost␣$5` | `␣$` |
| **5** | `\s*[\r\n]` | whitespace ending in a **newline** | `hi␣␣\n` | `␣␣\n` |
| **6** | `\s+(?!\S)` | **trailing** whitespace at end of text | `hi␣␣␣` | `␣␣␣` |
| **7** | `\s+` | any other whitespace run | `a␣␣␣␣b` | `␣␣␣␣` |

`\p{L}` = any Unicode **letter** (not just Latin!), `\p{N}` = any Unicode **number**,
`\s` = whitespace.

## It is NOT "split on whitespace" — it's class-based

A common misconception: this is whitespace splitting. It isn't. The organizing principle
is **character-class boundaries**: letters (rule 2), digits (rule 3), symbols (rule 4),
whitespace (rules 5–7), plus the contraction special-case (rule 1). A chunk almost never
mixes classes. That's why `happy,` → `␣happy` + `,` and `2024.` → `20` + `24` + `.`.

> **Q:** So it's basically whitespace-based + a few rules?
>
> **A:** Refined: it's **character-class-based + a few special cases**. Whitespace
> handling is only 3 of the 7 rules, not the principle. The two genuinely clever bits:
> 1. **Leading-space gluing** (rule 2's optional `?+`): the space attaches to the *front*
>    of the following word → `␣world`, `␣the`. A word at the start of a line tokenizes
>    differently from mid-sentence — the model gets word-boundary info for free.
>    (Whitespace-splitting would *throw the space away*.)
> 2. **Order = precedence**: rule 1 is first so `'re`/`'m` are claimed as contractions
>    *before* rule 4 could grab the apostrophe as a stray symbol.

## A worked scan

`"I'm paying $19.99!!!"`:

| pos | chars | rule | why |
|---|---|---|---|
| 0–1 | `I` | 2.word | letter, no leading symbol |
| 1–3 | `'m` | **1.contraction** | rule 1 tried *first* → wins over symbol |
| 3–10 | `␣paying` | 2.word | space glues to word |
| 10–12 | `␣$` | 4.symbols | space + `$` (rule 2 fails: `$` not a letter) |
| 12–14 | `19` | 3.digits | ≤2 digits |
| 14–15 | `.` | 4.symbols | decimal point **separated** from number |
| 15–17 | `99` | 3.digits | ≤2 digits |
| 17–20 | `!!!` | 4.symbols | possessive `++` grabs all three at once |

And the orphan-space artifact, `" The year was 2024."`:

```
'␣The' '␣year' '␣was' '␣' '20' '24' '.'
                       ↑ orphaned space (rule 7)
```

## Spaces glue to letters and symbols — but NOT digits

A leading space can join a **word** (rule 2) or a **symbol run** (rule 4), but **never a
digit** — rule 3 (`\p{N}{1,2}`) has no optional leading-char slot. So a space before a
number gets orphaned into its own rule-7 chunk:

```
"␣world"  → "␣world"        (rule 2: space + letters ✓)
"␣$5"     → "␣$" "5"        (rule 4: space + symbol ✓)
"␣2024"   → "␣" "20" "24"   (space orphaned)
```

No deep linguistic reason — just a design quirk inherited from GPT's pattern. Minor cost:
text with many " <number>" patterns wastes a token per orphaned space.

## The nanochat tweak: `\p{N}{1,2}`

GPT-4 uses `\p{N}{1,3}` (up to 3-digit number chunks). nanochat uses **`\p{N}{1,2}`** (up
to 2). Per the [code comment](../nanochat/tokenizer.py#L27), Karpathy found 2 is the sweet
spot for a 32K vocab — it doesn't "waste" token space on 3-digit numbers, which matters
more for small vocabs. The modern trend ([Llama-3](https://arxiv.org/abs/2407.21783),
DeepSeek, GPT-4) goes further toward `\p{N}{1}` (single digits) for better arithmetic.
More on this in [Ch05](05_tokenization_deep_dives.md).

## Why pre-tokenize at all?

You *could* run BPE on raw text with no regex — and people did early on, producing **bad
merges**: single tokens for `". The"` or `dog,` (spanning word/punct boundaries, wasting
vocab), plus thousands of number tokens. Pre-tokenization **draws hard walls** so BPE can
only merge *within* linguistically sensible regions.

> **The one rule that matters:** BPE merges happen only *inside* a chunk. Pre-tokenization
> draws the walls; BPE builds *within* each walled-off region. That's why `2024.` can
> never be one token, and why `␣the` is a great token candidate (space lives inside the
> chunk).

```
text.split()   → ['The','year','was','2024.']                  # space lost, "2024." mixed
SPLIT_PATTERN  → ['The','␣year','␣was','␣','20','24','.']        # space kept, classes separated
```

The same regex is used **at train time** (constrains which pairs are counted) and **at
inference** ([tiktoken](https://github.com/openai/tiktoken) uses this exact `pat_str` to
split before encoding — see [`tokenizer.py:186`](../nanochat/tokenizer.py#L186)).
