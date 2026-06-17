# Authoring Principles (meta)

> How these tutorial chapters are written. Read this before adding or editing a chapter
> so the series stays consistent. It is the "style guide + method" distilled from how
> Chapters 00–05 were produced.

## Origin & philosophy

These chapters are a byproduct of *actually reading and running* nanochat on a cluster,
captured as we went. The guiding belief:

> **A good tutorial is a real investigation, cleaned up — not a textbook written from
> memory.** Every claim should trace back to code we read or output we observed.

So the workflow is: explore code → run it → discuss/question → *then* write the chapter
from what actually happened.

## The 10 principles

1. **Pipeline order, not file order.** Chapters follow the order things *run*
   (env → data → pre-tokenization → BPE → …), because that's how understanding builds.
   Within a chapter: the *what/why* (the script) before the *how* (the module).

2. **Ground every claim in code or measured output.** Numbers (shard sizes, token
   counts, timings, vocab math) come from real runs on this machine, never estimates.
   When a number is observed, say so ("measured", "from our run"). Mechanisms come from
   reading the source, with file+line references.

3. **Link generously, three ways:**
   - **Code links** — relative paths with line anchors, e.g.
     `[tokenizer.py:30](../nanochat/tokenizer.py#L30)`. These work both in the IDE and
     when rendered on GitHub, and survive sharing the repo.
   - **Cross-chapter links** — connect prereqs and follow-ups so chapters form a graph,
     not a list. Each chapter header notes its prereq and next.
   - **External links** — papers (arXiv), tools (uv, tiktoken), datasets (HF), and
     canonical references (UTF-8, Chinchilla). Prefer primary sources.

4. **Preserve the discovery as Q&A callouts.** The insightful back-and-forth that
   produced an understanding is *kept*, not flattened into prose:
   ```markdown
   > **Q:** I thought BPE starts from characters?
   >
   > **A:** No — it starts from bytes. ...
   ```
   Use this specifically for **misconceptions corrected** and **"why is it this way"**
   moments — the things a reader is likely to wonder too.

5. **Lead with the answer, then explain.** Each section states the conclusion up front
   (bold one-liner or a "headline" sentence), then unpacks it. Readers skim first.

6. **Tables for anything enumerable.** The 7 regex rules, the byte ranges, the
   tokens-per-char comparison, the two data loaders — all tables. They compress and
   invite comparison.

7. **Concrete examples over abstraction.** Show the actual string → actual chunks →
   actual token ids. A worked `"I'm paying $19.99!!!"` scan beats a paragraph about how
   regex alternation works.

8. **Flag confidence honestly.** If something is uncertain or past the knowledge cutoff,
   say so and **verify it** (web search / run code) rather than asserting. Ch05's
   research claims were all web-verified; the chapter says so and cites sources. If a
   prior guess turns out right after checking, note that too — it teaches calibration.

9. **Separate core from frontier.** The essential pipeline is its own clean thread;
   tangents (multilingual, research papers, alternatives) live in a clearly-marked
   deep-dive chapter the reader can skip. Don't bloat the core path.

10. **Interview-aware framing.** Where a concept is a common interview question, call it
    out ("great interview talking point", "the answer this gives you is…"). The series'
    purpose is refreshing the LLM stack for interviews, so surface the high-leverage bits.

## Mechanical conventions

- **Filenames:** `NN_snake_case_title.md`, zero-padded so they sort (`00_`, `01_`, …).
- **Chapter title:** `# Chapter N — Title Case`.
- **Header blockquote:** a 2–4 line summary + prereq/next/background links.
- **Code fences:** annotate with the language; trim to the relevant lines; keep
  Karpathy's own comments when they explain *why*.
- **`␣`** denotes a literal space in examples where it would otherwise be invisible.
- **Sources footer** (frontier/research chapters): a `## Sources` list of the primary
  links used, so claims are auditable.
- **Status honesty:** mark "to be continued" where a thread genuinely continues later,
  rather than padding.

## How to add a new chapter

1. Do the work first (read the code, run it, capture real output/numbers).
2. Pick the number that preserves pipeline order (renumber later chapters if inserting).
3. Draft using the principles above; lead with conclusions, link as you go.
4. If it touches fast-moving research, **web-search to verify** and add a Sources footer.
5. Add a row to [`README.md`](README.md) and wire prereq/next cross-links.
6. Do a final pass: check every link resolves, every number is real, every Q&A is
   genuinely illuminating (cut filler Q&A).

## What to leave OUT

- Anything the repo already documents well (don't restate the README verbatim).
- Transient session detail (job IDs, one-off paths) unless it teaches a reusable lesson
  (e.g. the `.bashrc` interactive-guard trap *is* reusable, so it stayed).
- Speculation dressed as fact. If unsure, verify or omit.
