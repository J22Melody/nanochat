# Chapter 10 — The Embedding Stage

> The forward pass begins here: turning a batch of token **ids** (integers) into **vectors**
> the transformer can process. Three steps own this stage — the `wte` lookup, an immediate
> RMS-norm, and a "smear" that blends in the previous token — producing `x0`, the initial
> representation. Two more things are *prepared* here but *consumed* later (a `value_embeds`
> lookup, and the rotary position table), and we flag why **position is not added here at all**.
>
> Prereq: [Ch08](08_model_config_and_scaling.md) (`wte`, `lm_head`, the parameter groups),
> [Ch09](09_weight_initialization.md) (how these weights start). Next: the **attention** chapter
> (the first transformer block, where rotary position and `value_embeds` are actually used). Code:
> [`gpt.py:forward`](../nanochat/gpt.py#L416) lines 427-452.

## The journey: id → vector

```python
x  = self.transformer.wte(idx)   # 1) look up each token's embedding vector
x  = norm(x)                     # 2) RMS-normalize
x  = smear(x)                    # 3) blend in the previous token (cheap bigram info)
x0 = x                           # 4) save as the "initial embedding" (re-injected later)
```

For our depth-4 run, `idx` is `(B, T)` token ids and `wte` turns each into a **256-dim vector** →
`x` is `(B, T, 256)`. (`smear` is shorthand for the gated block at [gpt.py:432-437](../nanochat/gpt.py#L432);
unpacked below.)

## 1. `wte` — the token embedding lookup

[`wte`](../nanochat/gpt.py#L428) is a lookup table: row `i` is the learned vector for token id `i`,
and `wte(idx)` gathers one row per token. That's the entire mapping from discrete ids to continuous
vectors — the model's input layer.

> **Q:** Why the name `wte`?
>
> **A:** **"Word Token Embedding"**, a GPT-2 convention (`w` = word, `t` = token, `e` = embedding).
> The `w` is a historical fossil from word-embedding days ([word2vec](https://arxiv.org/abs/1301.3781));
> the units are really BPE *tokens*, not words. GPT-2 also had a sibling `wpe` ("word **position**
> embedding") added to `wte` for position — **nanochat has no `wpe`**, and that absence is the tell
> that it uses rotary position instead (below).

Init (from [Ch09](09_weight_initialization.md)): `Normal(std=0.8)`, then immediately normalized.

## 2. `norm` — RMSNorm (a shared primitive)

Right after lookup, `x = norm(x)` ([`norm`](../nanochat/gpt.py#L42) = `F.rms_norm`): each token
vector is rescaled to unit root-mean-square length. **RMSNorm** is a cheaper LayerNorm variant —
it normalizes *scale* only (no mean-centering, no bias), the modern default. Note this same `norm`
is reused **pre-norm inside every block** ([gpt.py:149](../nanochat/gpt.py#L149)); it's a small
reusable utility, not an embedding-specific step.

A useful consequence: because `wte` is normalized here, the forward activations are **largely
invariant to `wte`'s init scale** — which is why Ch09 could treat `std=0.8` as low-stakes.

## 3. Smear — cheap bigram context

[Lines 432-437](../nanochat/gpt.py#L432) (training path):

```python
gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))   # per-position gate
x = torch.cat([x[:, :1],  x[:, 1:] + gate * x[:, :-1]],  dim=1)             # add previous token
```

In words: **every position `t` (except the first) becomes `emb[t] + gate · emb[t−1]`** — its own
embedding plus a learned fraction of the token right before it. Position 0 has no predecessor, so
it's left untouched (the `x[:, :1]` slice). The gate is computed from just the first 24 embedding
dims (a tiny `Linear(24→1)`), squashed to `[0,1]` by sigmoid, and scaled by `smear_lambda` — which
**inits to 0**, so smear starts **off** and the model learns whether to use it.

```
"river bank":   x["bank"]  →  emb("bank") + 0.3 · emb("river")     (if gate ≈ 0.3)
```

**Why:** mixing across positions is normally attention's (expensive) job; smear is a free, fixed,
offset−1 mix that hands every position its single most-predictive neighbor — bigram-like context —
before attention even runs. (The `else` branch at [gpt.py:438-449](../nanochat/gpt.py#L438) does the
identical thing for KV-cache decoding, reading the previous embedding from the cache.)

> **How common is this? (web-verified)** Smear is **not** a standard transformer component —
> mainstream attention LLMs (Llama, Qwen, Mistral, Gemma) leave *all* cross-token mixing to
> attention. It comes from the **modded-nanogpt** speedrun lineage (nanochat's variant uses the
> first **24** dims and an *additive* `x + gate·prev` form; modded-nanogpt's described variant uses
> 12 dims and a convex `(1−α)x + αx_prev` blend — same idea, tuned differently). Its closest
> conceptual relative is **RWKV's "token shift"**, where a learned current/previous-token
> interpolation is a *core* mechanism (RWKV-6 "Finch" even makes it data-dependent, like nanochat's
> content-dependent gate). Sources: [RWKV paper](https://arxiv.org/pdf/2305.13048),
> [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
> [snimu: x0/value/UNet lambdas](https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html),
> [NanoGPT speedrun WR writeup](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months).
> Treat it as a "speedrun optimization," not a building block you'd expect in a standard model.

## 4. `x0` — saved for later

`x0 = x` ([gpt.py:452](../nanochat/gpt.py#L452)) stores this initial normalized+smeared embedding.
Every layer can later blend it back via `x0_lambdas` (Ch09's decaying schedule) — a learned
skip-connection from the input to every depth. **Mechanism deferred to the Block chapter**; just
note its *origin* is here.

## Prepared here, consumed in the block

Two things are computed at this stage but **belong to attention**, so we only stage them now:

| Thing | Prepared here | Used in the block |
|---|---|---|
| **`value_embeds` (ve)** | a second embedding lookup of `idx`, on alternating layers ([`has_ve`](../nanochat/gpt.py#L53)) | added to the **value** vector inside attention |
| **RoPE `cos/sin`** | precomputed table ([gpt.py:425](../nanochat/gpt.py#L425)) | rotates **Q/K** inside attention |

### Why there's no positional embedding here

> **Q:** Where does position get added? I don't see it in the embedding stage.
>
> **A:** It isn't added at all. nanochat uses **RoPE** ([Su et al. 2021](https://arxiv.org/abs/2104.09864)):
> position is injected by **rotating the query and key vectors inside attention**, not by adding a
> vector at the input. So the embedding stage only *precomputes* the rotation table; the actual
> position mechanism lives in the **attention chapter**, where the rotation acts on Q/K and gives
> attention a *relative*-position view. This is the modern norm (Llama, Qwen, Mistral all do this) —
> and why there's no `wpe`.

## What the embedding stage cleanly owns

```
OWNS (input stage):     wte lookup  →  norm  →  smear   ⇒  x0
PREPARES (used later):  value_embeds lookup,  rotary cos/sin
DOES NOT DO:            add any positional vector  (RoPE handles it, in attention)
```

> **One line:** a token id becomes a vector via the `wte` lookup, is RMS-normalized and "smeared"
> with its predecessor for cheap bigram context, and saved as `x0`; position is **not** added here —
> it's injected later by rotating Q/K (RoPE) inside attention.

---

*Next: the **attention** block — standard Q/K/V first, then the modern layers nanochat adds on top
(RoPE, grouped-query attention, sliding-window, and the `value_embeds` we just staged).*
