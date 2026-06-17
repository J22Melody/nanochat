# Chapter 1 — Environment Setup

> Getting nanochat runnable on a shared HPC/SLURM cluster: the `uv` virtual
> environment, and — crucially — putting data on the right filesystem so you don't
> blow your home quota.

## The tooling: `uv`, not conda

nanochat uses [`uv`](https://github.com/astral-sh/uv) (a fast Rust-based Python
package manager — [docs](https://docs.astral.sh/uv/)), not conda. The setup is three
steps (mirroring [`runs/speedrun.sh`](../runs/speedrun.sh#L19-L28)):

```bash
# 1) install uv (adds ~/.local/bin/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) create the venv (uv reads .python-version -> Python 3.10, fetches it itself)
cd ~/nanochat && uv venv

# 3) install deps. Torch has TWO mutually-exclusive extras:
uv sync --extra gpu      # CUDA 12.8 wheels (torch==2.9.1+cu128)
# or: uv sync --extra cpu  (CPU-only wheels)
```

Then **activate it in every new shell**:

```bash
cd ~/nanochat && source .venv/bin/activate
```

### Key decision: install the `gpu` extra even on a CPU node

CUDA torch wheels *install* fine on a CPU node (downloading/installing needs no GPU) —
they just can't *run* CUDA there. Since the `.venv` lives in the shared repo dir and is
reused later on the GPU node, install `--extra gpu` once and it's ready. On the CPU
node, `torch.cuda.is_available()` returns `False` (expected); it becomes `True` on the
GPU node — same venv, no reinstall.

> **Q:** We use conda for other projects. Will adding `uv` to PATH conflict?
>
> **A:** No. The two are fully isolated:
> - This cluster's conda is loaded *on demand* via `module load miniforge3` (Lmod),
>   not from `.bashrc`. It only enters PATH when you load the module, and then it
>   prepends itself ahead of everything.
> - `uv` lives in `~/.local/bin` (just the `uv`/`uvx` binaries — no Python, no shims).
> - nanochat uses its own isolated `.venv`, activated explicitly.
>
> The three never mix: conda by `module load`, nanochat in its venv, `uv` as a PATH
> tool.

## The filesystem trap: home quota vs. scratch

On a cluster, **where intermediate data lands matters**. nanochat resolves its base
directory like this ([`nanochat/common.py:70`](../nanochat/common.py#L70)):

```python
def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return os.environ["NANOCHAT_BASE_DIR"]
    return os.path.join(os.path.expanduser("~"), ".cache", "nanochat")  # default!
```

So **by default everything goes to `~/.cache/nanochat`** — on your quota'd home
filesystem. Check your quota before downloading anything:

```
/home/zifjia    275 GB used / 373 GB limit   (73.6% full -> only ~98 GB free)
/scratch/zifjia   0 B used / 20 TB limit      (essentially empty)
```

### What gets stored, and how much

| Artifact | Approx size |
|---|---|
| Pretraining data shards (see [Ch02](02_data_preparation.md)) | ~100 MB each; speedrun pulls ~170 → **~20 GB** |
| Tokenizer (see [Ch04](04_bpe_and_byte_vocabulary.md)) | a few MB |
| Base model checkpoints | ~1–4 GB each (depends on depth) |
| SFT/eval outputs, report | a few GB |

**Budget:** a learning walkthrough is ~5–10 GB; a full speedrun-scale run ~30–40 GB.
That dominant cost (data shards) is *regenerable*, which makes it a textbook **scratch**
use case.

### Point the base dir at scratch — robustly

```bash
export NANOCHAT_BASE_DIR=/scratch/zifjia/nanochat
```

> **Q:** I added the export to `~/.bashrc` but the download still went to home — why?
>
> **A:** A classic gotcha. `~/.bashrc` has an early guard:
> ```bash
> # If not running interactively, don't do anything
> case $- in *i*) ;; *) return;; esac
> ```
> Anything **after** this `return` does **not** run in non-interactive shells —
> including `bash -lc '...'` and **SLURM batch jobs**. So the export never reached the
> job. **Fix: put the export ABOVE the guard** so it reaches every shell:
> ```bash
> export NANOCHAT_BASE_DIR="/scratch/zifjia/nanochat"   # before the *i* guard
> case $- in *i*) ;; *) return;; esac
> ```
> Verify: `bash -lc 'echo $NANOCHAT_BASE_DIR'` should print the scratch path.

> **⚠️ Scratch caveat:** cluster scratch is typically **auto-purged after ~30 days of
> inactivity and not backed up**. Fine here — everything nanochat writes is
> regenerable. Never put anything irreplaceable there.

## "Login shell" ≠ "login node" — don't confuse them

- **Login *shell*** = a shell *startup mode* (`bash -l`). It controls *which startup
  files get sourced* (so `~/.bashrc`/`~/.profile` run and set env vars). It has
  **nothing to do with which machine you're on.**
- **Login *node*** = the physical box you SSH into (e.g. `u24-login-1`) to submit jobs.
  You should **not** run heavy compute here.

`bash -lc '...'` just forces login-shell mode (so `NANOCHAT_BASE_DIR` gets set). The
command still runs on whatever node it's invoked from. Always check with `hostname`
and `nproc` that you're on a compute node, not the login node.

## Checklist

- [ ] `uv` installed, on PATH
- [ ] `.venv` created (Python 3.10), `uv sync --extra gpu` done
- [ ] `NANOCHAT_BASE_DIR` exported **above** the `.bashrc` interactive guard → scratch
- [ ] verified `bash -lc 'echo $NANOCHAT_BASE_DIR'` prints the scratch path
- [ ] running on a **compute node**, not the login node
