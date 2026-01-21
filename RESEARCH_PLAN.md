# Early-Exit Draft Models for Speculative Decoding — Research Plan

## Goal
Train and evaluate *early-exit* “draft” models for speculative decoding by moving (and optionally tuning) the LM head to an intermediate transformer layer. Measure the resulting **speed–quality tradeoff** and **speculative decoding acceptance** for chat-style generation.

## Models
- **Target model (teacher / verifier):** `meta-llama/Llama-3.2-1B-Instruct` (16 transformer blocks).
- **Draft model:** same checkpoint, but truncated at an exit layer `ℓ` (0-based index), using a tuned head to map `h_ℓ → logits`.
  - Option A (default): fine-tune **LM head only**.
  - Option B (capacity ablation): fine-tune **LM head + last kept transformer block** at `ℓ` (often needed for earlier exits).

## Training objective
Distill the draft logits toward the target logits:
- **Token-mean KL** (masking padding and the first token): `KL(p_teacher || p_draft)` with temperature `T`.
- Optional **combined** objective: `CE(labels) + KL`.

## Data (train + eval)
Because the target use-case is **chat / speculative decoding**, prioritize chat-style text distributions.

- **Primary train:** ShareGPT-style conversations (e.g. `liyucheng/ShareGPT90K`), formatted as multi-turn text.
- **In-domain eval:** held-out ShareGPT split.
- **Out-of-domain eval (generalization):** FineWeb held-out slice (optional).

Other chat-ish datasets worth adding later (as separate eval sets, or as a mixture for training):
- UltraChat / OpenAssistant / LMSYS-chat style corpora (better prompt diversity than ShareGPT alone).

Rationale:
- Training on FineWeb but evaluating on chat is useful to measure generalization, but if the end goal is chat speculative decoding, *training on chat-like text* usually yields the best acceptance/speed behavior on chat prompts.

## What we will sweep
### Exit layers (coarse → full)
Coarse sweep (16-layer model, 0-based):
- layers `{3, 7, 11, 15}` corresponding to “~4/8/12/16” depth.

Follow-up:
- full sweep across all layers `ℓ ∈ [0, 15]` once the pipeline is stable.

### KL temperature
After picking 1–2 promising exit layers:
- `T ∈ {1.0, 2.0, 4.0}`.

### Capacity
If early exits underfit:
- compare `--ft_head` vs `--ft_head --ft_last_transformer`.

## Metrics (paper-grade)
### Distribution / LM quality (per token)
On held-out data (token-masked, token-weighted aggregation):
- **KL to teacher** (nats/token, unscaled; training objective uses `T^2` scaling) at the same temperature used for training (and optionally also at `T=1`).
- **CE / perplexity** vs ground-truth next tokens using the *draft logits*.
- **Top-1 next-token accuracy** vs ground-truth labels (and optionally top-k).

### Speculative decoding behavior (chat prompts)
For a fixed sampling policy (temperature, top-p/top-k) and `γ` draft length:
- `E[accepted]`: mean accepted draft tokens per verification.
- `accept_rate = E[accepted] / γ`.
- **Overlap / TV proxy**: `E[ Σ_v min(p(v), q(v)) ]` and `TV = 1 - overlap`.
- **Target efficiency proxy:** `target_calls_per_token ≈ 1 / (E[accepted] + 1)` (lower is better).

### Decode performance (what you ultimately care about)
With KV cache enabled:
- **Vanilla target decoding tokens/sec** (target-only).
- **Speculative decoding tokens/sec** (draft + verify).
- **Speedup** = spec_tokens/sec ÷ vanilla_tokens/sec.
- Optional: peak VRAM and prefill-vs-decode breakdown (if needed).

## Evaluation protocol
To keep comparisons clean:
- Fixed prompt length (or deterministic “take half” rule) and fixed `max_new_tokens`.
- Fixed random seed(s) and fixed sampling policy (`T`, `top_p`, `top_k`).
- Report means + bootstrap CIs across prompts (optional for final paper plots).

## Plots to generate
- **Layer sweep:** exit layer vs {KL, CE/perplexity, top-1}.
- **Acceptance sweep:** exit layer vs {E[accepted], accept_rate, TV}.
- **Pareto:** tokens/sec speedup vs quality (e.g. perplexity or KL), points labeled by layer.
- Optional: acceptance histograms per layer.

## Artifacts
For each experiment:
- `results.csv` with one row per checkpoint/layer (metrics + settings).
- `plots/*.pdf` and `plots/*.png` suitable for paper inclusion.

## How to run (current scripts)
- Coarse layer sweep (trains a few exit layers, then evaluates + plots): `./run_sweep_chat_layers.sh`
- Evaluate an arbitrary set of checkpoints: `python evaluate_checkpoints.py --model_path ... --checkpoint_glob ".../layer*/model_*.pt" --out_dir ...`

## Immediate next run (coarse layer sweep)
Run 4 training jobs at exit layers `{3,7,11,15}` with a short, fixed budget (enough to see monotonic improvements in KL/CE), then evaluate:
- quality metrics on ShareGPT held-out
- speculative acceptance + decode tokens/sec on ShareGPT prompts
