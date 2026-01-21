import argparse
import glob
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fineweb_dataset import get_fineweb_dataloaders
from share_gpt_dataset import get_sharegpt_dataloaders
from ultrachat_dataset import get_ultrachat_dataloaders
from sampling_utils import sampling_probs_from_logits, safe_normalize
from truncated_llama import TruncatedLlama
from transformers import AutoTokenizer, AutoModelForCausalLM


def _flash_attn_2_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        return bool(is_flash_attn_2_available())
    except Exception:
        try:
            import flash_attn  # noqa: F401

            return True
        except Exception:
            return False


def resolve_attn_implementation(attn_flag: str, device_type: str) -> str | None:
    if attn_flag != "auto":
        return attn_flag
    if device_type == "cuda" and _flash_attn_2_available():
        return "flash_attention_2"
    if device_type == "cuda":
        return "sdpa"
    return None


def resolve_torch_dtype(dtype_flag: str, device_type: str) -> torch.dtype:
    if dtype_flag == "bf16":
        return torch.bfloat16
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag != "auto":
        raise ValueError(f"Unknown torch_dtype: {dtype_flag}")
    if device_type == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def shift_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(attention_mask)
    if attention_mask.size(1) > 1:
        shifted[:, :-1] = attention_mask[:, 1:]
    return shifted


@dataclass(frozen=True)
class QualityMetrics:
    kl: Optional[float]
    ce: Optional[float]
    ppl: Optional[float]
    top1_acc: Optional[float]

def compute_causal_lm_metrics(
    lm,
    dataloader,
    *,
    device: str,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
    max_batches: int,
) -> QualityMetrics:
    """
    Computes CE/perplexity/top-1 next-token accuracy for a standard causal LM.
    """
    lm.eval()

    total_ce = 0.0
    total_ce_tokens = 0
    top1_correct = 0
    top1_tokens = 0
    non_blocking = device_type == "cuda"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
            labels = batch.get("labels")
            if labels is None:
                continue
            labels = labels.to(device, non_blocking=non_blocking)

            autocast_ctx = (
                torch.autocast(device_type=device_type, dtype=autocast_dtype)
                if use_autocast
                else torch.no_grad()
            )
            with autocast_ctx:
                out = lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits

            ce_tokens = int((labels != -100).sum().item())
            if ce_tokens <= 0:
                continue
            ce_sum = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_ce += float(ce_sum.item())
            total_ce_tokens += ce_tokens

            pred = logits.argmax(dim=-1)
            mask = labels != -100
            top1_correct += int(((pred == labels) & mask).sum().item())
            top1_tokens += ce_tokens

    ce = (total_ce / total_ce_tokens) if total_ce_tokens > 0 else None
    ppl = float(math.exp(ce)) if ce is not None else None
    top1_acc = (top1_correct / top1_tokens) if top1_tokens > 0 else None
    return QualityMetrics(kl=None, ce=ce, ppl=ppl, top1_acc=top1_acc)


def compute_quality_metrics(
    model: TruncatedLlama,
    dataloader,
    *,
    device: str,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
    kl_temperature: float,
    max_batches: int,
    compute_kl: bool,
) -> QualityMetrics:
    model.eval()

    total_kl = 0.0
    total_kl_tokens = 0
    total_ce = 0.0
    total_ce_tokens = 0
    top1_correct = 0
    top1_tokens = 0

    non_blocking = device_type == "cuda"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device, non_blocking=non_blocking)

            autocast_ctx = (
                torch.autocast(device_type=device_type, dtype=autocast_dtype)
                if use_autocast
                else torch.no_grad()
            )
            with autocast_ctx:
                if compute_kl:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        loss_type="kl_divergence",
                        kl_temperature=kl_temperature,
                    )
                else:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        loss_type="cross_entropy",
                        kl_temperature=kl_temperature,
                    )
            logits = outputs["logits"]

            if compute_kl:
                sm = shift_mask(attention_mask)
                kl_tokens = int(sm.sum().item())
                if kl_tokens > 0:
                    kl_mean = float(outputs["loss"].float().item())
                    # Model scales KL by T^2 for training; report unscaled KL (nats/token).
                    denom = float(kl_temperature) ** 2
                    if denom > 0:
                        kl_mean = kl_mean / denom
                    total_kl += kl_mean * kl_tokens
                    total_kl_tokens += kl_tokens

            if labels is not None:
                ce_tokens = int((labels != -100).sum().item())
                if ce_tokens > 0:
                    ce_sum = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_ce += float(ce_sum.item())
                    total_ce_tokens += ce_tokens

                    pred = logits.argmax(dim=-1)
                    mask = labels != -100
                    top1_correct += int(((pred == labels) & mask).sum().item())
                    top1_tokens += ce_tokens

    kl = (total_kl / total_kl_tokens) if total_kl_tokens > 0 else None
    ce = (total_ce / total_ce_tokens) if total_ce_tokens > 0 else None
    ppl = float(math.exp(ce)) if ce is not None else None
    top1_acc = (top1_correct / top1_tokens) if top1_tokens > 0 else None
    return QualityMetrics(kl=kl, ce=ce, ppl=ppl, top1_acc=top1_acc)


@dataclass(frozen=True)
class SpecMetrics:
    mean_accepted: float
    accept_rate: float
    mean_overlap: float
    mean_tv: float
    vanilla_toks_per_s: float
    spec_toks_per_s: float
    speedup: float


def _init_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
):
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if use_autocast
        else torch.no_grad()
    )
    with autocast_ctx:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return out.past_key_values, out.logits[:, -1, :]


def _step_cache(
    model,
    token: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    *,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
):
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if use_autocast
        else torch.no_grad()
    )
    with autocast_ctx:
        out = model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return out.past_key_values, out.logits[:, -1, :]

def vanilla_decode_tokens_per_second(
    target_model,
    prompts: list[torch.Tensor],
    *,
    device: str,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device)
    is_cuda = device_type == "cuda"

    def sample_tok(logits: torch.Tensor) -> torch.Tensor:
        probs = sampling_probs_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        return torch.multinomial(probs, 1)

    total_tokens = 0
    total_time = 0.0
    target_model.eval()
    if hasattr(target_model, "config"):
        target_model.config.use_cache = True

    with torch.inference_mode():
        for prompt_ids in prompts:
            prompt_ids = prompt_ids.to(device_obj)
            attn = torch.ones_like(prompt_ids, device=device_obj)
            past, next_logits = _init_cache(
                target_model,
                prompt_ids,
                attn,
                device_type=device_type,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
            )
            generated = 0
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            while generated < max_new_tokens:
                tok = sample_tok(next_logits)
                attn = torch.cat([attn, torch.ones((1, 1), device=device_obj, dtype=attn.dtype)], dim=1)
                past, next_logits = _step_cache(
                    target_model,
                    tok,
                    attn,
                    past,
                    device_type=device_type,
                    use_autocast=use_autocast,
                    autocast_dtype=autocast_dtype,
                )
                generated += 1
            if is_cuda:
                torch.cuda.synchronize()
            total_time += time.time() - t0
            total_tokens += generated

    return (total_tokens / total_time) if total_time > 0 else 0.0


def speculative_decode_metrics(
    model: TruncatedLlama,
    prompts: list[torch.Tensor],
    *,
    device: str,
    device_type: str,
    use_autocast: bool,
    autocast_dtype: Optional[torch.dtype],
    max_new_tokens: int,
    speculate_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    vanilla_toks_per_s_override: Optional[float] = None,
    progress: bool = False,
    progress_every: int = 8,
) -> SpecMetrics:
    """
    Runs speculative decoding (draft = truncated_model, target = reference_model) with KV cache,
    measuring acceptance and decode throughput.
    """
    if speculate_len < 1:
        raise ValueError("--speculate_len must be >= 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = torch.device(device)
    is_cuda = device_type == "cuda"

    def sample_tok(logits: torch.Tensor) -> torch.Tensor:
        probs = sampling_probs_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        return torch.multinomial(probs, 1)

    # Vanilla decode (target-only)
    vanilla_tokens = 0
    vanilla_time = 0.0

    # Spec decode
    spec_tokens = 0
    spec_time = 0.0
    accepted_counts: list[int] = []
    overlap_vals: list[float] = []

    model.reference_model.eval()
    model.truncated_model.eval()
    if hasattr(model.reference_model, "config"):
        model.reference_model.config.use_cache = True
    if hasattr(model.truncated_model, "config"):
        model.truncated_model.config.use_cache = True

    with torch.inference_mode():
        for prompt_idx, prompt_ids in enumerate(prompts):
            if progress and (prompt_idx % max(1, int(progress_every)) == 0):
                print(f"[spec] prompt {prompt_idx+1}/{len(prompts)}", flush=True)
            prompt_ids = prompt_ids.to(device_obj)
            attn = torch.ones_like(prompt_ids, device=device_obj)

            # Init caches (not timed; decode focus).
            draft_past, draft_next_logits = _init_cache(
                model.truncated_model,
                prompt_ids,
                attn,
                device_type=device_type,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
            )
            target_past, target_next_logits = _init_cache(
                model.reference_model,
                prompt_ids,
                attn,
                device_type=device_type,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
            )

            if vanilla_toks_per_s_override is None:
                # Vanilla tokens/sec for this prompt
                v_attn = attn
                v_past = target_past
                v_next = target_next_logits
                v_generated = 0
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                while v_generated < max_new_tokens:
                    tok = sample_tok(v_next)
                    v_attn = torch.cat(
                        [v_attn, torch.ones((1, 1), device=device_obj, dtype=v_attn.dtype)], dim=1
                    )
                    v_past, v_next = _step_cache(
                        model.reference_model,
                        tok,
                        v_attn,
                        v_past,
                        device_type=device_type,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    v_generated += 1
                if is_cuda:
                    torch.cuda.synchronize()
                vanilla_time += time.time() - t0
                vanilla_tokens += v_generated

            # Spec decode for this prompt
            s_attn = attn
            s_draft_past = draft_past
            s_draft_next = draft_next_logits
            s_target_past = target_past
            s_target_next = target_next_logits
            s_generated = 0

            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            while s_generated < max_new_tokens:
                # ----- Proposal (advance draft locally) -----
                draft_tokens: list[torch.Tensor] = []
                draft_logits: list[torch.Tensor] = []
                draft_past_stack = [s_draft_past]

                local_draft_attn = s_attn
                local_draft_past = s_draft_past
                local_draft_next = s_draft_next

                for _ in range(speculate_len):
                    q_logits = local_draft_next
                    tok = sample_tok(q_logits)
                    draft_logits.append(q_logits)
                    draft_tokens.append(tok)

                    local_draft_attn = torch.cat(
                        [local_draft_attn, torch.ones((1, 1), device=device_obj, dtype=local_draft_attn.dtype)],
                        dim=1,
                    )
                    local_draft_past, local_draft_next = _step_cache(
                        model.truncated_model,
                        tok,
                        local_draft_attn,
                        local_draft_past,
                        device_type=device_type,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    draft_past_stack.append(local_draft_past)

                # ----- Verify (advance target only as tokens are accepted) -----
                accepted = 0
                overlaps: list[torch.Tensor] = []

                local_target_attn = s_attn
                local_target_past = s_target_past
                local_target_next = s_target_next

                rejected = False
                for j in range(speculate_len):
                    p_logits = local_target_next
                    q_logits = draft_logits[j]

                    p = sampling_probs_from_logits(p_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    q = sampling_probs_from_logits(q_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    overlaps.append(torch.minimum(p, q).sum(dim=-1))

                    tok = draft_tokens[j]
                    p_tok = p.gather(-1, tok)
                    q_tok = q.gather(-1, tok).clamp_min(1e-12)
                    accept_prob = torch.minimum(torch.ones_like(p_tok), p_tok / q_tok)
                    u = torch.rand_like(accept_prob)

                    if (u < accept_prob).all():
                        accepted += 1
                        local_target_attn = torch.cat(
                            [
                                local_target_attn,
                                torch.ones((1, 1), device=device_obj, dtype=local_target_attn.dtype),
                            ],
                            dim=1,
                        )
                        local_target_past, local_target_next = _step_cache(
                            model.reference_model,
                            tok,
                            local_target_attn,
                            local_target_past,
                            device_type=device_type,
                            use_autocast=use_autocast,
                            autocast_dtype=autocast_dtype,
                        )
                        continue

                    # Rejection at position j: sample from corrected distribution (p - q)+.
                    fix = torch.clamp(p - q, min=0.0)
                    fix = safe_normalize(fix, eps=1e-12)
                    if not torch.isfinite(fix).all() or (fix.sum(dim=-1) == 0).any():
                        fix = p
                    new_tok = torch.multinomial(fix, 1)

                    # Update global state to prefix + accepted draft tokens + new_tok.
                    accepted_tokens = draft_tokens[:accepted]
                    s_attn = torch.cat(
                        [
                            s_attn,
                            torch.ones((1, accepted + 1), device=device_obj, dtype=s_attn.dtype),
                        ],
                        dim=1,
                    )

                    # Target is already advanced through accepted tokens.
                    s_target_past = local_target_past
                    s_target_next = local_target_next
                    s_target_past, s_target_next = _step_cache(
                        model.reference_model,
                        new_tok,
                        s_attn,
                        s_target_past,
                        device_type=device_type,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )

                    # Draft rolls back to prefix + accepted tokens, then consumes new_tok.
                    s_draft_past = draft_past_stack[accepted]
                    s_draft_next = draft_logits[accepted]
                    s_draft_past, s_draft_next = _step_cache(
                        model.truncated_model,
                        new_tok,
                        s_attn,
                        s_draft_past,
                        device_type=device_type,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )

                    accepted_counts.append(accepted)
                    mean_overlap = float(torch.stack(overlaps, dim=0).mean().item()) if overlaps else 0.0
                    overlap_vals.append(mean_overlap)
                    s_generated += accepted + 1
                    rejected = True
                    break

                if rejected:
                    continue

                # All gamma accepted: sample one more token from target and advance both models.
                p_next = sampling_probs_from_logits(
                    local_target_next, temperature=temperature, top_p=top_p, top_k=top_k
                )
                extra_tok = torch.multinomial(p_next, 1)

                s_attn = torch.cat(
                    [s_attn, torch.ones((1, speculate_len + 1), device=device_obj, dtype=s_attn.dtype)],
                    dim=1,
                )

                # Target: already advanced through gamma accepted tokens.
                s_target_past = local_target_past
                s_target_next = local_target_next
                s_target_past, s_target_next = _step_cache(
                    model.reference_model,
                    extra_tok,
                    s_attn,
                    s_target_past,
                    device_type=device_type,
                    use_autocast=use_autocast,
                    autocast_dtype=autocast_dtype,
                )

                # Draft: already advanced through gamma draft tokens during proposal.
                s_draft_past = local_draft_past
                s_draft_next = local_draft_next
                s_draft_past, s_draft_next = _step_cache(
                    model.truncated_model,
                    extra_tok,
                    s_attn,
                    s_draft_past,
                    device_type=device_type,
                    use_autocast=use_autocast,
                    autocast_dtype=autocast_dtype,
                )

                accepted_counts.append(speculate_len)
                mean_overlap = float(torch.stack(overlaps, dim=0).mean().item()) if overlaps else 0.0
                overlap_vals.append(mean_overlap)
                s_generated += speculate_len + 1

            if is_cuda:
                torch.cuda.synchronize()
            spec_time += time.time() - t0
            spec_tokens += s_generated

    mean_accepted = float(np.mean(accepted_counts)) if accepted_counts else 0.0
    accept_rate = mean_accepted / speculate_len if speculate_len > 0 else 0.0
    mean_overlap = float(np.mean(overlap_vals)) if overlap_vals else 0.0
    mean_tv = 1.0 - mean_overlap

    vanilla_toks_per_s = (
        float(vanilla_toks_per_s_override)
        if vanilla_toks_per_s_override is not None
        else ((vanilla_tokens / vanilla_time) if vanilla_time > 0 else 0.0)
    )
    spec_toks_per_s = (spec_tokens / spec_time) if spec_time > 0 else 0.0
    speedup = (spec_toks_per_s / vanilla_toks_per_s) if vanilla_toks_per_s > 0 else 0.0

    return SpecMetrics(
        mean_accepted=mean_accepted,
        accept_rate=accept_rate,
        mean_overlap=mean_overlap,
        mean_tv=mean_tv,
        vanilla_toks_per_s=vanilla_toks_per_s,
        spec_toks_per_s=spec_toks_per_s,
        speedup=speedup,
    )


def load_prompts_from_dataloader(
    dataloader,
    *,
    num_prompts: int,
    prompt_length: int,
    device: str,
) -> list[torch.Tensor]:
    prompts: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(dataloader):
        if len(prompts) >= num_prompts:
            break
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        true_len = int(attention_mask.sum().item())
        pl = min(prompt_length, max(1, true_len - 1))
        prompts.append(input_ids[:, :pl].to(device))
    return prompts


def parse_step(path: str) -> Optional[int]:
    m = re.search(r"model_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_layer(path: str) -> Optional[int]:
    m = re.search(r"/layer(\d+)_", path.replace("\\\\", "/"))
    return int(m.group(1)) if m else None


def find_checkpoints(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return files


def select_latest_per_layer(paths: list[str]) -> list[str]:
    best: dict[int, tuple[int, str]] = {}
    for p in paths:
        layer = parse_layer(p)
        if layer is None:
            continue
        step = parse_step(p) or -1
        cur = best.get(layer)
        if cur is None or step > cur[0]:
            best[layer] = (step, p)
    return [best[k][1] for k in sorted(best.keys())]


def plot_results(df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
        }
    )

    def _save(fig, name: str):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300)
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close(fig)

    preferred_variant_order = ["baseline", "tuned", "teacher"]
    variants = [v for v in preferred_variant_order if v in set(df["variant"].unique())]
    variants += [v for v in sorted(df["variant"].unique()) if v not in set(variants)]

    label_map = {
        "spec_mean_accepted": "Mean accepted draft tokens (E[accepted])",
        "spec_accept_rate": "Acceptance rate (E[accepted]/γ)",
        "spec_speedup": "Spec decode speedup vs target-only",
    }

    dataset_label = {"sharegpt": "ShareGPT", "fineweb": "FineWeb", "ultrachat": "UltraChat"}

    # Layer sweep: quality (any dataset present in results)
    suffixes = [
        ("_ppl", "perplexity (draft)", True),
        ("_kl", "KL to teacher (nats/token)", False),
        ("_top1_acc", "top-1 next-token accuracy", False),
    ]
    datasets: list[str] = []
    for col in df.columns:
        for suf, _, _ in suffixes:
            if col.endswith(suf):
                datasets.append(col[: -len(suf)])
    datasets = sorted(set(datasets))

    for ds in datasets:
        ds_name = dataset_label.get(ds, ds)
        for suf, desc, logy in suffixes:
            metric = f"{ds}{suf}"
            if metric not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            for variant in variants:
                sdf = df[df["variant"] == variant].sort_values("layer")
                y = pd.to_numeric(sdf[metric], errors="coerce")
                mask = y.notna()
                if not mask.any():
                    continue
                ax.plot(sdf.loc[mask, "layer"], y.loc[mask], marker="o", label=variant)
            ax.set_xlabel("Exit layer ℓ (0-based)")
            title = f"{ds_name} {desc}"
            ax.set_ylabel(title)
            ax.set_title(title)
            if logy:
                ax.set_yscale("log")
            ax.legend(frameon=True)
            _save(fig, f"layer_sweep_{metric}")

    # Layer sweep: speculative acceptance + speed
    for metric in ["spec_mean_accepted", "spec_accept_rate", "spec_speedup"]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for variant in variants:
            sdf = df[df["variant"] == variant].sort_values("layer")
            y = pd.to_numeric(sdf[metric], errors="coerce")
            mask = y.notna()
            if not mask.any():
                continue
            ax.plot(sdf.loc[mask, "layer"], y.loc[mask], marker="o", label=variant)
        ax.set_xlabel("Exit layer ℓ (0-based)")
        ax.set_ylabel(label_map.get(metric, metric.replace("_", " ")))
        ax.set_title(label_map.get(metric, metric.replace("_", " ")))
        ax.legend(frameon=True)
        _save(fig, f"layer_sweep_{metric}")

    # Pareto: speedup vs quality (ppl)
    ppl_candidates = [c for c in ["ultrachat_ppl", "sharegpt_ppl", "fineweb_ppl"] if c in df.columns]
    if "spec_speedup" in df.columns and ppl_candidates:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        sdf = df[df["variant"] == "tuned"].sort_values("layer")
        ppl_col = ppl_candidates[0]
        ax.scatter(sdf["spec_speedup"], sdf[ppl_col])
        for _, row in sdf.iterrows():
            ax.annotate(f"ℓ={int(row['layer'])}", (row["spec_speedup"], row[ppl_col]))
        ax.set_xlabel("Spec decode speedup vs target-only")
        ax.set_ylabel(f"{ppl_col.replace('_ppl', '').title()} perplexity (draft logits)")
        ax.set_title("Speed–quality tradeoff (tuned)")
        _save(fig, "pareto_speedup_vs_ppl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate early-exit checkpoints and produce plots.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_glob", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--quality_max_batches", type=int, default=32)
    parser.add_argument("--quality_kl_temperature", type=float, default=None)
    parser.add_argument("--quality_datasets", type=str, default="sharegpt")
    parser.add_argument(
        "--skip_quality",
        action="store_true",
        default=False,
        help="Skip KL/CE/ppl/top1 evaluation and only run speculative-decoding metrics.",
    )
    parser.add_argument("--sharegpt_seed", type=int, default=0)
    parser.add_argument(
        "--sharegpt_filter_non_english",
        action="store_true",
        default=False,
        help="Filter ShareGPT to English-only using langdetect (slow; default off for eval).",
    )
    parser.add_argument("--ultrachat_seed", type=int, default=0)
    parser.add_argument(
        "--ultrachat_streaming",
        action="store_true",
        default=True,
        help="Use streaming UltraChat loading (recommended).",
    )
    parser.add_argument("--no_ultrachat_streaming", action="store_false", dest="ultrachat_streaming")
    parser.add_argument("--ultrachat_shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--ultrachat_val_examples", type=int, default=2048)
    parser.add_argument("--ultrachat_test_examples", type=int, default=2048)
    parser.add_argument(
        "--ultrachat_date_string",
        type=str,
        default="2026-01-20",
        help="Fixed date_string for tokenizer.apply_chat_template (improves reproducibility).",
    )
    parser.add_argument("--fineweb_seed", type=int, default=0)
    parser.add_argument("--fineweb_val_examples", type=int, default=256)
    parser.add_argument("--fineweb_test_examples", type=int, default=256)

    parser.add_argument("--speculate_len", type=int, default=8)
    parser.add_argument("--spec_temperature", type=float, default=1.0)
    parser.add_argument("--spec_top_p", type=float, default=0.95)
    parser.add_argument("--spec_top_k", type=int, default=0)
    parser.add_argument("--spec_max_new_tokens", type=int, default=128)
    parser.add_argument("--spec_num_prompts", type=int, default=64)
    parser.add_argument("--spec_prompt_length", type=int, default=256)
    parser.add_argument("--spec_seed", type=int, default=0)
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Print stage timing + per-prompt progress during evaluation.",
    )
    parser.add_argument("--progress_every", type=int, default=8)
    parser.add_argument(
        "--spec_prompt_dataset",
        type=str,
        default="auto",
        help="Dataset to draw spec-decoding prompts from: auto|ultrachat|sharegpt|fineweb.",
    )
    parser.add_argument(
        "--include_teacher",
        action="store_true",
        default=True,
        help="Include full-model (teacher) CE/ppl/top1 baseline in results.",
    )
    parser.add_argument("--no_include_teacher", action="store_false", dest="include_teacher")
    parser.add_argument("--teacher_layer_index", type=int, default=15)

    parser.add_argument(
        "--latest_per_layer",
        action="store_true",
        default=True,
        help="Evaluate only the latest checkpoint per exit layer.",
    )
    parser.add_argument("--no_latest_per_layer", action="store_false", dest="latest_per_layer")

    args = parser.parse_args()

    device_type = "cuda" if str(args.device).startswith("cuda") else str(args.device)
    model_dtype = resolve_torch_dtype(args.torch_dtype, device_type)
    attn_impl = resolve_attn_implementation(args.attn_implementation, device_type)
    use_autocast = device_type == "cuda" and model_dtype in (torch.bfloat16, torch.float16)
    autocast_dtype = model_dtype if use_autocast else None

    os.makedirs(args.out_dir, exist_ok=True)
    t_global0 = time.time()

    # Gather checkpoints.
    ckpts = find_checkpoints(args.checkpoint_glob)
    if args.latest_per_layer:
        ckpts = select_latest_per_layer(ckpts)
    if not ckpts:
        raise SystemExit("No checkpoints matched.")

    # Tokenizer (shared).
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if args.progress:
        print(f"[time] tokenizer: {time.time() - t0:.2f}s", flush=True)

    # Quality dataloaders (batch_size=1 for predictable token weighting).
    quality_datasets = [d.strip() for d in args.quality_datasets.split(",") if d.strip()]
    quality_loaders: dict[str, Any] = {}
    if not args.skip_quality and quality_datasets:
        t0 = time.time()
        for name in quality_datasets:
            if name == "sharegpt":
                _, _, val = get_sharegpt_dataloaders(
                    1,
                    tokenizer,
                    args.max_length,
                    generate_labels=True,
                    nice_shape=False,
                    seed=args.sharegpt_seed,
                    filter_non_english=args.sharegpt_filter_non_english,
                )
                quality_loaders["sharegpt"] = val
            elif name == "ultrachat":
                _, _, val = get_ultrachat_dataloaders(
                    1,
                    tokenizer,
                    args.max_length,
                    generate_labels=True,
                    nice_shape=False,
                    seed=args.ultrachat_seed,
                    streaming=args.ultrachat_streaming,
                    shuffle_buffer_size=args.ultrachat_shuffle_buffer_size,
                    val_examples=args.ultrachat_val_examples,
                    test_examples=args.ultrachat_test_examples,
                    date_string=args.ultrachat_date_string,
                )
                quality_loaders["ultrachat"] = val
            elif name == "fineweb":
                _, _, val = get_fineweb_dataloaders(
                    1,
                    tokenizer,
                    args.max_length,
                    generate_labels=True,
                    seed=args.fineweb_seed,
                    streaming=True,
                    val_examples=args.fineweb_val_examples,
                    test_examples=args.fineweb_test_examples,
                )
                quality_loaders["fineweb"] = val
            else:
                raise ValueError(f"Unknown dataset: {name}")
        if args.progress:
            print(f"[time] quality loaders: {time.time() - t0:.2f}s", flush=True)

    # Spec prompts: choose a dataset.
    spec_prompt_ds = str(args.spec_prompt_dataset).strip().lower()
    if spec_prompt_ds == "auto":
        if "ultrachat" in quality_loaders:
            spec_prompt_ds = "ultrachat"
        elif "sharegpt" in quality_loaders:
            spec_prompt_ds = "sharegpt"
        else:
            spec_prompt_ds = quality_datasets[0] if quality_datasets else "sharegpt"

    t0 = time.time()
    if spec_prompt_ds == "ultrachat":
        from datasets import load_dataset

        prompts: list[torch.Tensor] = []
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="test_sft",
            streaming=True,
        )
        for ex in ds:
            if len(prompts) >= args.spec_num_prompts:
                break
            prompt_text = ex.get("prompt")
            if not prompt_text:
                continue
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=args.max_length,
                date_string=args.ultrachat_date_string,
            )
            pl = min(int(args.spec_prompt_length), max(1, len(ids)))
            prompts.append(torch.tensor(ids[:pl], device=args.device, dtype=torch.long).unsqueeze(0))
    elif spec_prompt_ds == "fineweb":
        _, _, spec_val = get_fineweb_dataloaders(
            1,
            tokenizer,
            args.max_length,
            generate_labels=True,
            seed=args.fineweb_seed,
            streaming=True,
            val_examples=max(args.fineweb_val_examples, args.spec_num_prompts),
            test_examples=args.fineweb_test_examples,
        )
        prompts = load_prompts_from_dataloader(
            spec_val,
            num_prompts=args.spec_num_prompts,
            prompt_length=args.spec_prompt_length,
            device=args.device,
        )
    else:
        # ShareGPT (and fallback): use prefixes from the val loader.
        _, _, spec_val = get_sharegpt_dataloaders(
            1,
            tokenizer,
            args.max_length,
            generate_labels=True,
            nice_shape=False,
            seed=args.sharegpt_seed,
            filter_non_english=args.sharegpt_filter_non_english,
        )
        prompts = load_prompts_from_dataloader(
            spec_val,
            num_prompts=args.spec_num_prompts,
            prompt_length=args.spec_prompt_length,
            device=args.device,
        )
    if args.progress:
        print(f"[time] prompts ({spec_prompt_ds}, n={len(prompts)}): {time.time() - t0:.2f}s", flush=True)

    rows: list[dict[str, Any]] = []

    # Load the teacher model once (reuse across layers).
    t0 = time.time()
    teacher_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if attn_impl is not None:
        teacher_kwargs["attn_implementation"] = attn_impl
    if device_type == "cuda":
        teacher_kwargs["dtype"] = model_dtype
    try:
        teacher = AutoModelForCausalLM.from_pretrained(args.model_path, **teacher_kwargs)
    except TypeError:
        # Older transformers expects torch_dtype.
        if "dtype" in teacher_kwargs:
            teacher_kwargs["torch_dtype"] = teacher_kwargs.pop("dtype")
        teacher = AutoModelForCausalLM.from_pretrained(args.model_path, **teacher_kwargs)
    teacher.eval()
    teacher.to(args.device)
    for p in teacher.parameters():
        p.requires_grad = False
    if args.progress:
        print(f"[time] teacher load: {time.time() - t0:.2f}s", flush=True)

    # Compute target-only decode throughput once (shared by baseline/tuned variants).
    t0 = time.time()
    vanilla_toks_per_s = vanilla_decode_tokens_per_second(
        teacher,
        prompts,
        device=args.device,
        device_type=device_type,
        use_autocast=use_autocast,
        autocast_dtype=autocast_dtype,
        max_new_tokens=args.spec_max_new_tokens,
        temperature=args.spec_temperature,
        top_p=args.spec_top_p,
        top_k=args.spec_top_k,
        seed=args.spec_seed,
    )
    if args.progress:
        print(f"[time] vanilla decode toks/s: {time.time() - t0:.2f}s ({vanilla_toks_per_s:.2f} toks/s)", flush=True)

    if args.include_teacher:
        teacher_row: dict[str, Any] = {
            "checkpoint": None,
            "variant": "teacher",
            "layer": int(args.teacher_layer_index),
            "step": None,
            "train_kl_temperature": None,
        }
        if not args.skip_quality:
            for ds_name, loader in quality_loaders.items():
                qm = compute_causal_lm_metrics(
                    teacher,
                    loader,
                    device=args.device,
                    device_type=device_type,
                    use_autocast=use_autocast,
                    autocast_dtype=autocast_dtype,
                    max_batches=args.quality_max_batches,
                )
                teacher_row[f"{ds_name}_kl"] = None
                teacher_row[f"{ds_name}_ce"] = qm.ce
                teacher_row[f"{ds_name}_ppl"] = qm.ppl
                teacher_row[f"{ds_name}_top1_acc"] = qm.top1_acc

        teacher_row["vanilla_toks_per_s"] = vanilla_toks_per_s
        teacher_row["spec_toks_per_s"] = None
        teacher_row["spec_speedup"] = None
        teacher_row["spec_mean_accepted"] = None
        teacher_row["spec_accept_rate"] = None
        teacher_row["spec_mean_overlap"] = None
        teacher_row["spec_mean_tv"] = None
        rows.append(teacher_row)

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta = ckpt.get("meta", {})
        layer = meta.get("target_layer", parse_layer(ckpt_path))
        if layer is None:
            raise ValueError(f"Could not infer layer for checkpoint: {ckpt_path}")

        step = meta.get("step", parse_step(ckpt_path))
        train_temp = float(meta.get("kl_temperature", 1.0))
        kl_temp = float(args.quality_kl_temperature) if args.quality_kl_temperature is not None else train_temp

        # Draft model (one instance per layer); evaluate baseline then tuned by loading checkpoint weights.
        draft = TruncatedLlama(
            args.model_path,
            early_exit_idx=int(layer),
            reference_model=teacher,
            attn_implementation=attn_impl,
            torch_dtype=model_dtype if device_type == "cuda" else None,
            use_cache=True,
            ft_head=True,
            ft_last_transformer=bool(ckpt.get("last_transformer") is not None),
            lm_head_random_init=False,
        ).to(args.device)

        for variant_name in ("baseline", "tuned"):
            if variant_name == "tuned":
                draft.load_from_checkpoint(ckpt.get("lm_head"), ckpt.get("last_transformer"))

            row: dict[str, Any] = {
                "checkpoint": ckpt_path,
                "variant": variant_name,
                "layer": int(layer),
                "step": int(step) if step is not None else None,
                "train_kl_temperature": train_temp,
            }

            if not args.skip_quality:
                for ds_name, loader in quality_loaders.items():
                    qm = compute_quality_metrics(
                        draft,
                        loader,
                        device=args.device,
                        device_type=device_type,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                        kl_temperature=kl_temp,
                        max_batches=args.quality_max_batches,
                        compute_kl=True,
                    )
                    row[f"{ds_name}_kl"] = qm.kl
                    row[f"{ds_name}_ce"] = qm.ce
                    row[f"{ds_name}_ppl"] = qm.ppl
                    row[f"{ds_name}_top1_acc"] = qm.top1_acc

            sm = speculative_decode_metrics(
                draft,
                prompts,
                device=args.device,
                device_type=device_type,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
                max_new_tokens=args.spec_max_new_tokens,
                speculate_len=args.speculate_len,
                temperature=args.spec_temperature,
                top_p=args.spec_top_p,
                top_k=args.spec_top_k,
                seed=args.spec_seed,
                vanilla_toks_per_s_override=vanilla_toks_per_s,
                progress=args.progress,
                progress_every=args.progress_every,
            )
            row["spec_mean_accepted"] = sm.mean_accepted
            row["spec_accept_rate"] = sm.accept_rate
            row["spec_mean_overlap"] = sm.mean_overlap
            row["spec_mean_tv"] = sm.mean_tv
            row["vanilla_toks_per_s"] = vanilla_toks_per_s
            row["spec_toks_per_s"] = sm.spec_toks_per_s
            row["spec_speedup"] = sm.speedup
            rows.append(row)

        del draft
        if device_type == "cuda":
            torch.cuda.empty_cache()

    del teacher
    if device_type == "cuda":
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["variant", "layer"])
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    plot_results(df, os.path.join(args.out_dir, "plots"))

    print(f"Wrote: {os.path.join(args.out_dir, 'results.csv')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'plots')}")
    if args.progress:
        print(f"[time] total: {time.time() - t_global0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
