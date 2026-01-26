# Overnight queue (2026-01-25)

## Context

Goal: run 2–3 full (train + eval) two-stage UltraChat experiments overnight with production-like spec-decode eval (`temp=0.7`, `top_p=0.95`).

## Current run (already started)

- **Session**: `two_stage_ultrachat_20260125_030225`
- **Training PID**: `12997`
- **Log**: `early-exit-tests/logs/two_stage_ultrachat_20260125_030225_20260125_030225.log`
- **RUN_TYPE_BASE**: `two_stage_ultrachat_layers11_13_20260125_030225`
- **Layers**: `11 13`
- **Stage steps**: head-only `2000`, ft-last `2000`
- **Training hyperparams**:
  - `MAX_LENGTH=2048`, `KL_TEMP=1.0`
  - head: `BATCH_SIZE=2`, `GRAD_ACCUM=8`, `LR=5e-5`, `lr_schedule=linear`
  - ft-last: `BATCH_SIZE=2`, `GRAD_ACCUM=8`, `LR=2e-5`, `lr_schedule=linear`
- **Eval hyperparams**:
  - `SPEC_T=0.7`, `SPEC_TOP_P=0.95`, `SPEC_TOP_K=0`, `SPEC_GAMMA=8`
  - `SPEC_NUM_PROMPTS=32`, `SPEC_PROMPT_LEN=256`, `SPEC_MAX_NEW=64`
  - `QUALITY_MAX_BATCHES=16`
- **Artifacts**:
  - checkpoints under `checkpoints/models/Llama-3.2-1B-Instruct/two_stage_ultrachat_layers11_13_20260125_030225_head/` (and later `..._ftlast/`)
  - evals under `evaluations/two_stage_ultrachat_layers11_13_20260125_030225_head/` (and later `..._ftlast/`)

## Queued runs (will start after current PID exits)

Queue runner:
- **Queue PID**: `13990`
- **Queue log**: `early-exit-tests/logs/night_queue_20260125_034946_20260125_034946.log`
- **Waits for**: `PID 12997` to exit

### Experiment A: deeper exit “upper bound”

- **Name**: `ultrachat_L14_const_2k_2k`
- **Layers**: `14`
- **Stage steps**: head-only `2000`, ft-last `2000`
- **LR schedule**: `constant` for both stages
- **LRs**: head `5e-5`, ft-last `2e-5`
- **RUN_TYPE_BASE**: `night_ultrachat_L14_const_2k_2k_<timestamp>`

### Experiment B: longer optimization at a strong exit

- **Name**: `ultrachat_L13_const_4k_4k`
- **Layers**: `13`
- **Stage steps**: head-only `4000`, ft-last `4000`
- **LR schedule**: `constant` for both stages
- **LRs**: head `5e-5`, ft-last `2e-5`
- **RUN_TYPE_BASE**: `night_ultrachat_L13_const_4k_4k_<timestamp>`

## Monitoring

- Current run: `tail -f early-exit-tests/logs/two_stage_ultrachat_20260125_030225_20260125_030225.log`
- Queue: `tail -f early-exit-tests/logs/night_queue_20260125_034946_20260125_034946.log`

## Notes

- `train.py` was updated to respect `WANDB_MODE` so we can run W&B in offline mode without a login.
- `run_sweep_ultrachat_two_stage.sh` was updated to accept `LR_SCHEDULE_HEAD` / `LR_SCHEDULE_FTLAST` for easy `constant` LR runs.
