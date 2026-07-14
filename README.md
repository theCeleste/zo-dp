# DPZero for Llama 2 7B

This directory adapts the OPT implementation to `meta-llama/Llama-2-7b-hf`.
It covers zero-shot/ICL evaluation, regular fine-tuning, LoRA, head tuning,
linear probing, direct KV prefix tuning, MeZO, and DPZero.

## Setup

Request access to Llama 2 on Hugging Face, then authenticate:

```bash
huggingface-cli login
pip install -r requirements.txt
```

The first implementation remains pinned to `transformers==4.28.1`, matching the
Trainer internals vendored by the original project. Upgrading Transformers requires
first replacing the copied training loop with a small `Trainer` extension.

## Recommended mode

For a 7B model, start with BF16 LoRA. Full-parameter zeroth-order training does
not retain backward activations, but it still perturbs and regenerates random
directions for every trainable parameter on every step.

```bash
cd llama
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-5 EPS=1e-3 bash examples/mezo.sh
```

DPZero:

```bash
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-5 EPS=1e-3 DP_EPS=6 DP_CLIP=10 \
  bash examples/dpzero.sh
```

Regular LoRA fine-tuning:

```bash
MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 MODE=lora \
  BS=4 LR=1e-4 bash examples/finetune.sh
```

Head tuning uses the same scripts with `MODE=head`. Linear probing is available
through the direct `run.py` interface with `--linear_probing`.

Prefix tuning uses `MODE=prefix`. It provides trainable per-layer KV cache entries
in Llama's native key/value-head layout. The current implementation supports the
legacy tuple cache used by the pinned Transformers 4.28.1 environment and direct
KV parameters (`--no_reparam`). With `--prefix_init_by_real_act`, it runs a short
no-gradient pass over randomly sampled real token IDs and initializes every layer
from the resulting RoPE-transformed native Llama KV cache.

Important constraints:

- DPZero requires `--trainer zo`, `--only_train_option`, and gradient accumulation 1.
- DPZero automatically enables `dataloader_drop_last` so every privacy step uses
  the configured fixed batch size.
- `--load_float16` and `--load_bfloat16` are mutually exclusive; BF16 is recommended.
- 8-bit loading is accepted only with LoRA.
- The privacy log reports sample rate, noise multiplier, Gaussian standard deviation,
  epsilon, and delta. Candidate-expanded classification batches are still accounted
  in terms of original examples by the configured per-device batch size.

Every DPZero checkpoint includes `dpzero_privacy.json` with the sampling rate,
noise multiplier, Gaussian standard deviation, batch and dataset sizes, clipping
threshold, finite-difference epsilon, world size, and planned step count. Resume
requires these privacy-relevant arguments to match exactly. To extend a completed
privacy schedule, start a new run and explicitly compose the two privacy budgets;
do not treat it as an ordinary same-budget resume.

The task names and prompt templates remain the same as in `opt/`.

## Loader messages

With PyTorch 2.4, Transformers 4.28 may emit a `torch.load(...,
weights_only=False)` FutureWarning while loading legacy `.bin` checkpoint shards.
This call is inside Transformers 4.28 and cannot be changed through the model API.
Use only the official gated Llama checkpoint or another checkpoint you trust. Do
not globally suppress the warning for arbitrary model paths. A future Trainer/API
upgrade should prefer safetensors and a Transformers version that passes
`weights_only=True` where applicable.

SentencePiece may print `precompiled_charsmap is empty; use identity normalization`.
That is an informational message for the tokenizer bundled with the checkpoint,
not a training or tokenization failure.

The Llama entry point defaults to `--optim adamw_torch` instead of the deprecated
Transformers 4.28 AdamW implementation. It can still be overridden explicitly on
the command line.

## Adapter checkpoints

LoRA, LM-head, and Prefix runs save adapter-only checkpoints by default. Each
checkpoint contains `pytorch_model.bin`, `adapter_manifest.json`, model config,
trainer state, optimizer, and scheduler state. On resume, reconstruct the same
mode through the original command; the trainer validates the manifest and loads
only the declared adapter tensors. Use `--save_adapter_only False` to request a
full model checkpoint. Full-parameter fine-tuning always saves the full model.

Adapter-only saving currently targets ordinary single-process/device-map runs.
FSDP and DeepSpeed retain their native full/sharded checkpoint behavior.

After the tiny checkpoint tests pass, run one real Llama 2 checkpoint smoke test:

```bash
MODE=prefix bash examples/validate_7b_adapter_checkpoint.sh
```

The script uses a new timestamped output directory, saves after one MeZO step,
reconstructs the base model and Prefix module, resumes to step two, verifies both
trainer states, and prints the adapter weight file sizes.

## Formal experiments and performance benchmarks

The locked baseline matrix is defined in `configs/formal_experiments.yaml`:

- `mezo_lr_sweep`: 6 dev-only LoRA/Prefix LR calibration jobs;
- `head_dp_clip_sweep`: 3 dev-only Head clipping calibration jobs;
- `zero_shot_control`: 1 evaluation-only control;
- `mezo_utility_pilot`: 3 seed-0, 1000-step utility pilots;
- `dpzero_utility_pilot`: 3 matched epsilon-6 utility pilots;
- `performance_smoke`: 3 short jobs, one for each PEFT mode;
- `mezo_baseline`: 9 jobs, three modes by three seeds;
- `dpzero_budget_sweep`: 36 jobs, three modes by four nominal epsilon levels by three seeds.

List a suite without executing anything:

```bash
python run_formal_experiment.py --suite performance_smoke
```

Execute exactly one selected job:

```bash
python run_formal_experiment.py --suite performance_smoke --index 0 --run
```

Bulk execution is intentionally disabled. Existing output directories are refused
unless `--resume` is supplied, and a resume command must match the saved
`run_config.json`. Each run records the exact command and configuration SHA-256.

Every training run writes `benchmark.json` with model load time, training wall
time, Trainer throughput, parameter counts, software/hardware versions, per-GPU
peak allocated/reserved memory, and DP settings. Peak memory is reset after model
loading, so it represents the live model plus training peak, not transient loader
allocations. Post-training evaluation is stored separately in `eval_metrics.json`.

Aggregate completed runs with:

```bash
python summarize_benchmarks.py \
  --root result/formal \
  --output result/formal/summary
```

This produces per-run `runs.csv` and grouped `groups.csv`/`groups.json` summaries.
The formal DP epsilon values retain the current Opacus-accounting convention
documented above; this configuration does not change the sampling implementation.

After utility pilots pass, run formal jobs sequentially by stage. Inspect first:

```bash
python run_formal_stage.py --stage mezo_baseline
```

Start only with an exact confirmation token:

```bash
python run_formal_stage.py \
  --stage mezo_baseline \
  --run \
  --confirm mezo_baseline
```

The stage runner skips outputs that already contain valid config, benchmark,
evaluation, and metric artifacts. An incomplete output is resumed; any process
failure, hard timeout, or artifact validation failure stops the stage immediately
without automatic retry. Progress is written under `result/formal/stages/`.

After reviewing the MeZO baseline, run only DPZero seed 0:

```bash
python run_formal_stage.py --stage dpzero_seed0 --run --confirm dpzero_seed0
```

Seed 1 and seed 2 are separate stages and should start only after seed-0 review.
