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
