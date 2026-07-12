# DPZero for Llama 2 7B

This directory adapts the OPT implementation to `meta-llama/Llama-2-7b-hf`.
The first supported release covers zero-shot/ICL evaluation, regular fine-tuning,
LoRA, MeZO, and DPZero. Prefix tuning, head tuning, and linear probing fail fast
with an explicit error because their original implementations are OPT-specific.

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

Important constraints:

- DPZero requires `--trainer zo`, `--only_train_option`, and gradient accumulation 1.
- `--load_float16` and `--load_bfloat16` are mutually exclusive; BF16 is recommended.
- 8-bit loading is accepted only with LoRA.
- The privacy log reports sample rate, noise multiplier, Gaussian standard deviation,
  epsilon, and delta. Candidate-expanded classification batches are still accounted
  in terms of original examples by the configured per-device batch size.

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
