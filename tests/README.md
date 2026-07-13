# Checkpoint validation

Run the offline tiny-Llama checkpoint suite from the `llama` directory:

```bash
python tests/validate_checkpoint.py --mode all
```

It validates LoRA, LM-head tuning, and direct-KV Prefix tuning without downloading
Llama 2 7B. For each mode it trains one MeZO step, saves `checkpoint-1`, rebuilds
the model and adapter, resumes to step two, verifies `checkpoint-2`, and checks
the exact family of trainable parameter names.

Temporary checkpoints are written under `tests/_checkpoint_validation`. The test
deletes only the selected mode's directory under that test root before rerunning.

Validate the zeroth-order and DPZero mathematical paths with:

```bash
python tests/validate_zo_math.py
```

This checks parameter restoration after symmetric perturbations, compares the
central finite difference against an autograd directional derivative, verifies
DP clipping, and confirms that DPZero receives one finite loss per example.

Validate the real Llama tokenizer boundary logic and causal-LM padding behavior:

```bash
python tests/validate_prompt_and_padding.py \
  --model_name meta-llama/Llama-2-7b-hf
```

This loads only the tokenizer from the named checkpoint. Model-side checks use a
tiny local Llama configuration. It verifies answer token boundaries, left
truncation, BOS preservation, padding-invariant option loss, genuine token ID 0,
and candidate-expanded classification loss cardinality.

Validate generation and cache behavior without downloading model weights:

```bash
python tests/validate_generation.py
```

This checks ordinary generation, the initial Prefix KV-cache length, three-token
Prefix cache continuation with a left-padded batch, and the project's batched
`zo_forward_nondiff` generation/F1 objective.

Run the complete parameter-efficient training matrix with:

```bash
python tests/validate_training_matrix.py
```

This runs one tiny-Llama step for all nine combinations of LoRA, LM-head tuning,
and Prefix tuning with regular backpropagation, MeZO, and DPZero. Every combination
must produce a finite loss, reach global step one, and modify at least one of its
declared trainable tensors.
