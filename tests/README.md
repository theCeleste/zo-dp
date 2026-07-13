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
