# Model Alpha

`train_alpha.py` provides an approximately 500M-parameter from-scratch causal LM profile: vocabulary 32,000, context 2,048, dimension 1,536, 16 layers, 24 heads, 4× FFN, and tied input/output embeddings. It enables CUDA bf16 autocast and TF32, uses fused AdamW, gradient accumulation, pinned-memory loading, validation, cosine decay with warmup, gradient clipping, resumable numbered checkpoints, and optional `torch.compile`.

The default output is `checkpoints/alpha-500m/`; older checkpoint directories are preserved. `parameter_report.json` records the exact count and training refuses counts outside 480M–530M. The default effective batch is 32 (`micro-batch-size=1`, `grad-accum=32`). The default production data guard requires 300,000,000 source tokens.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch wheel matching the A100 from https://pytorch.org/
python -m pip install -r requirements.txt
```

## One-step GPU smoke test

This uses the tiny local corpus and the exact production architecture. It creates a parameter report and fails if the model cannot fit in GPU memory.

```bash
python train_alpha.py --code-file smoke_data.txt --min-train-tokens 0 \
  --epochs 1 --micro-batch-size 1 --grad-accum 1 --workers 0 \
  --save-every 1 --log-every 1 --checkpoint-dir alpha-500m-smoke --smoke-test
```

## Production training

Use a properly licensed, shuffled corpus containing at least several hundred million tokens; keep validation data separate. The trainer refuses to start below 300M tokens.

```bash
python train_alpha.py --chat-samples 200000 --wikipedia-samples 100000 \
  --micro-batch-size 1 --grad-accum 32 --epochs 1 --learning-rate 1e-4 \
  --checkpoint-dir alpha-500m --workers 4 --compile
```

For a local blank-line-separated corpus: `python train_alpha.py --code-file corpus.txt --checkpoint-dir alpha-500m`. Resume with `--resume`; it restores model, optimizer, scheduler, and step state.

