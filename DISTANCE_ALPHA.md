# Experimental distance-memory Alpha

This separate experimental architecture does not load existing Alpha checkpoints.
It uses a causal Transformer, normalized 256-dimensional queries, 64 learned
cluster centers, and 64 key/value memories per cluster. All clusters receive soft
Gaussian weights; within each cluster, members receive Gaussian distance weights.
Their product weights the retrieved content. A learned gate blends it into context.
The radius is learned per token. A weak auxiliary objective targets eight effective
contributors, measured as 1/sum(weight²), with gentle load balancing.

There is no top-eight cutoff. All 4,096 entries participate (very small weights can
underflow numerically). Clustering is differentiable and soft; this version is not
an approximate nearest-neighbor index and makes no speed or quality improvement
claim. Query chunking bounds score allocation, but training retains autograd state.

Use a UTF-8 corpus and an existing tokenizer JSON. For rigorous evaluation, that
tokenizer must have been fitted on training-only data. The trainer splits the raw
corpus into a 95% training prefix and 5% validation suffix before encoding. It loads
the corpus into RAM; this is an experiment harness, not a streaming pretrainer.

```bash
python train_distance_alpha.py --corpus code_corpus.txt --tokenizer tokenizer.json --steps 1000 --output checkpoints/alpha-distance
```

Use an absolute Google Drive output path in Colab for persistent saves. Add
`--resume` to run additional steps with the saved tokenizer/model/optimizer/scaler.
Resume does not restore the exact data RNG position. Existing checkpoints require
explicit resume. Saves replace latest.pt atomically on supported filesystems; keep
independent backups for Drive disconnections. Validation currently reports a single
fixed sampled batch, not the full validation set.

CUDA uses BF16 when supported, otherwise FP16 with scaling; CPU uses FP32.
The script prints the actual parameter count and effective contributor count.
No pretrained weights or inference/chat interface are supplied yet. Existing
chat_alpha.py is for the original architecture. Compare this experiment against a
parameter-matched baseline before scaling up or claiming an improvement.

```bash
python -m unittest test_distance_alpha
```
