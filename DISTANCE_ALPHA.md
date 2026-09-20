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

## Benchmarking

Score a checkpoint against the held-out 5% suffix of the same corpus used for
training. The report contains cross-entropy loss, perplexity, effective memory
matches, total validation tokens, parameter count, and measured tokens per second.
Use the same corpus and `--max-blocks` setting when comparing runs.

```bash
python benchmark_distance_alpha.py \
  --checkpoint checkpoints/alpha-distance/latest.pt \
  --corpus code_corpus.txt \
  --max-blocks 100
```

For Hugging Face runs prepared by the chunked trainer, benchmark the persisted
validation chunks directly instead of loading a corpus into RAM:

```bash
python benchmark_distance_alpha.py \
  --checkpoint checkpoints/alpha-distance-1b/latest.pt \
  --chunk-dir checkpoints/alpha-distance-1b/token-chunks/validation \
  --max-blocks 100
```

The default report is `benchmark.json` beside the checkpoint. It is a checkpoint
benchmark, not a quality comparison with the original Alpha architecture. A fair
baseline must use the same tokenizer, corpus split, number of parameters, and
validation blocks.

## 1B profile and Hugging Face data

`--preset 1b` creates a fresh approximately 997M-parameter model: 32,000 BPE
tokens, 2,048 context tokens, 2,048 dimensions, 18 causal layers, 16 heads, and
the 4,096-entry distance memory. It cannot resume a small-model checkpoint.
The trainer prints the actual parameter count before its first optimizer update.

The following streams 200,000 UltraChat conversations from Hugging Face, trains a
tokenizer from the first 10,000 documents, and materializes the token stream into
one-million-token `.pt` chunks on disk. Only a small configurable chunk cache stays
in RAM while training. The tokenizer, chunks, and checkpoints share the output
directory:

```bash
python train_distance_alpha.py \
  --preset 1b \
  --hf-dataset HuggingFaceH4/ultrachat_200k \
  --hf-split train_sft \
  --hf-samples 200000 \
  --tokenizer-samples 10000 \
  --chunk-tokens 1000000 \
  --cache-chunks 4 \
  --steps 1000 \
  --batch-size 1 \
  --output checkpoints/alpha-distance-1b
```

Use `--resume` with exactly the same preset and architecture values to continue.
The token chunks are retained so resumed runs do not redownload or retokenize the
selected text.
This single-process trainer is an experiment harness. Training a 1B model needs
substantially more than a few hundred thousand conversations and usually needs
high-memory or multi-GPU hardware; it will not fit a 6GB GPU.

The model uses a GPT-style normal weight initialization (standard deviation .02).
An initial loss near the logarithm of the vocabulary size is expected; a very high
initial loss such as 100 indicates an unsuitable checkpoint or a broken run. The
default memory objective scale is .05, which pushes effective memory matches toward
eight. Check the `effective_matches` log: around 8 is the intended behavior, while
near 1 means one entry dominates and near 4,096 means retrieval is uniform.

The trainer uses a linear learning-rate warmup followed by cosine decay. The 1B
defaults are `--learning-rate 1e-4 --warmup-steps 1000 --schedule-steps 100000
--min-lr-ratio .1`. `schedule-steps` counts optimizer updates across resume runs.
