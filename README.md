# Model Alpha: PyTorch conversational neural network

Alpha fine-tunes `microsoft/DialoGPT-small` with PyTorch and Hugging Face Transformers on the Hugging Face `daily_dialog` dataset. It generates replies and keeps recent conversation turns as context.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

The first command downloads the pretrained model and dataset:

```bash
python conversational_model.py train --max-samples 5000
```

For a stronger experiment, use more examples and epochs:

```bash
python conversational_model.py train --max-samples 30000 --epochs 2
```

The checkpoint is saved under `checkpoints/alpha-dialogue/`. CPU training may take a while; CUDA is used automatically when available.

## Chat

```bash
python conversational_model.py chat
```

Use `/reset` to clear context and `/quit` to exit. This is a real pretrained neural language model, so it can discuss topics outside fixed intents. It can still produce incorrect or repetitive replies, especially after a small fine-tuning run.

## How it works

Dialogue turns are formatted as alternating `User` and `Assistant` messages. The tokenizer converts them to token IDs. The transformer predicts the next token, and PyTorch updates its weights using cross-entropy loss, backpropagation, and AdamW. At chat time, `generate()` samples a response from the learned token distribution.

`conversational_model.py` contains dataset loading, tokenization, training, checkpoint saving, history construction, and generation. The earlier `alpha.py` intent classifier remains as a simpler comparison.

## Our own model from random weights

Use `scratch_chat.py` for a model whose tokenizer, vocabulary, Transformer architecture, and weights are created in this repository. It downloads only the Hugging Face `daily_dialog` dataset:

```bash
python scratch_chat.py train --max-samples 5000 --epochs 3
python scratch_chat.py chat
```

This TinyGPT has four Transformer layers and starts with random weights. It learns next-token prediction with PyTorch. Increase `--max-samples` and `--epochs` for better results; expect rough conversation because this is a small model trained from scratch.

## Long-term training pipeline

`train_alpha.py` is the scalable from-scratch trainer. It trains a BPE tokenizer, packs mixed data into fixed-size blocks, reserves validation data, uses CUDA mixed precision when available, and saves a full resumable checkpoint.

```bash
pip install -r requirements.txt
python train_alpha.py --chat-samples 20000 --wikipedia-samples 5000
```

To add a local, license-reviewed code corpus whose files are separated by blank lines:

```bash
python train_alpha.py --code-file code_corpus.txt --resume
```

Training state is kept in `checkpoints/alpha-v2/`. Resume an interrupted run with `--resume`; it restores model weights, optimizer state, mixed-precision scaler, training step, tokenizer, and settings.

## Benchmark over time

Run the stable Alpha Base benchmark after each training run:

```bash
python benchmark_alpha.py --checkpoint checkpoints/alpha-v2/latest.pt --name alpha-base-v1
```

It reports per-category next-token loss, perplexity, and exact completion rate for language, knowledge, code, and conversation examples. Keep `benchmarks/alpha_base_v1.jsonl` unchanged so reports from different checkpoints remain comparable.
