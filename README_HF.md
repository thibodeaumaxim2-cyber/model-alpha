# Model Alpha

Model Alpha is an experimental causal language model built with PyTorch for text generation and conversational AI research.

## Model

- Approximately 25–35 million parameters
- Vocabulary: 32,000 tokens
- Context length: 512 tokens
- Embedding dimension: 512
- Transformer layers: 6
- Attention heads: 8
- Training objective: causal next-token prediction

## Installation

```bash
git clone https://github.com/thibodeaumaxim2-cyber/model-alpha.git
cd model-alpha

sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download the Trained Model

The trained checkpoint is available on Hugging Face at
[`MaxiMThi/FewHours`](https://huggingface.co/MaxiMThi/FewHours).

Download it into the directory expected by the chat script:

```bash
python -m pip install -U huggingface_hub
mkdir -p checkpoints/alpha-25m

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MaxiMThi/FewHours",
    repo_type="model",
    local_dir="checkpoints/alpha-25m",
)
PY
```

Check the downloaded files:

```bash
find checkpoints/alpha-25m -maxdepth 2 -type f
```

## Training

```bash
python train_alpha.py \
  --vocab-size 32000 \
  --block-size 512 \
  --dim 512 \
  --heads 8 \
  --layers 6 \
  --micro-batch-size 4 \
  --grad-accum 8 \
  --chat-samples 50000 \
  --wikipedia-samples 10000 \
  --min-train-tokens 60000000 \
  --epochs 10 \
  --learning-rate 1e-4 \
  --workers 2 \
  --checkpoint-dir checkpoints/alpha-25m
```

CUDA is used automatically when an NVIDIA GPU is available. CPU training is supported but significantly slower.

## Continue Training

```bash
python train_alpha.py \
  --chat-samples 50000 \
  --wikipedia-samples 10000 \
  --micro-batch-size 4 \
  --grad-accum 8 \
  --min-train-tokens 60000000 \
  --epochs 10 \
  --learning-rate 1e-4 \
  --workers 2 \
  --checkpoint-dir checkpoints/alpha-25m \
  --resume
```

When using `--resume`, keep the original values for `--vocab-size`, `--block-size`, `--dim`, `--heads`, and `--layers`.

## Chat

```bash
python chat_alpha.py \
  --checkpoint checkpoints/alpha-25m/latest.pt
```

If the checkpoint has a different filename, locate it with:

```bash
find checkpoints/alpha-25m -type f \( -name "*.pt" -o -name "*.pth" \)
```

Then pass the discovered file to `--checkpoint`.

Available chat commands:

```text
/reset
/quit
```

## Benchmark

```bash
python benchmark_alpha.py \
  --checkpoint checkpoints/alpha-25m/latest.pt \
  --name alpha-25m
```

## Chat Compatibility Fix

If `chat_alpha.py` reports `ImportError: cannot import name 'RUNS'`, apply this
one-time fix for newer trainer versions:

```bash
sed -i 's/from train_alpha import AlphaTransformer, RUNS/from train_alpha import AlphaTransformer/' chat_alpha.py
sed -i 's/default=RUNS \/ "latest.pt"/default=Path("checkpoints\/alpha-25m\/latest.pt")/' chat_alpha.py
```

Then run:

```bash
python chat_alpha.py --checkpoint checkpoints/alpha-25m/latest.pt
```

## Troubleshooting

If the GPU runs out of memory, reduce the micro-batch size and increase gradient accumulation:

```text
--micro-batch-size 1 --grad-accum 32
```

If Python reports `externally-managed-environment`, create and activate the virtual environment shown in the installation section.

## Limitations

This is an experimental model and may produce repetitive, inaccurate, or hallucinated text. It should not be used for medical, legal, financial, or safety-critical applications.

## License

Specify the license for this project and verify that the training data and model weights can legally be redistributed.
