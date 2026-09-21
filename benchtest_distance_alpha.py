"""No-cost local smoke and performance benchmark for a DistanceAlpha checkpoint."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from distance_alpha import DistanceAlpha


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", default="user: Explain what this model is.\nassistant:")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.tokens < 1 or args.warmup < 0 or args.runs < 1:
        p.error("tokens, warmup, and runs must be positive")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    tokenizer_path = args.checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.is_file():
        p.error(f"missing tokenizer beside checkpoint: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = DistanceAlpha(**config)
    model.load_state_dict(saved["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    parameters = sum(x.numel() for x in model.parameters())

    sequence_length = min(args.sequence_length, config["block_size"])
    x = torch.randint(config["vocab_size"], (1, sequence_length), device=device)
    with torch.inference_mode():
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, auxiliary, effective = model(x)
    assert logits.shape == (1, sequence_length, config["vocab_size"])
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary).all() and torch.isfinite(effective).all()

    timings = []
    for _ in range(args.warmup):
        with torch.inference_mode():
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                model(x)
    sync(device)
    for _ in range(args.runs):
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                model(x)
        sync(device)
        timings.append(time.perf_counter() - started)

    ids = tokenizer.encode(args.prompt).ids[-config["block_size"]:]
    generated = []
    eos = tokenizer.token_to_id("<eos>")
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.tokens):
            sample = torch.tensor([ids[-config["block_size"]:]], device=device)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                output, _, _ = model(sample)
            next_id = output[0, -1].float().argmax().item()
            if next_id == eos:
                break
            ids.append(next_id)
            generated.append(next_id)
    sync(device)
    generation_seconds = time.perf_counter() - started

    report = {
        "checkpoint": str(args.checkpoint),
        "step": saved.get("step"),
        "device": str(device),
        "dtype": str(dtype),
        "parameters": parameters,
        "config": config,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "forward_sequence_length": sequence_length,
        "forward_mean_seconds": sum(timings) / len(timings),
        "forward_tokens_per_second": sequence_length / (sum(timings) / len(timings)),
        "generated_tokens": len(generated),
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": len(generated) / max(generation_seconds, 1e-9),
        "sample": tokenizer.decode(generated),
        "effective_memory": effective.item(),
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
    }
    report_path = args.report or args.checkpoint.parent / "benchtest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
