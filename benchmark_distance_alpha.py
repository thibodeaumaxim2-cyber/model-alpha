"""Reproducible held-out benchmark for DistanceAlpha checkpoints."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from distance_alpha import DistanceAlpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path,
                        help="The same corpus used during training.")
    source.add_argument("--chunk-dir", type=Path,
                        help="Validation chunk directory from the chunked trainer.")
    parser.add_argument("--max-blocks", type=int, default=100,
                        help="Number of non-overlapping validation blocks to score.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--report", type=Path,
                        help="Defaults beside the checkpoint as benchmark.json.")
    args = parser.parse_args()
    if args.max_blocks < 1 or args.batch_size < 1:
        parser.error("--max-blocks and --batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    tokenizer_path = args.checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.is_file():
        parser.error(f"Tokenizer missing beside checkpoint: {tokenizer_path}")
    width = config["block_size"] + 1
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    block_tensors = []
    if args.corpus:
        raw = args.corpus.read_text(encoding="utf-8")
        validation_ids = torch.tensor(tokenizer.encode(raw[int(len(raw) * .95):]).ids)
        for offset in range(0, len(validation_ids) - width + 1, width):
            block_tensors.append(validation_ids[offset:offset + width])
            if len(block_tensors) == args.max_blocks:
                break
        source_description = str(args.corpus)
    else:
        if not args.chunk_dir.is_dir():
            parser.error(f"Validation chunk directory does not exist: {args.chunk_dir}")
        for path in sorted(args.chunk_dir.glob("*.pt")):
            tokens = torch.load(path, map_location="cpu")
            for offset in range(0, tokens.numel() - width + 1, width):
                block_tensors.append(tokens[offset:offset + width])
                if len(block_tensors) == args.max_blocks:
                    break
            if len(block_tensors) == args.max_blocks:
                break
        source_description = str(args.chunk_dir)
    if not block_tensors:
        parser.error("Validation data is shorter than one model block")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = DistanceAlpha(**config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    total_nll = total_tokens = 0.0
    effective_matches = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start_block in range(0, len(block_tensors), args.batch_size):
            batch = torch.stack(block_tensors[start_block:start_block + args.batch_size]).long().to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits, _, effective = model(x)
                nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
            total_nll += nll.item()
            total_tokens += y.numel()
            effective_matches.append(effective.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    loss = total_nll / total_tokens
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": saved.get("step"),
        "validation_source": source_description,
        "device": str(device),
        "parameters": model_parameters,
        "validation_tokens": int(total_tokens),
        "validation_blocks": len(block_tensors),
        "cross_entropy_loss": loss,
        "perplexity": math.exp(min(loss, 80)),
        "mean_effective_memory_matches": sum(effective_matches) / len(effective_matches),
        "tokens_per_second": total_tokens / elapsed,
        "elapsed_seconds": elapsed,
    }
    report_path = args.report or args.checkpoint.parent / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
