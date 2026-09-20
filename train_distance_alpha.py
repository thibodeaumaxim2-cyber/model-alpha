"""Train DistanceAlpha from RAM-bounded token chunks stored on disk."""
import argparse
from collections import OrderedDict
import itertools
import json
import math
import os
from pathlib import Path
import random

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn import functional as F

from distance_alpha import DistanceAlpha


PRESETS = {
    "small": dict(block_size=256, dim=256, heads=8, layers=4, clusters=64, slots=64),
    "1b": dict(block_size=2048, dim=2048, heads=16, layers=18, clusters=64, slots=64),
}
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def dataset_text(row):
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)
    for field in ("text", "content", "prompt", "conversation"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    return json.dumps(row, ensure_ascii=False)


def corpus_documents(path):
    lines = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                lines.append(line.rstrip("\n"))
            elif lines:
                yield "\n".join(lines)
                lines = []
    if lines:
        yield "\n".join(lines)


def hf_documents(args):
    dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    for row in itertools.islice(dataset, args.hf_samples):
        text = dataset_text(row)
        if text.strip():
            yield text


def parameter_count(config):
    d, v, b, layers = config["dim"], config["vocab_size"], config["block_size"], config["layers"]
    clusters, slots = config["clusters"], config["slots"]
    return v*d + b*d + layers*(12*d*d + 13*d) + clusters*d*(1 + 2*slots) + d*d + 4*d + 2


def save_chunk(directory, number, token_ids):
    path = directory / f"{number:06d}.pt"
    temp = path.with_suffix(".tmp")
    torch.save(torch.tensor(token_ids, dtype=torch.int32), temp)
    os.replace(temp, path)


def build_chunks(args, out, tokenizer):
    chunk_root = out / "token-chunks"
    manifest_path = chunk_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if chunk_root.exists() and any(chunk_root.iterdir()):
        raise SystemExit(f"Incomplete token chunk directory: {chunk_root}. Move it aside before retrying.")
    train_dir, valid_dir = chunk_root / "train", chunk_root / "validation"
    train_dir.mkdir(parents=True); valid_dir.mkdir(parents=True)
    documents = hf_documents(args) if args.hf_dataset else corpus_documents(args.corpus)
    eos = tokenizer.token_to_id("<eos>")
    buffers = {"train": [], "validation": []}
    directories = {"train": train_dir, "validation": valid_dir}
    counts = {"train": 0, "validation": 0}
    numbers = {"train": 0, "validation": 0}
    for index, document in enumerate(documents):
        split = "validation" if index % 20 == 0 else "train"
        buffers[split].extend(tokenizer.encode(document).ids + [eos])
        while len(buffers[split]) >= args.chunk_tokens:
            save_chunk(directories[split], numbers[split], buffers[split][:args.chunk_tokens])
            counts[split] += args.chunk_tokens; numbers[split] += 1
            buffers[split] = buffers[split][args.chunk_tokens:]
    for split in buffers:
        if buffers[split]:
            save_chunk(directories[split], numbers[split], buffers[split])
            counts[split] += len(buffers[split]); numbers[split] += 1
    manifest = {"train_tokens": counts["train"], "validation_tokens": counts["validation"],
                "train_chunks": numbers["train"], "validation_chunks": numbers["validation"],
                "chunk_tokens": args.chunk_tokens}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class DiskBlockSampler:
    def __init__(self, directory, block_size, cache_chunks, seed):
        self.files = sorted(directory.glob("*.pt"))
        self.files = [path for path in self.files if torch.load(path, map_location="cpu").numel() > block_size]
        if not self.files:
            raise SystemExit(f"No {block_size + 1}-token blocks in {directory}")
        self.block_size = block_size
        self.cache_chunks = cache_chunks
        self.cache = OrderedDict()
        self.random = random.Random(seed)

    def load(self, path):
        if path not in self.cache:
            self.cache[path] = torch.load(path, map_location="cpu")
            while len(self.cache) > self.cache_chunks:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def batch(self, batch_size, device):
        sequences = []
        for _ in range(batch_size):
            data = self.load(self.random.choice(self.files))
            start = self.random.randrange(data.numel() - self.block_size)
            sequences.append(data[start:start + self.block_size + 1])
        tokens = torch.stack(sequences).long().to(device, non_blocking=device.type == "cuda")
        return tokens[:, :-1], tokens[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--hf-dataset", help="Hugging Face dataset ID; tokenized in disk-backed chunks.")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-samples", type=int, default=100000)
    parser.add_argument("--tokenizer", type=Path, help="Optional existing BPE tokenizer.")
    parser.add_argument("--tokenizer-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/alpha-distance"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--preset", choices=PRESETS, default="small")
    for name in ("block_size", "dim", "heads", "layers", "clusters", "slots"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--schedule-steps", type=int, default=100_000,
                        help="Total optimizer steps for cosine decay, including resumed steps.")
    parser.add_argument("--min-lr-ratio", type=float, default=.1)
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000)
    parser.add_argument("--cache-chunks", type=int, default=4)
    parser.add_argument("--memory-loss-scale", type=float, default=.05,
                        help="Weight for the effective-memory-match objective.")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    architecture = PRESETS[args.preset].copy()
    for key in ("block_size", "dim", "heads", "layers", "clusters", "slots"):
        value = getattr(args, key)
        if value is not None: architecture[key] = value
    if min(args.steps, args.batch_size, args.save_every, args.chunk_tokens, args.cache_chunks, args.tokenizer_samples) < 1:
        parser.error("Numeric arguments must be positive")
    if args.memory_loss_scale < 0:
        parser.error("--memory-loss-scale must be non-negative")
    if args.learning_rate <= 0 or args.warmup_steps < 0 or args.schedule_steps < 1 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("Invalid learning-rate schedule arguments")
    if args.hf_dataset and args.hf_samples < 1:
        parser.error("--hf-samples must be positive")
    if architecture["dim"] % architecture["heads"]:
        parser.error("--dim must be divisible by --heads")

    out = args.output; out.mkdir(parents=True, exist_ok=True)
    checkpoint, tokenizer_path = out / "latest.pt", out / "tokenizer.json"
    if checkpoint.exists() and not args.resume:
        raise SystemExit("Checkpoint exists: use --resume or a new --output.")
    if args.resume and not checkpoint.exists():
        raise SystemExit("Resume requested but checkpoint is missing.")
    if args.resume:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    elif args.tokenizer:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        documents = hf_documents(args) if args.hf_dataset else corpus_documents(args.corpus)
        tokenizer.train_from_iterator(itertools.islice(documents, args.tokenizer_samples), trainers.BpeTrainer(
            vocab_size=args.vocab_size, special_tokens=SPECIAL))
    tokenizer.save(str(tokenizer_path))
    manifest = build_chunks(args, out, tokenizer)

    config = dict(vocab_size=tokenizer.get_vocab_size(), **architecture)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if args.resume else None
    if saved:
        if saved["config"] != config:
            raise SystemExit("Use the original architecture and tokenizer when resuming.")
        start = saved["step"]
    else:
        start = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = DistanceAlpha(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not bf16)
    if saved:
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
    actual_parameters = sum(item.numel() for item in model.parameters())
    assert actual_parameters == parameter_count(config)
    print(json.dumps({"parameters": actual_parameters, "preset": args.preset,
                      "train_tokens": manifest["train_tokens"], "validation_tokens": manifest["validation_tokens"],
                      "device": str(device), "chunk_tokens": args.chunk_tokens}))
    train = DiskBlockSampler(out / "token-chunks" / "train", config["block_size"], args.cache_chunks, 42)
    validation = DiskBlockSampler(out / "token-chunks" / "validation", config["block_size"], args.cache_chunks, 123)

    def scheduled_lr(step):
        if step <= args.warmup_steps:
            return args.learning_rate * step / max(1, args.warmup_steps)
        progress = min(1., (step - args.warmup_steps) / max(1, args.schedule_steps - args.warmup_steps))
        cosine = .5 * (1 + math.cos(math.pi * progress))
        return args.learning_rate * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)

    for step in range(start + 1, start + args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr(step)
        x, y = train.batch(args.batch_size, device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, auxiliary, effective = model(x)
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss = ce + args.memory_loss_scale * auxiliary
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer); scaler.update()
        if step % 10 == 0:
            print(f"step={step} loss={ce.item():.4f} lr={scheduled_lr(step):.2e} memory_auxiliary={auxiliary.item():.4f} effective_matches={effective.item():.2f}", flush=True)
        if step % args.save_every == 0 or step == start + args.steps:
            model.eval()
            with torch.inference_mode():
                x, y = validation.batch(args.batch_size, device)
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logits, _, _ = model(x)
                    val = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
            temporary = out / "latest.tmp"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "config": config, "step": step,
                        "validation_loss": val, "training_config": vars(args)}, temporary)
            os.replace(temporary, checkpoint)
            print(f"step={step} sampled_validation_loss={val:.4f} checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
