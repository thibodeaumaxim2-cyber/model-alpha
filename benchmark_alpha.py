"""Evaluate an Alpha checkpoint against a stable, versioned benchmark."""
import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer, decoders

from train_alpha import AlphaTransformer, RUNS

ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = ROOT / "benchmarks" / "alpha_base_v1.jsonl"


def load_suite(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sequence_nll(model, tokenizer, prompt, target, block_size, device):
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(target).ids
    ids = (prompt_ids + target_ids)[-block_size:]
    prompt_length = min(len(prompt_ids), len(ids))
    if len(ids) < 2 or not target_ids:
        return float("inf"), False
    x = torch.tensor(ids[:-1], device=device)[None]
    y = torch.tensor(ids[1:], device=device)
    with torch.inference_mode():
        logits = model(x)[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    target_positions = range(max(0, prompt_length - 1), len(y))
    losses = [-log_probs[position, y[position]].item() for position in target_positions]
    predicted = [logits[position].argmax().item() for position in target_positions]
    expected = [y[position].item() for position in target_positions]
    return sum(losses) / len(losses), predicted == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS / "latest.pt")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--name", help="Optional checkpoint label in the report")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    tokenizer = Tokenizer.from_file(str(args.checkpoint.parent / "tokenizer.json"))
    tokenizer.decoder = decoders.ByteLevel()
    model = AlphaTransformer(tokenizer.get_vocab_size(), config["block_size"], config["dim"], config["heads"], config["layers"])
    model.load_state_dict(saved["model"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    scores = defaultdict(list); exact = defaultdict(int)
    for item in load_suite(args.suite):
        nll, is_exact = sequence_nll(model, tokenizer, item["prompt"], item["target"], config["block_size"], device)
        scores[item["category"]].append(nll); exact[item["category"]] += is_exact
        print(f"{item['id']}: nll={nll:.3f} exact={is_exact}")
    categories = {name: {"mean_nll": sum(values) / len(values), "perplexity": math.exp(sum(values) / len(values)), "exact_completion_rate": exact[name] / len(values), "examples": len(values)} for name, values in scores.items()}
    report = {"suite": args.suite.name, "checkpoint": args.name or str(args.checkpoint), "created_at": datetime.now(UTC).isoformat(), "training_step": saved.get("step"), "categories": categories}
    output = args.checkpoint.parent / f"benchmark-{args.suite.stem}.json"
    output.write_text(json.dumps(report, indent=2)); print("\n" + json.dumps(report, indent=2)); print(f"Saved report to {output}")


if __name__ == "__main__": main()
