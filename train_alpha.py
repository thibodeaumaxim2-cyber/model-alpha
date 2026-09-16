"""Scalable from-scratch Alpha trainer: BPE, packed data, validation, AMP, resume."""
import argparse
import json
import random
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "checkpoints" / "alpha-v2"
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


class AlphaTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, dim, heads, layers, dropout=.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(block_size, dim)
        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, .02)
            if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
    def forward(self, tokens):
        n = tokens.size(1)
        hidden = self.token(tokens) + self.position(torch.arange(n, device=tokens.device))[None]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), diagonal=1)
        return self.output(self.norm(self.blocks(hidden, mask=mask)))


def dialogue_text(row):
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in row["messages"])


def source_texts(args):
    sources = []
    if args.chat_samples:
        data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        sources += [("chat", dialogue_text(row)) for row in islice(data, args.chat_samples)]
    if args.wikipedia_samples:
        data = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        sources += [("wikipedia", row["text"]) for row in islice(data, args.wikipedia_samples)]
    if args.code_file:
        lines = Path(args.code_file).read_text(errors="ignore").split("\n\n")
        sources += [("code", text) for text in lines if text.strip()]
    if not sources:
        raise SystemExit("Choose at least one data source.")
    random.Random(args.seed).shuffle(sources)
    return sources


def make_tokenizer(texts, args):
    path = RUNS / "tokenizer.json"
    if path.exists() and args.resume:
        return Tokenizer.from_file(str(path))
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=SPECIAL)
    tokenizer.train_from_iterator((text for _, text in texts), trainer=trainer)
    RUNS.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def pack(texts, tokenizer, block_size):
    eos = tokenizer.token_to_id("<eos>")
    stream = []
    for _, text in texts:
        stream.extend(tokenizer.encode(text).ids + [eos])
    width = block_size + 1
    return torch.tensor([stream[i:i + width] for i in range(0, len(stream) - width + 1, width)], dtype=torch.long)


def checkpoint(path, model, optimizer, scaler, step, config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "config": config}, path)


def evaluate(model, loader, loss_fn, device, amp):
    model.eval(); total = count = 0
    with torch.inference_mode():
        for (batch,) in loader:
            x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
            with torch.autocast(device_type=device.type, enabled=amp): loss = loss_fn(model(x).flatten(0, 1), y.flatten())
            total += loss.item() * len(batch); count += len(batch)
    model.train(); return total / max(1, count)


def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed); RUNS.mkdir(parents=True, exist_ok=True)
    texts = source_texts(args); tokenizer = make_tokenizer(texts, args); blocks = pack(texts, tokenizer, args.block_size)
    if len(blocks) < 20: raise SystemExit("Not enough packed blocks; increase data samples.")
    order = torch.randperm(len(blocks)); cut = max(1, int(len(blocks) * (1 - args.validation_fraction)))
    train_data, valid_data = blocks[order[:cut]], blocks[order[cut:]]
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(valid_data), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    model = AlphaTransformer(tokenizer.get_vocab_size(), args.block_size, args.dim, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp); loss_fn = nn.CrossEntropyLoss(); step = 0
    latest = RUNS / "latest.pt"
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); step = saved["step"]; print(f"Resumed at step {step}.")
    config = vars(args).copy(); config.pop("func", None); (RUNS / "run.json").write_text(json.dumps(config, indent=2))
    for epoch in range(args.epochs):
        for (batch,) in train_loader:
            x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp): loss = loss_fn(model(x).flatten(0, 1), y.flatten())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); step += 1
            if step % args.log_every == 0: print(f"step={step} train_loss={loss.item():.4f}")
            if step % args.save_every == 0:
                valid = evaluate(model, valid_loader, loss_fn, device, amp); print(f"step={step} validation_loss={valid:.4f}"); checkpoint(latest, model, optimizer, scaler, step, config)
    valid = evaluate(model, valid_loader, loss_fn, device, amp); checkpoint(latest, model, optimizer, scaler, step, config); print(f"Finished: step={step}, validation_loss={valid:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chat-samples", type=int, default=20000); p.add_argument("--wikipedia-samples", type=int, default=5000); p.add_argument("--code-file", help="Local UTF-8 source corpus; files are separated by blank lines.")
    p.add_argument("--vocab-size", type=int, default=16000); p.add_argument("--block-size", type=int, default=512); p.add_argument("--dim", type=int, default=256); p.add_argument("--heads", type=int, default=8); p.add_argument("--layers", type=int, default=6); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--epochs", type=int, default=3); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--validation-fraction", type=float, default=.05); p.add_argument("--log-every", type=int, default=50); p.add_argument("--save-every", type=int, default=500); p.add_argument("--seed", type=int, default=42); p.add_argument("--resume", action="store_true")
    train(p.parse_args())

if __name__ == "__main__": main()
