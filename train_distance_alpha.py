"""Local UTF-8 corpus experiment; separate checkpoints from existing Alpha models."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from distance_alpha import DistanceAlpha


PRESETS = {
    "small": dict(block_size=256, dim=256, heads=8, layers=4, clusters=64, slots=64),
    "1b": dict(block_size=2048, dim=2048, heads=16, layers=18, clusters=64, slots=64),
}


def dataset_text(row):
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)
    for field in ("text", "content", "prompt", "conversation"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    return json.dumps(row, ensure_ascii=False)


def parameter_count(config):
    d, v, b, layers = config["dim"], config["vocab_size"], config["block_size"], config["layers"]
    clusters, slots = config["clusters"], config["slots"]
    transformer = layers * (12 * d * d + 13 * d)
    memory = clusters * d * (1 + 2 * slots) + d * d + 2 * d + 2
    return v * d + b * d + transformer + memory + 2 * d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--corpus', type=Path)
    source.add_argument('--hf-dataset', help='Hugging Face dataset ID; streamed to a persistent corpus file.')
    p.add_argument('--hf-config')
    p.add_argument('--hf-split', default='train')
    p.add_argument('--hf-samples', type=int, default=100000)
    p.add_argument('--tokenizer', type=Path, help='Optional existing BPE tokenizer. A new one is trained when omitted.')
    p.add_argument('--output', default='checkpoints/alpha-distance')
    p.add_argument('--steps', type=int, default=1000, help='Additional optimizer steps')
    p.add_argument('--preset', choices=PRESETS, default='small')
    p.add_argument('--block-size', type=int)
    p.add_argument('--dim', type=int)
    p.add_argument('--heads', type=int)
    p.add_argument('--layers', type=int)
    p.add_argument('--clusters', type=int)
    p.add_argument('--slots', type=int)
    p.add_argument('--vocab-size', type=int, default=32000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    architecture = PRESETS[a.preset].copy()
    for key in ("block_size", "dim", "heads", "layers", "clusters", "slots"):
        value = getattr(a, key)
        if value is not None:
            architecture[key] = value
    if min(a.steps, architecture["block_size"], a.batch_size, a.save_every, a.hf_samples) < 1:
        p.error('Numeric arguments must be positive')
    if architecture["dim"] % architecture["heads"]:
        p.error('--dim must be divisible by --heads')
    torch.manual_seed(42)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'latest.pt'
    if checkpoint.exists() and not a.resume:
        raise SystemExit('Checkpoint exists: use --resume or a new --output.')
    if a.resume and not checkpoint.exists():
        raise SystemExit('Resume requested but checkpoint is missing.')
    corpus_path = a.corpus
    if a.hf_dataset:
        corpus_path = out / 'hf-corpus.txt'
        if not corpus_path.exists():
            dataset = load_dataset(a.hf_dataset, a.hf_config, split=a.hf_split, streaming=True)
            dataset = dataset.shuffle(seed=42, buffer_size=10_000)
            with corpus_path.open('x', encoding='utf-8') as corpus:
                for row in dataset.take(a.hf_samples):
                    text = dataset_text(row)
                    if text.strip():
                        corpus.write(text + '\n\n')
    raw = corpus_path.read_text(encoding='utf-8')
    cut = int(len(raw)*.95)
    tokenizer_path = out / 'tokenizer.json'
    if a.resume:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    elif a.tokenizer:
        tokenizer = Tokenizer.from_file(str(a.tokenizer))
    else:
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator([raw[:cut]], trainers.BpeTrainer(
            vocab_size=a.vocab_size, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']))
    # Split raw text before tokenization to keep the validation suffix separate.
    train_ids = torch.tensor(tokenizer.encode(raw[:cut]).ids)
    valid_ids = torch.tensor(tokenizer.encode(raw[cut:]).ids)
    config = dict(vocab_size=tokenizer.get_vocab_size(), **architecture)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if a.resume else None
    if saved:
        config = saved['config']
        if config != dict(vocab_size=tokenizer.get_vocab_size(), **architecture):
            raise SystemExit('Use the original architecture values when resuming.')
    if min(len(train_ids), len(valid_ids)) <= config['block_size']:
        raise SystemExit('Corpus too small for separate training and validation blocks.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = DistanceAlpha(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not bf16)
    start = 0
    if saved:
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); start = saved['step']
    tokenizer.save(str(out/'tokenizer.json'))
    actual_parameters = sum(x.numel() for x in model.parameters())
    assert actual_parameters == parameter_count(config)
    print(json.dumps(dict(parameters=actual_parameters, preset=a.preset,
                          train_tokens=len(train_ids), validation_tokens=len(valid_ids), device=str(device))))

    def batch(ids):
        starts = torch.randint(len(ids)-config['block_size'], (a.batch_size,))
        b = torch.stack([ids[i:i+config['block_size']+1] for i in starts]).to(device)
        return b[:, :-1], b[:, 1:]

    for step in range(start+1, start+a.steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        x, y = batch(train_ids)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
            logits, auxiliary, effective = model(x)
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss = ce + .001 * auxiliary
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer); scaler.update()
        if step % 10 == 0:
            print(f'step={step} loss={ce.item():.4f} effective_matches={effective.item():.2f}', flush=True)
        if step % a.save_every == 0 or step == start+a.steps:
            model.eval()
            with torch.no_grad(), torch.random.fork_rng():
                torch.manual_seed(123)
                x, y = batch(valid_ids)
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    logits, _, _ = model(x)
                    val = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
            temp = out/'latest.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), config=config, step=step,
                            validation_loss=val), temp)
            os.replace(temp, checkpoint)
            print(f'step={step} sampled_validation_loss={val:.4f} checkpoint={checkpoint}', flush=True)


if __name__ == '__main__':
    main()
