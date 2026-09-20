"""Local UTF-8 corpus experiment; separate checkpoints from existing Alpha models."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from distance_alpha import DistanceAlpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', default='checkpoints/alpha-distance')
    p.add_argument('--steps', type=int, default=1000, help='Additional optimizer steps')
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.block_size, a.batch_size, a.save_every) < 1:
        p.error('Numeric arguments must be positive')
    torch.manual_seed(42)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'latest.pt'
    if checkpoint.exists() and not a.resume:
        raise SystemExit('Checkpoint exists: use --resume or a new --output.')
    if a.resume and not checkpoint.exists():
        raise SystemExit('Resume requested but checkpoint is missing.')
    tokenizer = Tokenizer.from_file(str(out/'tokenizer.json') if a.resume else a.tokenizer)
    raw = Path(a.corpus).read_text(encoding='utf-8')
    cut = int(len(raw)*.95)
    # Split raw text before tokenization to keep the validation suffix separate.
    train_ids = torch.tensor(tokenizer.encode(raw[:cut]).ids)
    valid_ids = torch.tensor(tokenizer.encode(raw[cut:]).ids)
    config = dict(vocab_size=tokenizer.get_vocab_size(), block_size=a.block_size)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if a.resume else None
    if saved:
        config = saved['config']
        if config['block_size'] != a.block_size:
            raise SystemExit('Use the original --block-size when resuming.')
    if min(len(train_ids), len(valid_ids)) <= a.block_size:
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
    print(json.dumps(dict(parameters=sum(x.numel() for x in model.parameters()),
                          train_tokens=len(train_ids), validation_tokens=len(valid_ids), device=str(device))))

    def batch(ids):
        starts = torch.randint(len(ids)-a.block_size, (a.batch_size,))
        b = torch.stack([ids[i:i+a.block_size+1] for i in starts]).to(device)
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
