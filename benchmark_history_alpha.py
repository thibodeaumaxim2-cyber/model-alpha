"""Repeatable, append-only performance benchmark for DistanceAlpha."""
import argparse, hashlib, json, platform, statistics, time
from datetime import datetime, UTC
from pathlib import Path
import torch
from tokenizers import Tokenizer
from distance_alpha import DistanceAlpha

def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()

def sha256(path, block=16 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(block), b''): h.update(data)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--history', type=Path, help='JSONL history file; defaults beside checkpoint')
    p.add_argument('--label', default='default')
    p.add_argument('--lengths', default='64,128,256')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--generation-tokens', type=int, default=64)
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0: p.error('runs must be positive and warmup non-negative')
    lengths = [int(x) for x in args.lengths.split(',') if x.strip()]
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']; tokenizer_path = args.checkpoint.parent / 'tokenizer.json'
    if not tokenizer_path.is_file(): p.error(f'missing tokenizer: {tokenizer_path}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistanceAlpha(**config).load_state_dict(saved['model']) if False else DistanceAlpha(**config)
    model.load_state_dict(saved['model']); model.to(device).eval()
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    parameters = sum(x.numel() for x in model.parameters())
    forward = {}
    sanity = True
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    for length in lengths:
        length = min(length, config['block_size'])
        x = torch.randint(config['vocab_size'], (1, length), device=device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
            logits, aux, effective = model(x)
        sanity &= logits.shape == (1, length, config['vocab_size']) and bool(torch.isfinite(logits).all())
        times = []
        for _ in range(args.warmup):
            with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'): model(x)
        sync(device)
        for _ in range(args.runs):
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'): model(x)
            sync(device); times.append(time.perf_counter() - start)
        forward[str(length)] = {'mean_s': statistics.mean(times), 'stdev_s': statistics.stdev(times) if len(times) > 1 else 0, 'min_s': min(times), 'tokens_per_s': length / statistics.mean(times)}
    ids = tokenizer.encode('user: Explain this model.\nassistant:').ids[-config['block_size']:]
    generated = 0; start = time.perf_counter(); eos = tokenizer.token_to_id('<eos>')
    with torch.inference_mode():
        for _ in range(args.generation_tokens):
            x = torch.tensor([ids[-config['block_size']:]], device=device)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'): logits, _, _ = model(x)
            token = logits[0, -1].float().argmax().item()
            if token == eos: break
            ids.append(token); generated += 1
    sync(device); gen_s = time.perf_counter() - start
    record = {'timestamp_utc': datetime.now(UTC).isoformat(), 'label': args.label, 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_bytes': args.checkpoint.stat().st_size, 'step': saved.get('step'), 'parameters': parameters, 'config': config, 'device': str(device), 'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(), 'torch': torch.__version__, 'python': platform.python_version(), 'dtype': str(dtype), 'forward': forward, 'generation': {'tokens': generated, 'seconds': gen_s, 'tokens_per_s': generated / max(gen_s, 1e-9)}, 'effective_memory': effective.item(), 'cuda_peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0, 'sanity_checks_passed': bool(sanity)}
    history = args.history or args.checkpoint.parent / 'benchmark-history.jsonl'; history.parent.mkdir(parents=True, exist_ok=True)
    with history.open('a', encoding='utf-8') as f: f.write(json.dumps(record) + '\n')
    print(json.dumps(record, indent=2)); print(f'history={history}')

if __name__ == '__main__': main()
