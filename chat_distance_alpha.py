"""Chat with a checkpoint produced by train_distance_alpha.py."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from distance_alpha import DistanceAlpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/alpha-distance-chat/latest.pt'))
    p.add_argument('--max-new-tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-k', type=int, default=40)
    a = p.parse_args()
    if a.temperature <= 0 or a.top_k < 1 or a.max_new_tokens < 1:
        p.error('temperature, top-k and max-new-tokens must be positive')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = DistanceAlpha(**config)
    model.load_state_dict(saved['model'])
    del saved
    tokenizer = Tokenizer.from_file(str(a.checkpoint.parent/'tokenizer.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    history = []
    print(f'Alpha distance ready on {device}. /reset clears history; /quit exits.')
    while True:
        try:
            message = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message == '/quit': break
        if message == '/reset':
            history.clear()
            continue
        if not message: continue
        history.append('user: '+message)
        prompt = '\n'.join(history[-8:])+'\nassistant:'
        ids = tokenizer.encode(prompt).ids
        generated = []
        with torch.inference_mode():
            for _ in range(a.max_new_tokens):
                x = torch.tensor([ids[-config['block_size']:]], device=device)
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    logits, _, _ = model(x)
                scores = logits[0, -1].float() / a.temperature
                for token in ('<pad>', '<bos>', '<unk>'):
                    idx = tokenizer.token_to_id(token)
                    if idx is not None: scores[idx] = -float('inf')
                values, indices = scores.topk(min(a.top_k, scores.numel()))
                next_id = indices[torch.multinomial(values.softmax(-1), 1)].item()
                if next_id == tokenizer.token_to_id('<eos>'): break
                ids.append(next_id); generated.append(next_id)
                answer = tokenizer.decode(generated)
                if '\nuser:' in answer: break
        answer = tokenizer.decode(generated).split('\nuser:')[0].strip()
        print('Alpha:', answer)
        history.append('assistant: '+answer)


if __name__ == '__main__': main()
