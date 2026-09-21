"""Fast interactive chat for a DistanceAlpha checkpoint.

Uses inference mode, CUDA BF16, optional torch.compile, and decodes only once
per response. DistanceAlpha currently recomputes the prefix for each token, so
this is faster and lower-overhead but is not a KV-cache implementation.
"""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from distance_alpha import DistanceAlpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--history-turns', type=int, default=4)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--compile', action='store_true', help='Compile the model on CUDA; first reply is slower.')
    a = p.parse_args()
    if a.temperature <= 0 or a.top_k < 1 or a.max_new_tokens < 1:
        p.error('temperature, top-k and max-new-tokens must be positive')

    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = DistanceAlpha(**config)
    model.load_state_dict(saved['model'])
    del saved
    tokenizer = Tokenizer.from_file(str(a.checkpoint.parent / 'tokenizer.json'))
    if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device).eval()
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    if a.compile and device.type == 'cuda':
        model = torch.compile(model, mode='reduce-overhead')

    banned = [tokenizer.token_to_id(x) for x in ('<pad>', '<bos>', '<unk>')]
    eos = tokenizer.token_to_id('<eos>')
    history = []
    print(f'Alpha fast chat on {device}; compile={a.compile}. /reset clears history; /quit exits.')
    while True:
        try:
            message = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message == '/quit':
            break
        if message == '/reset':
            history.clear()
            print('History cleared.')
            continue
        if not message:
            continue
        history.append('user: ' + message)
        prompt = '\n'.join(history[-2 * a.history_turns:]) + '\nassistant:'
        ids = tokenizer.encode(prompt).ids[-config['block_size']:]
        generated = []
        with torch.inference_mode():
            for _ in range(a.max_new_tokens):
                x = torch.tensor([ids[-config['block_size']:]], device=device)
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    logits, _, _ = model(x)
                scores = logits[0, -1].float().div(a.temperature)
                for idx in banned:
                    if idx is not None:
                        scores[idx] = -float('inf')
                values, indices = scores.topk(min(a.top_k, scores.numel()))
                next_id = indices[torch.multinomial(values.softmax(-1), 1)].item()
                if next_id == eos:
                    break
                ids.append(next_id)
                generated.append(next_id)
                if tokenizer.decode(generated).endswith('\nuser:'):
                    break
        answer = tokenizer.decode(generated).split('\nuser:')[0].strip()
        print('Alpha:', answer)
        history.append('assistant: ' + answer)


if __name__ == '__main__':
    main()
