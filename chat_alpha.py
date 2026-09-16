"""Chat with a trained Alpha Base checkpoint."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_alpha import AlphaTransformer, RUNS


def generate(model, tokenizer, prompt, block_size, device, max_tokens, temperature, top_k):
    ids = tokenizer.encode(prompt).ids
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    for _ in range(max_tokens):
        tokens = torch.tensor(ids[-block_size:], dtype=torch.long, device=device)[None]
        with torch.inference_mode():
            logits = model(tokens)[0, -1].clone() / temperature
        logits[pad] = -float("inf")
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        filtered = torch.full_like(logits, -float("inf")); filtered[indices] = values
        next_id = torch.multinomial(torch.softmax(filtered, dim=-1), 1).item()
        ids.append(next_id)
        if next_id == eos: break
    return tokenizer.decode(ids[len(tokenizer.encode(prompt).ids):]).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS / "latest.pt")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=.75)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    tokenizer = Tokenizer.from_file(str(args.checkpoint.parent / "tokenizer.json"))
    model = AlphaTransformer(tokenizer.get_vocab_size(), config["block_size"], config["dim"], config["heads"], config["layers"])
    model.load_state_dict(saved["model"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    history = []
    print("Alpha: ready. Type /reset to clear context or /quit to exit.")
    while True:
        try: message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nAlpha: Goodbye!"); return
        if message.lower() in {"/quit", "/exit"}: print("Alpha: Goodbye!"); return
        if message.lower() == "/reset": history.clear(); print("Alpha: Context cleared."); continue
        if not message: continue
        history.append(f"user: {message}\nassistant:")
        prompt = "\n".join(history[-args.max_turns:])
        answer = generate(model, tokenizer, prompt, config["block_size"], device, args.max_new_tokens, args.temperature, args.top_k)
        answer = answer.split("user:")[0].split("assistant:")[0].strip() or "I am still learning."
        print(f"Alpha: {answer}")
        history.append(answer)


if __name__ == "__main__": main()
