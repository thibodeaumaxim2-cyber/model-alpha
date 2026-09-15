"""Fine-tune and chat with a small pretrained causal language model."""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "checkpoints" / "alpha-dialogue"
DEFAULT_MODEL = "microsoft/DialoGPT-small"

def format_dialogue(row):
    speakers = ["User" if i % 2 == 0 else "Assistant" for i in range(len(row["dialog"]))]
    return "".join(f"{s}: {t.strip()} <|endoftext|>\n" for s, t in zip(speakers, row["dialog"]))

def make_dataset(tokenizer, limit, max_length):
    data = load_dataset("daily_dialog", split="train")
    if limit: data = data.select(range(min(limit, len(data))))
    data = data.map(lambda row: {"text": format_dialogue(row)}, remove_columns=data.column_names)
    return data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    training = TrainingArguments(output_dir=str(OUTPUT), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accumulation, learning_rate=args.learning_rate, warmup_ratio=.05, logging_steps=10, save_strategy="epoch", report_to="none", fp16=torch.cuda.is_available(), seed=42)
    Trainer(model=model, args=training, train_dataset=make_dataset(tokenizer, args.max_samples, args.max_length), data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), tokenizer=tokenizer).train()
    model.save_pretrained(OUTPUT); tokenizer.save_pretrained(OUTPUT)
    print(f"Saved fine-tuned model to {OUTPUT}")

def chat(args):
    path = Path(args.model)
    if str(path) == DEFAULT_MODEL and (OUTPUT / "config.json").exists(): path = OUTPUT
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    history = []
    print(f"Alpha: ready ({path}). Type /reset or /quit.")
    while True:
        try: message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nAlpha: Goodbye!"); return
        if message.lower() in {"/quit", "/exit"}: print("Alpha: Goodbye!"); return
        if message.lower() == "/reset": history.clear(); print("Alpha: Conversation reset."); continue
        if not message: continue
        history.append(f"User: {message}\nAssistant:")
        prompt = "\n".join(history[-args.max_turns:])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.12, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("\nUser:")[0].split("\nAssistant:")[0].strip() or "I’m not sure how to respond yet."
        print(f"Alpha: {answer}"); history.append(answer)

def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("--model", default=DEFAULT_MODEL); t.add_argument("--max-samples", type=int, default=5000); t.add_argument("--max-length", type=int, default=256); t.add_argument("--epochs", type=float, default=1.0); t.add_argument("--batch-size", type=int, default=2); t.add_argument("--grad-accumulation", type=int, default=8); t.add_argument("--learning-rate", type=float, default=5e-5); t.set_defaults(func=train)
    c = sub.add_parser("chat"); c.add_argument("--model", default=DEFAULT_MODEL); c.add_argument("--max-turns", type=int, default=6); c.add_argument("--max-new-tokens", type=int, default=100); c.add_argument("--temperature", type=float, default=.75); c.add_argument("--top-p", type=float, default=.9); c.set_defaults(func=chat)
    args = p.parse_args(); args.func(args)
if __name__ == "__main__": main()
