"""Train and chat with Alpha: a tiny decoder-only Transformer built from scratch."""
import argparse, re
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "scratch-alpha.pt"
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]

def words(text): return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())
def make_examples(rows):
    out=[]
    for row in rows:
        if row.get("messages"):
            dialogue = [m.get("content", "") for m in row["messages"]]
        else:
            dialogue = row.get("dialog", row.get("dialogue"))
        tokens=["<bos>"]
        for i, turn in enumerate(dialogue): tokens += (["<user>"] if i%2==0 else ["<assistant>"]) + words(turn) + ["<eos>"]
        out.append(tokens)
    return out
def build_vocab(examples, size):
    counts={}
    for e in examples:
        for t in e: counts[t]=counts.get(t,0)+1
    learned=sorted((t for t in counts if t not in SPECIAL), key=lambda t:(-counts[t],t))[:size-len(SPECIAL)]
    return SPECIAL+learned
class DialogueDataset(Dataset):
    def __init__(self, examples, vocab, block):
        lookup={t:i for i,t in enumerate(vocab)}; unk=lookup["<unk>"]; self.items=[]
        for e in examples:
            ids=[lookup.get(t,unk) for t in e]
            for start in range(0,max(1,len(ids)-1),block):
                chunk=ids[start:start+block+1]
                if len(chunk)>1: self.items.append(torch.tensor(chunk))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i][:-1], self.items[i][1:]

def pad_batch(batch):
    """Pad short final chunks so every batch has one rectangular tensor."""
    width = max(x.numel() for x, _ in batch)
    inputs = torch.zeros(len(batch), width, dtype=torch.long)
    targets = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, (x, y) in enumerate(batch):
        inputs[row, :x.numel()] = x
        targets[row, :y.numel()] = y
    return inputs, targets
class TinyGPT(nn.Module):
    def __init__(self,vocab,block,dim,heads,layers):
        super().__init__(); self.block=block; self.tok=nn.Embedding(vocab,dim); self.pos=nn.Embedding(block,dim)
        layer=nn.TransformerEncoderLayer(dim,heads,4*dim,.1,batch_first=True,activation="gelu")
        self.blocks=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,vocab,bias=False); self.head.weight=self.tok.weight
        self.apply(self.init)
    def init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0,.02); nn.init.zeros_(m.bias) if getattr(m,"bias",None) is not None else None
    def forward(self,x):
        n=x.size(1); h=self.tok(x)+self.pos(torch.arange(n,device=x.device))[None,:,:]; mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),1)
        return self.head(self.norm(self.blocks(h,mask=mask)))
def train(a):
    torch.manual_seed(42); raw=load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"); raw=raw.select(range(min(a.max_samples,len(raw)))); examples=make_examples(raw); vocab=build_vocab(examples,a.vocab_size); data=DataLoader(DialogueDataset(examples,vocab,a.block_size),batch_size=a.batch_size,shuffle=True,collate_fn=pad_batch)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=TinyGPT(len(vocab),a.block_size,a.dim,a.heads,a.layers).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.1); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(a.epochs):
        total=0
        for step,(x,y) in enumerate(data,1):
            loss=loss_fn(model(x.to(device)).reshape(-1,len(vocab)),y.to(device).reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()
            if step%100==0: print(f"epoch {epoch+1}/{a.epochs} step {step} loss {total/step:.3f}")
        print(f"epoch {epoch+1} average loss {total/step:.3f}")
    CHECKPOINT.parent.mkdir(exist_ok=True)
    config = {key: value for key, value in vars(a).items() if key != "func"}
    torch.save({"vocab":vocab,"config":config,"model":model.state_dict()},CHECKPOINT)
    print(f"Saved to {CHECKPOINT}")
def chat(a):
    if not CHECKPOINT.exists(): raise SystemExit("Train first: python scratch_chat.py train")
    saved=torch.load(CHECKPOINT,map_location="cpu",weights_only=False); c=saved["config"]; vocab=saved["vocab"]; lookup={t:i for i,t in enumerate(vocab)}; model=TinyGPT(len(vocab),c["block_size"],c["dim"],c["heads"],c["layers"]); model.load_state_dict(saved["model"]); model.eval(); history=[]; print("Alpha: from-scratch model ready. Type /reset or /quit.")
    while True:
        message=input("You: ").strip()
        if message.lower() in {"/quit","/exit"}: print("Alpha: Goodbye!"); return
        if message.lower()=="/reset": history=[]; print("Alpha: Conversation reset."); continue
        if not message: continue
        history += ["<user>"]+words(message)+["<eos>","<assistant>"]; sequence=[lookup["<bos>"]]+[lookup.get(t,lookup["<unk>"]) for t in history]
        generated=[]
        for _ in range(a.max_new_tokens):
            with torch.inference_mode(): logits=model(torch.tensor(sequence[-c["block_size"]:],dtype=torch.long)[None,:])[0,-1].clone()/a.temperature
            logits = logits.clone()
            logits[lookup["<bos>"]]=-float("inf"); logits[lookup["<pad>"]]=-float("inf"); logits[lookup["<assistant>"]]=-float("inf")
            top_values, top_indices = torch.topk(logits, min(a.top_k, logits.numel())); filtered=torch.full_like(logits, -float("inf")); filtered[top_indices]=top_values; logits=filtered
            nxt=torch.multinomial(torch.softmax(logits,-1),1).item(); sequence.append(nxt); generated.append(nxt)
            if vocab[nxt] in {"<eos>","<user>"}: break
        answer=[]
        for i in generated:
            if vocab[i] in {"<eos>","<user>","<assistant>","<bos>","<pad>"}: break
            answer.append(vocab[i])
        print("Alpha:"," ".join(answer) or "I do not know how to answer yet.")
def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True); t=s.add_parser("train"); t.add_argument("--max-samples",type=int,default=30000); t.add_argument("--vocab-size",type=int,default=16000); t.add_argument("--block-size",type=int,default=256); t.add_argument("--dim",type=int,default=256); t.add_argument("--heads",type=int,default=8); t.add_argument("--layers",type=int,default=6); t.add_argument("--batch-size",type=int,default=8); t.add_argument("--epochs",type=int,default=3); t.add_argument("--learning-rate",type=float,default=3e-4); t.set_defaults(func=train); c=s.add_parser("chat"); c.add_argument("--max-new-tokens",type=int,default=100); c.add_argument("--temperature",type=float,default=.75); c.add_argument("--top-k",type=int,default=40); c.set_defaults(func=chat); args=p.parse_args(); args.func(args)
if __name__=="__main__": main()
