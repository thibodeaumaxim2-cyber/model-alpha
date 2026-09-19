"""Scalable 500M Alpha trainer. See README for full-run and smoke-test commands."""
import argparse, json, math, random
from pathlib import Path
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parent; SPECIAL=["<pad>","<unk>","<bos>","<eos>"]
class AlphaTransformer(nn.Module):
    def __init__(self,vocab_size,block_size,dim,heads,layers):
        super().__init__(); self.token=nn.Embedding(vocab_size,dim); self.position=nn.Embedding(block_size,dim); l=nn.TransformerEncoderLayer(dim,heads,4*dim,0,batch_first=True,activation="gelu",norm_first=True); self.blocks=nn.TransformerEncoder(l,layers); self.norm=nn.LayerNorm(dim); self.output=nn.Linear(dim,vocab_size,bias=False); self.output.weight=self.token.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0,.02)
            if getattr(m,'bias',None) is not None: nn.init.zeros_(m.bias)
    def forward(self,t):
        n=t.size(1); h=self.token(t)+self.position(torch.arange(n,device=t.device))[None]; mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=t.device),1); return self.output(self.norm(self.blocks(h,mask=mask)))
def report(model,cfg,path):
    n=sum(p.numel() for p in model.parameters()); r={"total_parameters":n,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"tied_embeddings":model.token.weight.data_ptr()==model.output.weight.data_ptr(),"config":cfg}; Path(path).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2)); return n
def dialogue(r): return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in r['messages'])
def sources(a):
    out=[]
    if a.chat_samples:
        d=load_dataset('HuggingFaceH4/ultrachat_200k',split='train_sft'); d=d.shuffle(seed=a.seed).select(range(min(a.chat_samples,len(d)))); out += [('chat',dialogue(r)) for r in d]
    if a.wikipedia_samples:
        d=load_dataset('wikimedia/wikipedia','20231101.en',split='train'); d=d.shuffle(seed=a.seed+1).select(range(min(a.wikipedia_samples,len(d)))); out += [('wikipedia',r['text']) for r in d]
    if a.code_file: out += [('code',x) for x in Path(a.code_file).read_text(errors='ignore').split('\n\n') if x.strip()]
    if not out: raise SystemExit('Choose at least one data source.')
    if a.smoke_test: out = out * 200
    random.Random(a.seed).shuffle(out); return out
def tokenizer(texts,a,run):
    p=run/'tokenizer.json'
    if p.exists() and a.resume: return Tokenizer.from_file(str(p))
    t=Tokenizer(models.BPE(unk_token='<unk>')); t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); t.decoder=decoders.ByteLevel(); t.train_from_iterator((x for _,x in texts),trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,special_tokens=SPECIAL)); run.mkdir(parents=True,exist_ok=True); t.save(str(p)); return t
def pack(texts,t,b):
    eos=t.token_to_id('<eos>'); s=[]
    for _,x in texts: s.extend(t.encode(x).ids+[eos])
    w=b+1; return torch.tensor([s[i:i+w] for i in range(0,len(s)-w+1,w)],dtype=torch.long),len(s)
def save(p,m,o,sch,step,epoch,cfg): torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'scheduler':sch.state_dict(),'step':step,'epoch':epoch,'config':cfg},p)
@torch.no_grad()
def evaluate(m,loader,loss,dev,amp):
    m.eval(); total=count=0
    for (b,) in loader:
        x,y=b[:,:-1].to(dev,non_blocking=True),b[:,1:].to(dev,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp): z=loss(m(x).flatten(0,1),y.flatten())
        total+=z.item()*len(b); count+=len(b)
    m.train(); return total/max(1,count)
def train(a):
    if not torch.cuda.is_available() and not a.smoke_test: raise SystemExit('CUDA GPU required for the full run.')
    if torch.cuda.is_available(): torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    random.seed(a.seed); torch.manual_seed(a.seed); run=ROOT/'checkpoints'/a.checkpoint_dir; run.mkdir(parents=True,exist_ok=True); txt=sources(a); tok=tokenizer(txt,a,run); blocks,tokens=pack(txt,tok,a.block_size)
    if tokens<a.min_train_tokens: raise SystemExit(f'Only {tokens:,} tokens; need {a.min_train_tokens:,}. Use a several-hundred-million-token corpus or --smoke-test.')
    if len(blocks)<2: raise SystemExit('Not enough packed blocks.')
    order=torch.randperm(len(blocks),generator=torch.Generator().manual_seed(a.seed)); cut=max(1,int(len(blocks)*(1-a.validation_fraction))); tr,va=blocks[order[:cut]],blocks[order[cut:]]; kw=dict(num_workers=a.workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.workers>0); tl=DataLoader(TensorDataset(tr),batch_size=a.micro_batch_size,shuffle=True,drop_last=True,**kw); vl=DataLoader(TensorDataset(va),batch_size=a.micro_batch_size,**kw)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp=dev.type=='cuda'; cfg=vars(a).copy(); m=AlphaTransformer(a.vocab_size,a.block_size,a.dim,a.heads,a.layers).to(dev); n=report(m,cfg,run/'parameter_report.json')
    if a.checkpoint_dir == 'alpha-500m' and not 480_000_000<=n<=530_000_000:
        raise SystemExit(f'Parameter count {n:,} outside requested ~500M range.')
    if a.compile and hasattr(torch,'compile'): m=torch.compile(m)
    o=torch.optim.AdamW(m.parameters(),lr=a.learning_rate,weight_decay=.1,fused=dev.type=='cuda'); updates=max(1,math.ceil(len(tl)/a.grad_accum)*a.epochs); warm=max(1,int(updates*a.warmup_fraction)); sch=torch.optim.lr_scheduler.LambdaLR(o,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,updates-warm)))); step=epoch0=0; latest=run/'latest.pt'
    if a.resume and latest.exists():
        q=torch.load(latest,map_location=dev,weights_only=False); m.load_state_dict(q['model']); o.load_state_dict(q['optimizer']); sch.load_state_dict(q['scheduler']); step=q['step']; epoch0=q.get('epoch',0); print(f'Resumed at step {step}.')
    (run/'run.json').write_text(json.dumps(cfg,indent=2)); print(f'tokens={tokens:,} train_blocks={len(tr):,} validation_blocks={len(va):,} effective_batch={a.micro_batch_size*a.grad_accum}'); loss=nn.CrossEntropyLoss(); o.zero_grad(set_to_none=True)
    for epoch in range(epoch0,a.epochs):
        for micro,(b,) in enumerate(tl):
            x,y=b[:,:-1].to(dev,non_blocking=True),b[:,1:].to(dev,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp): z=loss(m(x).flatten(0,1),y.flatten())/a.grad_accum
            z.backward()
            if (micro+1)%a.grad_accum==0:
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step(); sch.step(); o.zero_grad(set_to_none=True); step+=1
                if step%a.log_every==0: print(f'step={step} train_loss={z.item()*a.grad_accum:.4f} lr={sch.get_last_lr()[0]:.3g}')
                if step%a.save_every==0: print(f'step={step} validation_loss={evaluate(m,vl,loss,dev,amp):.4f}'); save(run/f'step-{step:07d}.pt',m,o,sch,step,epoch,cfg); save(latest,m,o,sch,step,epoch,cfg)
        save(latest,m,o,sch,step,epoch+1,cfg)
    print(f'Finished step={step}, validation_loss={evaluate(m,vl,loss,dev,amp):.4f}')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--chat-samples',type=int,default=0); p.add_argument('--wikipedia-samples',type=int,default=0); p.add_argument('--code-file'); p.add_argument('--vocab-size',type=int,default=32000); p.add_argument('--block-size',type=int,default=2048); p.add_argument('--dim',type=int,default=1536); p.add_argument('--heads',type=int,default=24); p.add_argument('--layers',type=int,default=16); p.add_argument('--micro-batch-size',type=int,default=1); p.add_argument('--grad-accum',type=int,default=32); p.add_argument('--epochs',type=int,default=1); p.add_argument('--learning-rate',type=float,default=1e-4); p.add_argument('--warmup-fraction',type=float,default=.02); p.add_argument('--validation-fraction',type=float,default=.05); p.add_argument('--min-train-tokens',type=int,default=300_000_000); p.add_argument('--log-every',type=int,default=10); p.add_argument('--save-every',type=int,default=500); p.add_argument('--workers',type=int,default=2); p.add_argument('--checkpoint-dir',default='alpha-500m'); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',action='store_true'); p.add_argument('--compile',action='store_true'); p.add_argument('--smoke-test',action='store_true'); train(p.parse_args())
if __name__=='__main__': main()

