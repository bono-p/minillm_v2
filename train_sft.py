"""Executable masked-label SFT trainer for French QA."""
from __future__ import annotations
import argparse, os, time, torch, tiktoken
from torch.utils.data import DataLoader
from config import PRESETS
from model import MiniLLM
from sft_data import QASFTDataset, sft_collate

def load_model(path,size,seq,device):
    if path and os.path.exists(path):
        ck=torch.load(path,map_location=device,weights_only=False); cfg=ck['config']; model=MiniLLM(cfg).to(device); model.load_state_dict({k.replace('_orig_mod.',''):v for k,v in ck['model'].items()}); return model,cfg,ck
    cfg=PRESETS[size]; cfg.max_seq_len=seq; return MiniLLM(cfg).to(device),cfg,None

def run(a):
    device=a.device if a.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    model,cfg,old=load_model(a.checkpoint,a.size,a.seq,device); cfg.max_seq_len=a.seq
    enc=tiktoken.get_encoding(a.encoding)
    train=QASFTDataset(a.data,enc,a.seq,'train',a.val_ratio); valid=QASFTDataset(a.data,enc,a.seq,'validation',a.val_ratio)
    tl=DataLoader(train,batch_size=a.batch,shuffle=True,collate_fn=sft_collate); vl=DataLoader(valid,batch_size=a.batch,shuffle=False,collate_fn=sft_collate)
    opt=model.configure_optimizers(a.lr,a.lr*.1,a.weight_decay,.9,.95,device)
    if old and 'optimizer' in old: opt.load_state_dict(old['optimizer'])
    os.makedirs(a.out,exist_ok=True); best=float(old.get('best_val_loss',float('inf'))) if old else float('inf'); best_iter=old.get('best_iter',0) if old else 0; iterator=iter(tl)
    for step in range(a.iters):
        try: inputs,labels=next(iterator)
        except StopIteration: iterator=iter(tl); inputs,labels=next(iterator)
        inputs,labels=inputs.to(device),labels.to(device); opt.zero_grad(set_to_none=True); _,loss=model(inputs,labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip); opt.step()
        if step%a.log_every==0: print(f'sft step={step} loss={loss.item():.4f}')
        if step%a.eval_every==0 or step==a.iters-1:
            model.eval(); losses=[]
            with torch.no_grad():
                for i,(x,y) in enumerate(vl):
                    _,v=model(x.to(device),y.to(device)); losses.append(v.item())
                    if i+1>=a.eval_batches: break
            val=sum(losses)/len(losses); improved=val<best
            if improved: best,best_iter=val,step
            raw=model._orig_mod if hasattr(model,'_orig_mod') else model
            payload={'iter':step,'model':raw.state_dict(),'optimizer':opt.state_dict(),'val_loss':val,'best_val_loss':best,'best_iter':best_iter,'config':cfg,'mode':'sft'}
            torch.save(payload,os.path.join(a.out,'last.pt'))
            if improved: torch.save(payload,os.path.join(a.out,'best.pt'))
            print(f'sft step={step} val_loss={val:.4f} best={best:.4f}'); model.train()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',nargs='+',required=True); p.add_argument('--checkpoint',default=''); p.add_argument('--size',default='15M',choices=list(PRESETS)); p.add_argument('--seq',type=int,default=512); p.add_argument('--batch',type=int,default=2); p.add_argument('--iters',type=int,default=1000); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=.1); p.add_argument('--grad-clip',type=float,default=1.); p.add_argument('--val-ratio',type=float,default=.1); p.add_argument('--eval-every',type=int,default=100); p.add_argument('--eval-batches',type=int,default=20); p.add_argument('--log-every',type=int,default=10); p.add_argument('--out',default='checkpoints/sft'); p.add_argument('--encoding',default='cl100k_base'); p.add_argument('--device',default='auto'); run(p.parse_args())
if __name__=='__main__': main()
