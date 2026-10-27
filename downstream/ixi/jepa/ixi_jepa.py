#!/usr/bin/env python3
"""Fine-tune a pretrained JEPA encoder for IXI brain-age regression."""






import argparse
import os
import sys
import copy
import csv
import importlib.util
import inspect
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "implementation"))
from neurojepa.utils.init_utils import init_backbone

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("BRAINMRI_IXI_DATA_ROOT", str(REPO_ROOT / "data/IXI")))
EXPORT = Path(os.environ.get("BRAINMRI_JEPA_ENCODER_DIR", str(REPO_ROOT / "checkpoints/jepa")))
CONFIG_DIR = REPO_ROOT / "configs/pretrain/jepa_archived"
ORIG = REPO_ROOT / "downstream/ixi/mae/train_ixi_brain_age_mae_vitbase_staged.py"


def get_args():
    """Parse the experiment command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('--fraction', choices=['12p5','25','50','100'], required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--warmup-epochs', type=int, default=15)
    p.add_argument('--stage2-epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--encoder-lr', type=float, default=5e-6)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--warmup-lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def original_module():
    """Load the original IXI dataset and preprocessing definitions."""
    if not ORIG.is_file():
        raise RuntimeError('Original IXI script missing: '+str(ORIG))
    spec=importlib.util.spec_from_file_location('ixi_original', str(ORIG))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod,'IXIBrainAgeDataset'):
        raise RuntimeError('IXIBrainAgeDataset missing in original')
    return mod


def clone_original_head(original):

    klass=getattr(original, 'MAEViTBase3DRegressor', None)
    if klass is None:
        return nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.1),nn.Linear(768,1)), 'new_layernorm_dropout_linear'
    sig=inspect.signature(klass)
    known={'img_size':96,'patch_size':16,'embed_dim':768,'depth':12,'num_heads':12,
           'dropout':0.1,'drop':0.1,'mlp_ratio':4.0,'in_chans':1}
    kwargs={k:known[k] for k in known if k in sig.parameters}
    try:

        old=klass(**kwargs)
        for attr in ('head','regression_head','regressor','age_head'):
            head=getattr(old,attr,None)
            if isinstance(head,nn.Module):
                with torch.no_grad():
                    test=head(torch.zeros(2,768))
                if test.numel()==2:
                    return copy.deepcopy(head), 'copied_original_'+attr
        print('WARNING: No compatible named head found, using documented simple head',flush=True)
    except Exception as exc:
        print('WARNING: Could not construct old MAE head: '+repr(exc),flush=True)
    return nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.1),nn.Linear(768,1)), 'new_layernorm_dropout_linear'


class JepaAge(nn.Module):
    def __init__(self, cfg, head):
        super().__init__()
        m=cfg.model
        meta=cfg.meta
        self.encoder=init_backbone(
            device=torch.device('cpu'), model_name=m.backbone_name,
            img_size=list(m.img_size), patch_size=list(m.patch_size),
            in_chans=int(m.in_chans), uniform_power=bool(m.get('uniform_power',False)),
            use_sdpa=bool(meta.get('use_sdpa',False)), use_rope=bool(m.get('use_rope',False)),
            use_silu=bool(m.get('use_silu',False)), wide_silu=bool(m.get('wide_silu',False)),
            use_activation_checkpointing=False)
        self.head=head

    def forward(self,x):
        out=self.encoder(x)
        tokens=out[0] if isinstance(out,(tuple,list)) else out
        if tokens.ndim!=3 or tokens.shape[1:]!=(216,768):
            raise RuntimeError('Bad JEPA tokens '+str(tokens.shape))
        return self.head(tokens.mean(dim=1)).reshape(-1)


def eval_mae(model, loader, device, mean, std):
    model.eval()
    ys=[]; preds=[]
    with torch.inference_mode():
        for b in loader:
            x=b['image'].to(device,non_blocking=True)
            if device.type=='cuda':
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=model(x).float()
            else:
                pred=model(x).float()
            yp=pred.cpu().numpy()*std+mean
            yt=b['age'].numpy()
            ys.extend(yt.tolist()); preds.extend(yp.tolist())
    ys=np.array(ys); preds=np.array(preds)
    if not np.isfinite(preds).all(): raise RuntimeError('Nonfinite predicted ages')
    return float(np.mean(np.abs(ys-preds))),ys,preds


def train_epoch(model, loader, opt, device, scaler, accum, smoke=False):
    """Train the model for one epoch."""
    model.train()
    opt.zero_grad(set_to_none=True)
    totals=[]
    steps=0
    for ix,b in enumerate(loader):
        if smoke and ix>=2: break
        x=b['image'].to(device,non_blocking=True)
        y=b['target'].to(device,non_blocking=True)
        if device.type=='cuda':
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(x).float(); loss=(pred-y).abs().mean()
        else:
            pred=model(x); loss=(pred-y).abs().mean()
        if not bool(torch.isfinite(loss).all()): raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss/accum).backward()
        totals.append(loss.item()); steps+=1
        if steps%accum==0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    if steps%accum:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return float(np.mean(totals))


def main():
    args=get_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(args.seed+args.fold)
    np.random.seed(args.seed+args.fold)
    torch.manual_seed(args.seed+args.fold)
    torch.cuda.manual_seed_all(args.seed+args.fold)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not args.smoke and device.type!='cuda':
        raise RuntimeError('Training requires a CUDA GPU')
    split=BASE/'splits'; original=original_module()
    tr=split/f'folds/fold{args.fold}_train_npy.csv'
    va=split/f'folds/fold{args.fold}_val_npy.csv'
    te=split/'ixi_t1_test_npy_562.csv'
    config_path = CONFIG_DIR / f"jepa_{args.fraction}_paper.yaml"
    for p in (tr,va,te,config_path):
        if not p.exists(): raise FileNotFoundError(p)
    ds_tr=original.IXIBrainAgeDataset(tr,training=True)
    ds_va=original.IXIBrainAgeDataset(va,age_mean=ds_tr.age_mean,age_std=ds_tr.age_std,training=False)
    ds_te=original.IXIBrainAgeDataset(te,age_mean=ds_tr.age_mean,age_std=ds_tr.age_std,training=False)
    if not (len(ds_tr) in (359,360) and len(ds_va) in (89,90) and len(ds_te)==113):
        raise RuntimeError('Incorrect IXI splits')
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    dl_va=DataLoader(ds_va,batch_size=1,num_workers=0)
    dl_te=DataLoader(ds_te,batch_size=1,num_workers=0)
    head,head_type=clone_original_head(original)
    cfg=OmegaConf.load(str(config_path))
    model=JepaAge(cfg,head)
    ckpt=EXPORT/f'neurojepa_fomo{args.fraction}_encoder_e100.pt'
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    pack=torch.load(str(ckpt),map_location='cpu',weights_only=True)
    if pack.get('epoch')!=99: raise RuntimeError('Pretrain epoch not 99')
    model.encoder.load_state_dict(pack['model'],strict=True)
    model.to(device)
    print('JEPA encoder loaded',args.fraction,'fold',args.fold,'head=',head_type,flush=True)
    print('N train,val,test',len(ds_tr),len(ds_va),len(ds_te),flush=True)
    print('head params',sum(p.numel() for p in model.head.parameters()),flush=True)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    if args.smoke:
        for p in model.encoder.parameters(): p.requires_grad=False
        opt=torch.optim.AdamW(model.head.parameters(),lr=args.warmup_lr,weight_decay=args.weight_decay)
        l=train_epoch(model,dl_tr,opt,device,scaler,1,smoke=True)
        print('Head test loss:',l,flush=True)
        for param in model.encoder.parameters():
            param.requires_grad = True
        opt = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
            {'params': model.head.parameters(), 'lr': args.head_lr},
        ], weight_decay=args.weight_decay)
        l = train_epoch(model, dl_tr, opt, device, scaler, 1, smoke=True)
        print('Fine-tuning test loss:',l,flush=True)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={'fraction':args.fraction,'fold':args.fold,'seed':args.seed,
              'head_origin':head_type,'encoder_checkpoint':str(ckpt),
              'train_n':len(ds_tr),'val_n':len(ds_va),'test_n':len(ds_te),
              'warmup_epochs':args.warmup_epochs,'stage2_epochs':args.stage2_epochs,
              'encoder_lr':args.encoder_lr,'head_lr':args.head_lr,'warmup_lr':args.warmup_lr,
              'batch_size':args.batch_size,'grad_accum':args.grad_accum,'patience':args.patience,
              'best_selection':'validation_only','metric':'MAE in years'}
    (args.out/'config.json').write_text(json.dumps(manifest,indent=2))
    hist=[];best=float('inf');best_stage=None;best_epoch=None
    def save_best(stage,epoch,mae):
        nonlocal best,best_stage,best_epoch
        if mae < best:
            best=mae;best_stage=stage;best_epoch=epoch
            tmp=args.out/'best.tmp.pt'
            torch.save({'model':model.state_dict(),'epoch':epoch,'stage':stage,'val_mae':mae,'config':manifest},tmp)
            tmp.replace(args.out/'best.pt')
            return True
        return False
    for p in model.encoder.parameters():p.requires_grad=False
    opt=torch.optim.AdamW(model.head.parameters(),lr=args.warmup_lr,weight_decay=args.weight_decay)
    for epoch in range(args.warmup_epochs):
        start=time.time()
        loss=train_epoch(model,dl_tr,opt,device,scaler,args.grad_accum)
        mae,_,_=eval_mae(model,dl_va,device,ds_tr.age_mean,ds_tr.age_std)
        save_best('warmup',epoch+1,mae)
        hist.append({'stage':'warmup','epoch':epoch+1,'train_loss_norm':loss,'val_mae':mae,'seconds':round(time.time()-start,2)})
        print('WARMUP',epoch+1,'loss',round(loss,4),'val_mae',round(mae,4),'sec',round(time.time()-start,1),flush=True)
    for p in model.encoder.parameters():p.requires_grad=True
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':args.encoder_lr},
                           {'params':model.head.parameters(),'lr':args.head_lr}],weight_decay=args.weight_decay)
    bad=0
    for epoch in range(args.stage2_epochs):
        start=time.time()
        loss=train_epoch(model,dl_tr,opt,device,scaler,args.grad_accum)
        mae,_,_=eval_mae(model,dl_va,device,ds_tr.age_mean,ds_tr.age_std)
        improved=save_best('finetune',epoch+1,mae)
        bad=0 if improved else bad+1
        hist.append({'stage':'finetune','epoch':epoch+1,'train_loss_norm':loss,'val_mae':mae,'seconds':round(time.time()-start,2)})
        print('FINETUNE',epoch+1,'loss',round(loss,4),'val_mae',round(mae,4),'best',round(best,4),'bad',bad,'sec',round(time.time()-start,1),flush=True)
        if bad>=args.patience:
            print('EARLY STOP at',epoch+1,flush=True)
            break
    with (args.out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['stage','epoch','train_loss_norm','val_mae','seconds']);w.writeheader();w.writerows(hist)
    best_ckpt=torch.load(args.out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_ckpt['model'],strict=True)
    test_mae,truth,pred=eval_mae(model,dl_te,device,ds_tr.age_mean,ds_tr.age_std)
    with (args.out/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject_id','age','pred_age'])
        for sid,y,z in zip(ds_te.df['subject_id'].astype(str),truth,pred): w.writerow([sid,y,z])
    summary={'fraction':args.fraction,'fold':args.fold,'val_mae':best,
             'test_mae':test_mae,'best_epoch':best_epoch,'best_stage':best_stage,'head_origin':head_type}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('IXI training finished',json.dumps(summary),flush=True)

if __name__=='__main__': main()
