#!/usr/bin/env python3
"""Fine-tune a pretrained JEPA encoder for ADNI classification."""





import argparse
import os
import sys
import csv
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "implementation"))
from neurojepa.utils.init_utils import init_backbone

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("BRAINMRI_ADNI_DATA_ROOT", str(REPO_ROOT / "data/ADNI")))
SRC = REPO_ROOT / "downstream/adni/mae/11_train_adni_classifier.py"
CONFIG_DIR = REPO_ROOT / "configs/pretrain/jepa_archived"
ENC = Path(os.environ.get("BRAINMRI_JEPA_ENCODER_DIR", str(REPO_ROOT / "checkpoints/jepa")))


def args_parse():
 """Parse classification experiment settings."""
 p=argparse.ArgumentParser()
 p.add_argument('--fraction',choices=['12p5','25','50','100'],required=True)
 p.add_argument('--fold',type=int,choices=range(5),required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--smoke',action='store_true')
 p.add_argument('--epochs',type=int,default=25)
 p.add_argument('--batch-size',type=int,default=2)
 p.add_argument('--patience',type=int,default=8)
 p.add_argument('--seed',type=int,default=123)
 p.add_argument('--encoder-lr',type=float,default=5e-6)
 p.add_argument('--head-lr',type=float,default=3e-4)
 p.add_argument('--freeze-epochs',type=int,default=3)
 return p.parse_args()


class Classifier(nn.Module):
 def __init__(self,cfg):
  super().__init__();m=cfg.model;meta=cfg.meta
  self.encoder=init_backbone(device=torch.device('cpu'),model_name=m.backbone_name,
   img_size=list(m.img_size),patch_size=list(m.patch_size),in_chans=int(m.in_chans),
   uniform_power=bool(m.get('uniform_power',False)),use_sdpa=bool(meta.get('use_sdpa',False)),
   use_rope=bool(m.get('use_rope',False)),use_silu=bool(m.get('use_silu',False)),
   wide_silu=bool(m.get('wide_silu',False)),use_activation_checkpointing=False)
  self.head=nn.Linear(768,2)
 def forward(self,x):
  z=self.encoder(x)[0]
  if z.ndim !=3 or z.shape[2]!=768: raise RuntimeError('Unexpected encoder output')
  return self.head(z.mean(1))


def load_data():
 if not SRC.is_file(): raise RuntimeError('Missing original ADNI code')
 spec=importlib.util.spec_from_file_location('adni_original',str(SRC))
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if not all(hasattr(mod,x) for x in ['read_items','make_transforms']):
  raise RuntimeError('ADNI original data/transform APIs differ')
 return mod


def fix_image_paths(items):
 """Resolve scans under the configured ADNI data root."""
 out=[]
 for d in items:
  d=dict(d)
  fixed=0
  for k,v in list(d.items()):
   if isinstance(v,(str,Path)) and '/nifti/' in str(v) and str(v).endswith(('.nii','.nii.gz')):
    v=str(v);rel=v.split('/nifti/',1)[1]
    p=ROOT/'nifti'/rel
    if not p.is_file(): raise FileNotFoundError(p)
    d[k]=str(p);fixed+=1

  for v in d.values():
   if (isinstance(v, str) and v.endswith(('.nii', '.nii.gz')) and Path(v).is_absolute() and not Path(v).resolve().is_relative_to((ROOT / 'nifti').resolve())):
    raise RuntimeError('Unresolved scan path: '+v)
  if fixed<1: raise RuntimeError('No NIfTI path was rewritten: fields '+repr(list(d)))
  out.append(d)
 return out


def metric(model,loader,device):
 """Compute validation AUC from class probabilities."""
 model.eval();labels=[];scores=[]
 with torch.inference_mode():
  for b in loader:
   x=b['image'].float().to(device)
   with torch.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
    yhat=model(x)
   scores.extend(torch.softmax(yhat.float(),dim=1)[:,1].cpu().tolist())
   labels.extend(b['label'].view(-1).cpu().long().tolist())
 if len(set(labels))<2: raise RuntimeError('Validation fold missing one class')
 return float(roc_auc_score(labels,scores)),np.asarray(labels),np.asarray(scores)


def main():
 a=args_parse()
 if not ROOT.is_dir():
  raise FileNotFoundError(ROOT)
 random.seed(a.seed+a.fold);np.random.seed(a.seed+a.fold)
 torch.manual_seed(a.seed+a.fold);torch.cuda.manual_seed_all(a.seed+a.fold)
 torch.set_num_threads(4)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda': raise RuntimeError('Requires GPU')
 mod=load_data()
 train=ROOT/f'splits/adni_ad_cn_fold{a.fold}_train.csv'
 val=ROOT/f'splits/adni_ad_cn_fold{a.fold}_val.csv'
 config_path = CONFIG_DIR / f"jepa_{a.fraction}_paper.yaml"
 for p in (train,val,config_path):
  if not p.is_file(): raise FileNotFoundError(p)
 a_items=fix_image_paths(mod.read_items(train))
 b_items=fix_image_paths(mod.read_items(val))
 if len(a_items)+len(b_items)!=828: raise RuntimeError('Expected 828 ADNI subjects combined, got '+str(len(a_items)+len(b_items)))
 if hasattr(mod,'Dataset'):
  Dataset=mod.Dataset
 else:
  from monai.data import Dataset
 ds_train=Dataset(data=a_items,transform=mod.make_transforms(train=True,image_size=96))
 ds_val=Dataset(data=b_items,transform=mod.make_transforms(train=False,image_size=96))
 tr=DataLoader(ds_train,batch_size=a.batch_size,shuffle=True,num_workers=0,pin_memory=True)
 va=DataLoader(ds_val,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=True)
 m=Classifier(OmegaConf.load(str(config_path)))
 p=ENC/f'neurojepa_fomo{a.fraction}_encoder_e100.pt'
 if not p.is_file(): raise FileNotFoundError(p)
 ck=torch.load(p,map_location='cpu',weights_only=True)
 if ck.get('epoch')!=99:raise RuntimeError('Wrong upstream epoch')
 m.encoder.load_state_dict(ck['model'],strict=True)
 m.to(device)
 print('JEPA encoder loaded',a.fraction,'fold',a.fold,'train/val',len(ds_train),len(ds_val),flush=True)
 ce=nn.CrossEntropyLoss(label_smoothing=0.05)
 scaler=torch.amp.GradScaler('cuda')
 def one_epoch(smoke=False):
  m.train();losses=[]
  for i,b in enumerate(tr):
   if smoke and i>=2:break
   x=b['image'].float().to(device);y=b['label'].view(-1).long().to(device)
   if x.shape[1:]!=(1,96,96,96):raise RuntimeError('Wrong transformed ADNI shape '+str(x.shape))
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):pred=m(x);loss=ce(pred.float(),y)
   if not bool(torch.isfinite(loss)):raise RuntimeError('NaN ADNI loss')
   scaler.scale(loss).backward();scaler.unscale_(opt)
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
   scaler.step(opt);scaler.update();losses.append(float(loss.item()))
  return float(np.mean(losses))
 opt=torch.optim.AdamW([{'params':m.encoder.parameters(),'lr':a.encoder_lr},
                         {'params':m.head.parameters(),'lr':a.head_lr}],weight_decay=0.05)
 if a.smoke:
  for w in m.encoder.parameters():w.requires_grad=False
  loss=one_epoch(smoke=True)
  print('ADNI test batch loss',loss,flush=True);return
 a.out.mkdir(parents=True,exist_ok=True)
 config={'fraction':a.fraction,'fold':a.fold,'seed':a.seed,'epochs':a.epochs,
         'encoder_lr':a.encoder_lr,'head_lr':a.head_lr,'freeze_epochs':a.freeze_epochs,
         'selection':'validation_AUC','model':'Neuro-JEPA native backbone'}
 (a.out/'config.json').write_text(json.dumps(config,indent=2))
 best=-1;bad=0
 for epoch in range(a.epochs):
  t=time.time()
  requires=epoch>=a.freeze_epochs
  for w in m.encoder.parameters():w.requires_grad=requires
  loss=one_epoch();auc,yy,ss=metric(m,va,device)
  if auc>best:
   best=auc;bad=0
   tmp=a.out/'best.tmp.pt'
   torch.save({'model':m.state_dict(),'config':config,'val_auc':auc,'epoch':epoch+1},tmp)
   tmp.replace(a.out/'best.pt')
   with (a.out/'val_predictions.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['label','prob_AD']);writer.writerows(zip(yy,ss))
  else:bad+=1
  print('ADNI',epoch+1,'train_loss',round(loss,4),'val_AUC',round(auc,4),
        'best',round(best,4),'time',round(time.time()-t,1),flush=True)
  if bad>=a.patience:break
 (a.out/'summary.json').write_text(json.dumps({'fraction':a.fraction,'fold':a.fold,'best_val_auc':best},indent=2))
 print('ADNI training finished',best,flush=True)

if __name__=='__main__':main()
