"""EM/REM/SEM original defenders and common untargeted AP generation.
Revised common downstream protocol. All surrogate/generator fitting is train-only.
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from benchmark import BASE,CachedDataset,model_for,seed_all,loader,atomic_json,save_checkpoint

SOURCE=BASE/'vendor/Unlearn2_source'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class CenteredModel(nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x):return self.net(x+.5)

class EOT:
    def __init__(self,h):self.transform=transforms.Compose([transforms.RandomResizedCrop(h),transforms.RandomHorizontalFlip()])
    def __call__(self,x):return torch.stack([self.transform(image) for image in x])/255-.5


def fit_em(dataset,method,ds,k,folder,steps=5000):
    seed_all(42);batch=128 if dataset=='CIFAR-10' else 32
    model=CenteredModel(model_for(dataset,k)).cuda();opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4);criterion=nn.CrossEntropyLoss()
    h=ds.x.shape[-1];trans=EOT(h)
    pgd=module(SOURCE/'REM/attacks/pgd_attacker.py','original_pgd').PGDAttacker
    common=dict(radius=8,steps=10,step_size=1.6,random_start=True)
    if method=='EM':defender=pgd(**common,norm_type='l-infty',ascending=False)
    else:
        name='StablePGDDefender' if method=='SEM' else 'RobustMinimaxPGDDefender'
        path=SOURCE/('SEM/attacks/stable_workers.py' if method=='SEM' else 'REM/attacks/robust_workers.py')
        defender=getattr(module(path,'original_defender'),name)(samp_num=5,trans=trans,**common,atk_radius=4,atk_steps=10,atk_step_size=.8,atk_random_start=True)
    attacker=pgd(radius=4 if method!='EM' else 0,steps=10 if method!='EM' else 0,step_size=.8,random_start=True,norm_type='l-infty',ascending=True)
    ckpath=folder/(method+'_surrogate.pt');start=0
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    dl=loader(ds,batch,True,torch.Generator().manual_seed(42+start));iterator=iter(dl)
    for step in range(start,steps):
        t=time.monotonic()
        try:x,y,_=next(iterator)
        except StopIteration:iterator=iter(dl);x,y,_=next(iterator)
        x=x.cuda();y=y.cuda()
        if method=='EM':poison=defender.perturb(model,criterion,x-.5,y)
        else:
            delta=defender.perturb(model,criterion,x*255,y)
            poison=trans((x*255+(delta*255).round()).clamp(0,255))
        poison=attacker.perturb(model,criterion,poison.detach(),y)
        model.train();opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=.1*(.1**(step//2000))
        loss=criterion(model(poison),y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite surrogate loss')
        loss.backward();opt.step()
        if (step+1)%10==0 or step==start:
            status={'stage':'surrogate','step':step+1,'steps':steps,'loss':loss.item(),'step_seconds':time.monotonic()-t};atomic_json(folder/(method+'_status.json'),status);print(json.dumps(status),flush=True)
        if (step+1)%100==0 or step+1==steps:
            save_checkpoint(ckpath,{'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    return model,defender


def generate_ap(dataset,method,ds,k,folder,dest,steps=250,restarts=8):
    checkpoint=BASE/'runs'/dataset/'CLEAN'/'seed42'/'best.pt';model=model_for(dataset,k).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    eps=8/255;step_size=.05*eps;batch=128 if dataset=='CIFAR-10' else 32
    best=math.inf;candidate=dest.with_suffix('.candidate.npy');bestpath=dest.with_suffix('.best.npy');scores=[]
    progress=folder/(method+'_restarts.json')
    if progress.exists() and bestpath.exists():
        saved=json.loads(progress.read_text());scores=saved['scores'];best=min(scores)
    for restart in range(len(scores),restarts):
        seed_all(42+restart);arr=np.lib.format.open_memmap(candidate,mode='w+',dtype='float32',shape=ds.x.shape);total=0.
        for x,y,ii in loader(ds,batch):
            x=x.cuda();y=y.cuda();target=y
            adv=(x+torch.randn_like(x)*eps).clamp(0,1);adv=x+(adv-x).clamp(-eps,eps)
            for _ in range(steps):
                adv=adv.detach().requires_grad_();loss=F.cross_entropy(model(adv),target)
                loss=-loss
                grad=torch.autograd.grad(loss,adv)[0]
                with torch.no_grad():adv=(x+(adv-step_size*grad.sign()-x).clamp(-eps,eps)).clamp(0,1)
            with torch.no_grad():score=F.cross_entropy(model(adv),target,reduction='sum').item()
            total+=-score
            arr[ii.numpy()]=adv.detach().cpu().numpy()
            print('AP',dataset,method,'restart',restart+1,'samples',int(ii[-1])+1,flush=True)
        arr.flush();del arr;scores.append(total/len(ds))
        if scores[-1]<best:best=scores[-1];os.replace(candidate,bestpath)
        elif candidate.exists():candidate.unlink()
        atomic_json(progress,{'scores':scores,'selection':'lowest total train objective over entire restart; no test data'})
    os.replace(bestpath,dest)
    return {'norm':'Linf','epsilon':eps,'steps':steps,'restarts':restarts,'step_size':step_size,'initialization':'Gaussian eps, projected','surrogate':'validation-selected clean checkpoint; revised from original 40-epoch crafting model','surrogate_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source':'https://github.com/lhfowl/adversarial_poisons','restart_scores':scores}


def generate(dataset,method,smoke=False):
    seed_all(42);ds=CachedDataset(dataset,'train');classes=json.loads((BASE/'splits_v1'/dataset/'classes.json').read_text());k=len(classes)
    folder=BASE/'poisons'/dataset;folder.mkdir(parents=True,exist_ok=True);dest=folder/(method+'.npy');meta=Path(str(dest)+'.json')
    if meta.exists() and dest.exists():print('already generated',dest);return
    partial=dest.with_suffix('.partial.npy')
    if method in ['EM','REM','SEM']:
        model,defender=fit_em(dataset,method,ds,k,folder)
        out=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=ds.x.shape)
        for x,y,ii in loader(ds,128 if dataset=='CIFAR-10' else 32):
            x=x.cuda();y=y.cuda()
            if method=='EM':poison=defender.perturb(model,nn.CrossEntropyLoss(),x-.5,y)+.5
            else:poison=(x+defender.perturb(model,nn.CrossEntropyLoss(),x*255,y)).clamp(0,1)
            out[ii.numpy()]=poison.detach().cpu().numpy();print('final noise',dataset,method,int(ii[-1])+1,flush=True)
        out.flush();del out
        info={'norm':'Linf','epsilon':8/255,'surrogate_steps':5000,'defender_steps':10,'defender_step_size':1.6/255,'EOT_samples':1 if method=='EM' else 5,'inner_radius':0 if method=='EM' else 4/255,'inner_steps':0 if method=='EM' else 10,'source':'repository REM/SEM original defender classes','resume_note':'resumed surrogate uses deterministic fresh train order seeded by saved step; continuation is not bitwise identical to uninterrupted training'}
    elif method=='UT-AP':info=generate_ap(dataset,method,ds,k,folder,partial)
    else:raise ValueError(method)
    arr=np.load(partial,mmap_mode='r');max_linf=0.
    for i in range(0,len(ds),32):
        x=np.asarray(arr[i:i+32]);delta=x-np.asarray(ds.x[i:i+32],dtype=np.float32)/255
        if not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('Invalid poison pixels')
        max_linf=max(max_linf,float(np.abs(delta).max()))
    del arr
    if max_linf>info['epsilon']+1e-6:raise ValueError('Linf budget exceeded')
    info.update(dataset=dataset,method=method,seed=42,test_used=False,observed_max_linf=max_linf,split_sha256=hashlib.sha256((BASE/'splits_v1'/dataset/'train.csv').read_bytes()).hexdigest())
    partial.replace(dest);atomic_json(meta,info);print('GENERATION COMPLETE',dataset,method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',required=True);a=p.parse_args();generate(a.dataset,a.method)
