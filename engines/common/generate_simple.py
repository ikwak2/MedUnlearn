"""Adapted AR, LSP and HYPO generation for explicit corrected train manifests.
All generation sees training data only. Norms follow repository defaults:
AR L2=1; LSP L2=(6/255)*sqrt(3HW); HYPO L2=.5.
"""
import argparse, hashlib, importlib.util, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from benchmark import BASE, CachedDataset, model_for, seed_all, loader, atomic_json


def l2project(delta,eps):
    norm=delta.flatten(1).norm(dim=1).view(-1,1,1,1)
    return delta*torch.clamp(eps/(norm+1e-12),max=1.)


def ar_kernel():
    from numba import njit
    @njit
    def recurrence(x,b):
        for c in range(3):
            for i in range(x.shape[1]-2):
                for j in range(x.shape[2]-2):
                    val=0.
                    for u in range(3):
                        for v in range(3): val+=x[c,i+u,j+v]*b[c,u,v]
                    x[c,i+2,j+2]=val
        return x
    return recurrence


def generate(dataset,method,seed=42,smoke=False):
    seed_all(seed);ds=CachedDataset(dataset,'train');classes=json.loads((BASE/'splits_v1'/dataset/'classes.json').read_text());k=len(classes)
    directory=BASE/'poisons'/dataset;directory.mkdir(parents=True,exist_ok=True);dest=directory/(method+'.npy')
    if dest.exists() and Path(str(dest)+'.json').exists(): print('already generated',dest);return
    if method=='LSPAUG':
        generate(dataset,'LSPNormal',seed,smoke=smoke)
        original=directory/'LSPNormal.npy'
        if not dest.exists(): os.link(original,dest)
        info=json.loads(Path(str(original)+'.json').read_text());info['method']=method;atomic_json(str(dest)+'.json',info);return
    n,_,h,w=ds.x.shape; y=np.array([int(r['class_index']) for r in ds.rows]);rng=np.random.RandomState(seed)
    partial=dest.with_suffix('.partial.npy');out=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=ds.x.shape)
    info={'dataset':dataset,'method':method,'seed':seed,'split_sha256':hashlib.sha256((BASE/'splits_v1'/dataset/'train.csv').read_bytes()).hexdigest(),'dtype':'float32','domain':'[0,1]','source_repo':'third_party/Unlearn2','test_used':False,'smoke_test':smoke}
    if method=='LSPNormal':
        from sklearn.datasets import make_classification
        patch=8;side=h//patch;dim=side*side*3
        features,labels=make_classification(n_samples=2*n,n_features=dim,n_classes=k,n_informative=dim,n_redundant=0,n_repeated=0,class_sep=10.,flip_y=0.,n_clusters_per_class=1,weights=np.bincount(y,minlength=k)/n,random_state=seed)
        noise=features.astype(np.float32).reshape(-1,side,side,3)
        noise*=((6/255)*math.sqrt(dim)/np.maximum(np.linalg.norm(noise.reshape(len(noise),-1),axis=1),1e-12))[:,None,None,None]
        buckets=[np.flatnonzero(labels==c) for c in range(k)];used=[0]*k
        info.update(norm='L2',epsilon=(6/255)*math.sqrt(3*h*w),patch=8,correction='Add noise to [0,1] pixels; never multiply original uint8 pixels by 255. Generate enough examples per class for imbalanced datasets.')
        for i,c in enumerate(y):
            if used[c]>=len(buckets[c]): raise ValueError('Insufficient synthetic samples for class')
            delta=np.repeat(np.repeat(noise[buckets[c][used[c]]],patch,0),patch,1).transpose(2,0,1);used[c]+=1
            out[i]=np.clip(np.asarray(ds.x[i],dtype=np.float32)/255+delta,0,1)
    elif method=='AR':
        path=BASE/'vendor/Unlearn2_source/AR'/f'params-classes-{k}-mr-{10 if k==10 else 5}.pt'
        coeff=torch.load(path,map_location='cpu',weights_only=False)
        coeff=np.asarray([np.asarray(b) for b in coeff],dtype=np.float64)
        if coeff.shape!=(k,3,3,3) or not np.allclose(coeff[:,:,2,2],0): raise ValueError('Unexpected AR coefficients')
        recurrence=ar_kernel();crop=4 if dataset=='CIFAR-10' else 0
        info.update(norm='L2',epsilon=1.,coefficients=str(path),coefficients_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop=crop)
        for i,c in enumerate(y):
            signal=recurrence(rng.randn(3,h+crop,w+crop),coeff[c])[:,crop:,crop:]
            norm=np.linalg.norm(signal)
            if not np.isfinite(norm) or norm==0: raise ValueError('Nonfinite/zero AR process')
            out[i]=np.clip(np.asarray(ds.x[i],dtype=np.float32)/255+signal/norm,0,1)
            if i%100==0: print('AR',dataset,i,n,flush=True)
    elif method in ['HYPO_P1','HYPO_P3','HYPO_P5']:
        eps=.5;steps=2 if smoke else 100;step_size=eps/5
        info.update(norm='L2',epsilon=eps,steps=steps,step_size=step_size,target='(true_class+1) % number_of_classes; P3 target universal optimized over entire train pool',correction='Use actual class count instead of hardcoded %10')
        if method=='HYPO_P5':
            universal=torch.randn(k,3,h,w);universal=eps*universal/universal.flatten(1).norm(dim=1).view(-1,1,1,1)
        else:
            checkpoint=BASE/'runs'/dataset/'CLEAN'/'seed42'/'best.pt'
            model=model_for(dataset,k).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
            for p in model.parameters():p.requires_grad_(False)
            info['surrogate_checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if method=='HYPO_P3':
                universal=[];dl=loader(ds,32,True,torch.Generator().manual_seed(seed));iterator=iter(dl)
                for target in range(k):
                    delta=torch.zeros(1,3,h,w,device='cuda')
                    for step in range(steps*5):
                        try:x,_,_=next(iterator)
                        except StopIteration:iterator=iter(dl);x,_,_=next(iterator)
                        x=x.cuda();delta=delta.detach().requires_grad_();loss=torch.nn.functional.cross_entropy(model((x+delta).clamp(0,1)),torch.full((len(x),),target,device='cuda',dtype=torch.long));grad=torch.autograd.grad(loss,delta)[0]
                        delta=l2project(delta-step_size*grad/(grad.norm()+1e-12),eps).detach()
                    universal.append(delta.cpu()[0]);print('P3 target',target,flush=True)
                universal=torch.stack(universal)[list(range(1,k))+[0]]
                info['universal_steps_per_target']=steps*5
        if method!='HYPO_P1':
            np.save(directory/(method+'_universal.npy'),universal.numpy())
        for x,targets,ii in loader(ds,32):
            if method=='HYPO_P1':
                x=x.cuda();target=(targets.cuda()+1)%k;adv=x.clone()
                for _ in range(steps):
                    adv=adv.detach().requires_grad_();loss=torch.nn.functional.cross_entropy(model(adv),target);g=torch.autograd.grad(loss,adv)[0]
                    with torch.no_grad():adv=(x+l2project(adv-step_size*g/(g.flatten(1).norm(dim=1).view(-1,1,1,1)+1e-10)-x,eps)).clamp(0,1)
                values=adv.detach().cpu().numpy()
            else:values=(x+universal[targets]).clamp(0,1).numpy()
            out[ii.numpy()]=values
            if int(ii[0])%320==0:print(method,dataset,int(ii[-1])+1,n,flush=True)
    else:raise ValueError(method)
    # Validate observed perturbation norms in batches, using original resized clean pixels.
    max_l2=max_linf=0.
    for i in range(0,n,32):
        delta=np.asarray(out[i:i+32])-np.asarray(ds.x[i:i+32],dtype=np.float32)/255
        if not np.isfinite(delta).all(): raise ValueError('Nonfinite poison')
        max_l2=max(max_l2,float(np.linalg.norm(delta.reshape(len(delta),-1),axis=1).max()));max_linf=max(max_linf,float(np.max(np.abs(delta))))
    if max_l2>info['epsilon']+1e-4: raise ValueError('Perturbation exceeds L2 budget')
    info.update(observed_max_l2=max_l2,observed_max_linf=max_linf)
    out.flush();del out;partial.replace(dest);atomic_json(str(dest)+'.json',info);print('GENERATION COMPLETE',dataset,method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',required=True);a=p.parse_args();generate(a.dataset,a.method)
