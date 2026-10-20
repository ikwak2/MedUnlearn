"""Corrected benchmark v1. Fixed semantic labels/splits; no test-driven selection."""
import argparse, csv, hashlib, json, os, pickle, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from classwise_metrics import evaluate, compare, write_csv

BASE=Path(__file__).resolve().parent

def atomic_json(path,obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj,indent=2)); temp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)

def model_for(dataset,k):
    model=resnet18(weights=None,num_classes=k)
    if dataset=='CIFAR-10':
        model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False); model.maxpool=nn.Identity()
    return model

def cache_split(dataset,split):
    manifest=BASE/'splits_v1'/dataset/(split+'.csv')
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows=list(csv.DictReader(manifest.open())); size=32 if dataset=='CIFAR-10' else 256
    path=BASE/'cache'/dataset; path.mkdir(parents=True,exist_ok=True)
    dest=path/(split+'.npy'); meta=path/(split+'.json')
    if meta.exists() and dest.exists():
        if json.loads(meta.read_text())['split_sha256']!=digest: raise ValueError('Stale cache')
        return dest,rows
    partial=dest.with_suffix('.partial.npy')
    arr=np.lib.format.open_memmap(partial,mode='w+',dtype='uint8',shape=(len(rows),3,size,size)); batches={}
    for i,r in enumerate(rows):
        if dataset=='CIFAR-10':
            if r['path'] not in batches:
                with open(r['path'],'rb') as f: batches[r['path']]=pickle.load(f,encoding='bytes')[b'data']
            arr[i]=batches[r['path']][int(r['index'])].reshape(3,32,32)
        else:
            with Image.open(r['path']) as im: arr[i]=np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)).transpose(2,0,1)
        if i%1000==0: print('cache',dataset,split,i,len(rows),flush=True)
    arr.flush();del arr;partial.replace(dest)
    atomic_json(meta,{'split_sha256':digest,'shape':[len(rows),3,size,size],'resize':'PIL bilinear RGB','version':1})
    return dest,rows

class CachedDataset(Dataset):
    def __init__(self,dataset,split,poison=None):
        path,self.rows=cache_split(dataset,split);self.x=np.load(path,mmap_mode='r');self.poison=None
        if poison:
            self.poison=np.load(poison,mmap_mode='r')
            if self.poison.shape!=self.x.shape: raise ValueError('Poison shape mismatch')
            info=json.loads(Path(str(poison)+'.json').read_text())
            manifest=BASE/'splits_v1'/dataset/(split+'.csv')
            if info['split_sha256']!=hashlib.sha256(manifest.read_bytes()).hexdigest(): raise ValueError('Poison/split mismatch')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        source=self.poison if self.poison is not None else self.x
        x=torch.from_numpy(np.array(source[i],copy=True)).float()
        if source.dtype==np.uint8: x=x/255
        return x,int(self.rows[i]['class_index']),i

def loader(ds,batch,shuffle=False,generator=None):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=4,pin_memory=True,persistent_workers=True,generator=generator)

def predict(model,dl,classes,device):
    model.eval();rows=[]
    with torch.inference_mode():
        for x,y,ii in dl:
            logits=model(x.to(device,non_blocking=True))
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite evaluation output')
            pred=logits.argmax(1).cpu().tolist()
            for i,t,p in zip(ii.tolist(),y.tolist(),pred):
                rows.append(dict(sample_id=dl.dataset.rows[i]['sample_id'],y_true=classes[t],y_pred=classes[p]))
    return rows,evaluate(rows,classes)

def save_checkpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def run(args):
    seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for scheduled training')
    classes=json.loads((BASE/'splits_v1'/args.dataset/'classes.json').read_text())
    out=BASE/'runs'/args.dataset/args.method/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): print('already complete',out); return
    poison=None if args.method in ['CLEAN','CLEAN_AUG'] else BASE/'poisons'/args.dataset/(args.method+'.npy')
    ds=CachedDataset(args.dataset,'train',poison); val=CachedDataset(args.dataset,'validation')
    generator=torch.Generator().manual_seed(args.seed)
    batch=args.batch_size or (128 if args.dataset=='CIFAR-10' else 32)
    trainloader=loader(ds,batch,True,generator); valloader=loader(val,batch)
    model=model_for(args.dataset,len(classes)).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.CrossEntropyLoss(); start=0; best=-1.; history=[]
    config={'dataset':args.dataset,'method':args.method,'seed':args.seed,'epochs':args.epochs,'batch_size':batch,
            'optimizer':'Adam','lr':args.lr,'weight_decay':0,'lr_schedule':'constant','model':'torchvision ResNet18; CIFAR stem 3x3 stride1/no maxpool',
            'weights':'random initialization','input':'RGB 256x256 medical / native CIFAR 32; float [0,1]',
            'augmentation':'random crop padding4 and horizontal flip' if args.method in ['LSPAUG','CLEAN_AUG'] else 'none',
            'selection':'highest clean validation balanced accuracy; earliest tie','test_use':'only after training',
            'torch':torch.__version__,'split_sha256':{s:hashlib.sha256((BASE/'splits_v1'/args.dataset/(s+'.csv')).read_bytes()).hexdigest() for s in ['train','validation','test']}}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config: raise ValueError('Cannot resume changed config')
    atomic_json(out/'config.json',config)
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck['best'];history=ck['history']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);generator.set_state(ck['loader_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    for epoch in range(start,args.epochs):
        t=time.monotonic();model.train();total=correct=0;loss_sum=0.
        for x,y,_ in trainloader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            if args.method in ['LSPAUG','CLEAN_AUG']:
                from torchvision.transforms import RandomCrop, RandomHorizontalFlip
                x=torch.stack([RandomHorizontalFlip()(RandomCrop(x.shape[-1],padding=4)(image)) for image in x])
            optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
        _,vm=predict(model,valloader,classes,'cuda');elapsed=time.monotonic()-t
        row={'epoch':epoch+1,'train_loss':loss_sum/total,'train_accuracy':correct/total,'validation_accuracy':vm['accuracy'],'validation_balanced_accuracy':vm['balanced_accuracy'],'seconds':elapsed}
        history.append(row)
        if vm['balanced_accuracy']>best:
            best=vm['balanced_accuracy'];save_checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'classes':classes,'config':config})
        save_checkpoint(out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'history':history,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
        atomic_json(out/'history.json',history);atomic_json(out/'status.json',dict(row,state='training',epochs=args.epochs,eta_hours=(args.epochs-epoch-1)*np.mean([h['seconds'] for h in history[-5:]])/3600))
        print(json.dumps(row),flush=True)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model'])
    test=CachedDataset(args.dataset,'test'); pred,metrics=predict(model,loader(test,batch),classes,'cuda')
    write_csv(out/'test_predictions.csv',pred);atomic_json(out/'metrics.json',metrics)
    if args.method not in ['CLEAN','CLEAN_AUG']:
        clean=list(csv.DictReader((BASE/'runs'/args.dataset/('CLEAN_AUG' if args.method=='LSPAUG' else 'CLEAN')/f'seed{args.seed}'/'test_predictions.csv').open()))
        _,metrics=compare(clean,pred,classes);atomic_json(out/'metrics.json',metrics)
    atomic_json(out/'complete.json',{'selected_epoch':selected['epoch']+1,'metrics':metrics,'completed_at':time.time()})
    atomic_json(out/'status.json',{'state':'completed','selected_epoch':selected['epoch']+1,'metrics':metrics})
    print('COMPLETE',args.dataset,args.method,json.dumps(metrics),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',default='CLEAN');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100);p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=0);run(p.parse_args())
