"""Final common CIFAR-10 SGD protocol. Model selection uses validation only.
The test loader is constructed only in evaluate_winner, after every candidate is complete.
"""
import argparse,csv,fcntl,hashlib,json,math,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.transforms import InterpolationMode
from benchmark import BASE,CachedDataset,model_for,seed_all,predict,save_checkpoint,atomic_json
from classwise_metrics import write_csv

SEARCH=BASE/'runs_sgd'/'CIFAR-10'
CANDIDATES=[dict(id='cifar_sgd_cosine',dataset='CIFAR-10',pretrained=False,weighted=False,optimizer='sgd',lr=.1,wd=.0005,epochs=200,batch=128)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def training_stats(ds,dataset):
 p=SEARCH/(dataset+'_train_statistics.json')
 digest=sha(BASE/'splits_v1'/dataset/'train.csv')
 if p.exists():
  data=json.loads(p.read_text());assert data['train_manifest_sha256']==digest;return data
 sums=np.zeros(3,dtype=np.float64);squares=np.zeros(3,dtype=np.float64);pixels=0
 for start in range(0,len(ds),64):
  a=np.asarray(ds.x[start:start+64],dtype=np.float64)/255
  sums+=a.sum(axis=(0,2,3));squares+=(a*a).sum(axis=(0,2,3));pixels+=a.shape[0]*a.shape[2]*a.shape[3]
 mean=sums/pixels;std=np.sqrt(np.maximum(squares/pixels-mean**2,1e-12))
 data={'train_manifest_sha256':digest,'mean':mean.tolist(),'std':std.tolist(),'pixel_count_per_channel':pixels};atomic_json(p,data);return data

class NormalizedResNet(nn.Module):
 def __init__(self,net,mean,std):
  super().__init__();self.net=net
  self.register_buffer('mean',torch.tensor(mean).view(1,3,1,1));self.register_buffer('std',torch.tensor(std).view(1,3,1,1))
 def forward(self,x):return self.net((x-self.mean)/self.std)

class AugmentedDataset(Dataset):
 def __init__(self,base,dataset):
  self.base=base;self.rows=base.rows
  if dataset=='CIFAR-10':self.transform=transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),transforms.RandomHorizontalFlip()])
  else:self.transform=transforms.RandomAffine(degrees=5,translate=(.02,.02),interpolation=InterpolationMode.BILINEAR,fill=0)
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x,y,idx=self.base[i];return self.transform(x),y,idx

def seed_worker(_):
 seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def make_loader(ds,batch,shuffle=False,generator=None):
 # Workers restart each epoch so a restored generator fully specifies augmentation seeds.
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=4,pin_memory=True,worker_init_fn=seed_worker,generator=generator)

def lr_factor(epoch,total):
 if epoch<5:return (epoch+1)/5
 return .5*(1+math.cos(math.pi*(epoch-5)/(total-5)))

def make_model(spec,k,mean,std,download=False):
 if spec['pretrained']:
  net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if download else None);net.fc=nn.Linear(net.fc.in_features,k)
 else:net=model_for(spec['dataset'],k)
 return NormalizedResNet(net,mean,std)

def class_weights(labels,k):
 counts=np.bincount(labels,minlength=k)
 if (counts==0).any():raise ValueError('Empty training class')
 w=np.sqrt(len(labels)/(k*counts));return torch.tensor(w/w.mean(),dtype=torch.float32)

def train_candidate(name):
 spec=next(dict(c) for c in CANDIDATES if c['id']==name);seed_all(42)
 out=SEARCH/name/'seed42';out.mkdir(parents=True,exist_ok=True)
 if (out/'training_complete.json').exists():return
 dataset=spec['dataset'];classes=json.loads((BASE/'splits_v1'/dataset/'classes.json').read_text());k=len(classes)
 train=CachedDataset(dataset,'train',None if name=='CLEAN' else BASE/'poisons'/dataset/(name+'.npy'));val=CachedDataset(dataset,'validation')
 stats=training_stats(train,dataset)
 if spec['pretrained']:
  transform=ResNet18_Weights.IMAGENET1K_V1.transforms();mean,std=transform.mean,transform.std
 else:mean,std=stats['mean'],stats['std']
 config={**spec,'seed':42,'mean':mean,'std':std,'selection':'maximum validation BA; earliest epoch on tie','schedule':'5 epoch linear warmup + cosine','augmentation':'none',
  'weights':'ImageNet1K V1, all layers fine-tuned' if spec['pretrained'] else 'random initialization','class_weighting':'sqrt inverse train frequency normalized to mean1' if spec['weighted'] else 'none','split_sha256':{s:sha(BASE/'splits_v1'/dataset/(s+'.csv')) for s in ['train','validation','test']},'torch':torch.__version__,'test_access_during_search':False}
 if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Changed config on resume')
 atomic_json(out/'config.json',config)
 model=make_model(spec,k,mean,std,download=not (out/'last.pt').exists()).cuda()
 if spec['optimizer']=='sgd':opt=torch.optim.SGD(model.parameters(),lr=spec['lr'],momentum=.9,weight_decay=spec['wd'],nesterov=True)
 else:opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['wd'])
 labels=[int(r['class_index']) for r in train.rows]
 weights=class_weights(labels,k).cuda() if spec['weighted'] else None
 criterion=nn.CrossEntropyLoss(weight=weights)
 generator=torch.Generator().manual_seed(42);valgen=torch.Generator().manual_seed(4242)
 trainloader=make_loader(train,spec['batch'],True,generator);valloader=make_loader(val,spec['batch'],False,valgen)
 history=[];start=0;best=-1
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];start=ck['epoch']+1;best=ck['best'];generator.set_state(ck['loader_rng']);valgen.set_state(ck['val_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
 for epoch in range(start,spec['epochs']):
  t=time.monotonic();model.train();total=correct=0;loss_sum=0.
  lr=spec['lr']*lr_factor(epoch,spec['epochs'])
  for group in opt.param_groups:group['lr']=lr
  for x,y,_ in trainloader:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();opt.step();total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
  pred,vm=predict(model,valloader,classes,'cuda')
  record={'epoch':epoch+1,'lr':lr,'train_accuracy':correct/total,'train_batch_weighted_loss_mean':loss_sum/total,'validation_accuracy':vm['accuracy'],'validation_balanced_accuracy':vm['balanced_accuracy'],'validation_macro_f1':vm['macro_f1'],'seconds':time.monotonic()-t}
  history.append(record)
  if vm['balanced_accuracy']>best:
   best=vm['balanced_accuracy'];save_checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':epoch+1,'config':config,'classes':classes});atomic_json(out/'validation_metrics.json',vm);write_csv(out/'validation_predictions.csv',pred)
  save_checkpoint(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,'best':best,'loader_rng':generator.get_state(),'val_rng':valgen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
  atomic_json(out/'history.json',history);atomic_json(out/'status.json',{**record,'state':'training','epochs':spec['epochs'],'eta_hours':(spec['epochs']-epoch-1)*np.mean([r['seconds'] for r in history[-5:]])/3600});print(json.dumps(record),flush=True)
 selected=max(history,key=lambda r:r['validation_balanced_accuracy'])
 atomic_json(out/'training_complete.json',{'candidate':name,'config':config,'selected':selected,'completed_at':time.time(),'test_evaluated':False});atomic_json(out/'status.json',{'state':'completed','selected':selected})


def main():
 global CANDIDATES
 from classwise_metrics import compare
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',required=True);a=p.parse_args()
 assert a.dataset=='CIFAR-10' and a.method not in ['CLEAN_AUG','LSPAUG']
 CANDIDATES=[dict(next(c for c in CANDIDATES if c['id']=='cifar_sgd_cosine'),id=a.method)]
 out=SEARCH/a.method/'seed42'
 if (out/'complete.json').exists():return
 train_candidate(a.method)
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);config=ck['config'];classes=ck['classes']
 model=make_model(config,len(classes),config['mean'],config['std']).cuda();model.load_state_dict(ck['model'])
 pred,metrics=predict(model,make_loader(CachedDataset(a.dataset,'test'),128),classes,'cuda')
 write_csv(out/'test_predictions.csv',pred)
 if a.method!='CLEAN':
  clean=list(csv.DictReader((SEARCH/'CLEAN'/'seed42'/'test_predictions.csv').open()))
  _,metrics=compare(clean,pred,classes)
 atomic_json(out/'metrics.json',metrics)
 result={'selected_epoch':ck['epoch'],'metrics':metrics,'completed_at':time.time(),'protocol':'cifar_sgd_v2','generation_protocol':'unchanged v1; downstream optimizer and normalization changed'}
 atomic_json(out/'complete.json',result);atomic_json(out/'status.json',dict(result,state='completed'))
if __name__=='__main__':main()
