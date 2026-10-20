"""Official GUE core, corrected train-only manifests, final generator selection."""
import sys,json,time,contextlib,os,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from benchmark import BASE,CachedDataset,atomic_json
sys.path.insert(0,str(BASE/'official'))
import gue
from models import ResNet18,UNet
P=json.loads((BASE/'assignment.json').read_text());D=P['dataset'];B=P['generation_batch'];K=len(json.loads((BASE/'splits_v1'/D/'classes.json').read_text()))
gue.create_net=lambda args:ResNet18(num_classes=K)
args=SimpleNamespace(num_classes=K,eps=8/255,lr_tmp=.0001,eta=1.5)
def main():
 torch.set_num_threads(4);gue.setup_seed(3);random.seed(3)
 ds=CachedDataset(D,'train');folder=BASE/'poisons'/D;folder.mkdir(parents=True,exist_ok=True)
 dest=folder/'GUE.npy';meta=folder/'GUE.npy.json'
 if meta.exists():return
 class Pairs(Dataset):
  def __len__(self):return len(ds)
  def __getitem__(self,i):x,y,_=ds[i];return x,y
 data=Pairs();a=UNet(3).cuda();c=ResNet18(num_classes=K).cuda();ao=torch.optim.SGD(a.parameters(),lr=.1);co=torch.optim.SGD(c.parameters(),lr=.01)
 # One real BOME batch validates memory and gradients. Reset RNG/models afterwards.
 with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
  losses=gue.bome(args,a,c,co,ao,DataLoader(torch.utils.data.Subset(data,range(min(B,len(data)))),batch_size=B),10,0)
 assert np.isfinite(losses).all()
 atomic_json(BASE/'generation_smoke.json',{'passed':True,'batch':B,'peak_memory':torch.cuda.max_memory_allocated(),'torch':torch.__version__})
 del a,c,ao,co;torch.cuda.empty_cache();gue.setup_seed(3);random.seed(3)
 a=UNet(3).cuda();c=ResNet18(num_classes=K).cuda();ao=torch.optim.SGD(a.parameters(),lr=.1);co=torch.optim.SGD(c.parameters(),lr=.01)
 ck=folder/'GUE_generator.pt';start=0
 if ck.exists():
  saved=torch.load(ck,map_location='cpu');a.load_state_dict(saved['attacker']);c.load_state_dict(saved['classifier']);ao.load_state_dict(saved['ao']);co.load_state_dict(saved['co']);start=saved['epoch'];torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
 dl=DataLoader(data,batch_size=B,shuffle=True,pin_memory=True,num_workers=0)
 for epoch in range(start,50):
  t=time.time()
  with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):losses=gue.bome(args,a,c,co,ao,dl,10,epoch+1)
  if not np.isfinite(losses).all():raise RuntimeError('Nonfinite GUE losses')
  # Preserve upstream per-epoch model initialization RNG consumption, but omit test selection.
  scratch=gue.create_net(args).cuda();del scratch
  tmp=ck.with_suffix('.tmp');torch.save({'attacker':a.state_dict(),'classifier':c.state_dict(),'ao':ao.state_dict(),'co':co.state_dict(),'epoch':epoch+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},tmp);tmp.replace(ck)
  state={'epoch':epoch+1,'epochs':50,'seconds':time.time()-t,'losses':losses};atomic_json(folder/'GUE_status.json',state);print(json.dumps(state),flush=True)
 out=np.lib.format.open_memmap(dest.with_suffix('.partial.npy'),mode='w+',dtype='float32',shape=ds.x.shape);a.eval();maxdelta=0.
 with torch.no_grad():
  for x,_,ii in DataLoader(ds,batch_size=B):
   x=x.cuda();z=(x+a(x)*(8/255)).clamp(0,1)
   # Same float poison representation as common benchmark; no reference PIL uint8 conversion.
   maxdelta=max(maxdelta,float((z-x).abs().max()));out[ii.numpy()]=z.cpu().numpy()
 assert maxdelta<=8/255+1e-6
 out.flush();del out;dest.with_suffix('.partial.npy').replace(dest)
 atomic_json(meta,{'generation_protocol':'gue_official_generation_common_eval_v2','dataset':D,'batch_size':B,'epochs':50,'seed':3,'lr_cls':.01,'lr_atk':.1,'lr_tmp':.0001,'inner_iterations':10,'eta':1.5,'epsilon':8/255,'observed_max_linf':maxdelta,'selection':'fixed final epoch; no validation/test access','split_sha256':hashlib.sha256((BASE/'splits_v1'/D/'train.csv').read_bytes()).hexdigest(),'representation':'float32 common benchmark; no uint8 quantization','upstream_commit':'d23ca8e15de2b1ff4e5301004d1b2522bfae2842','torch':torch.__version__})
if __name__=='__main__':main()
