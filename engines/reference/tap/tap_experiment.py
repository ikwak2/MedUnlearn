"""Official-like targeted AP reference: separate 40-epoch crafting network and 200-epoch evaluation."""
import argparse
import ast
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import experiment as e

PROTOCOL = 'tap_official_augmented_reference_v1'
UPSTREAM = 'a1d58d8e31bcb01da494f39ad2b0a707c09acbf4'
evaluation_model = e.module('tap_evaluation_model', e.CODE/'tap_official/poison_evaluation/models/resnet.py')
augmentations = e.module('tap_augmentation', e.CODE/'tap_official/village/materials/diff_data_augmentation.py')


def crafting_model(task, classes):
    if task != 'cifar10':
        return torchvision.models.resnet18(weights=None, num_classes=len(classes))
    # Load the exact upstream class, avoiding unrelated optional network dependencies.
    source = ast.parse((e.CODE/'tap_official/village/clients/models.py').read_text())
    definition = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name=='ResNet')
    namespace = dict(torch=torch, torchvision=torchvision,
                     BasicBlock=torchvision.models.resnet.BasicBlock,
                     Bottleneck=torchvision.models.resnet.Bottleneck)
    exec(compile(ast.Module(body=[definition],type_ignores=[]), '<official TAP ResNet>', 'exec'), namespace)
    return namespace['ResNet'](torchvision.models.resnet.BasicBlock, [2,2,2,2],
                               num_classes=len(classes), base_width=64, initial_conv=[3,1,1])


def downstream_model(task, classes):
    if task == 'cifar10':
        return evaluation_model.ResNet18()
    return torchvision.models.resnet18(weights=None, num_classes=len(classes))


def normalization(data, crafting=False):
    if crafting and data['train_x'].shape[-1] == 32:
        return ([.4914672374725342,.4822617471218109,.4467701315879822],
                [.24703224003314972,.24348513782024384,.26158785820007324])
    return data['normalization']


def transform(size):
    return augmentations.RandomTransform(size,size,shift=8 if size==32 else size//4,fliplr=True)


def attack_step(model, pixels, delta, labels, stats, augmentation, epsilon=8/255):
    mean = torch.as_tensor(stats[0],device=pixels.device,dtype=pixels.dtype)[None,:,None,None]
    std = torch.as_tensor(stats[1],device=pixels.device,dtype=pixels.dtype)[None,:,None,None]
    delta = delta.detach().requires_grad_(True)
    outputs = model(augmentation((pixels+delta-mean)/std))
    loss = F.cross_entropy(outputs, (labels+1) % outputs.shape[1])
    gradient = torch.autograd.grad(loss,delta)[0]
    updated = (delta-(epsilon*.05)*gradient.sign()).clamp(-epsilon,epsilon)
    updated = torch.maximum(torch.minimum(updated,1-pixels),-pixels)
    return updated.detach(), loss.detach()


def train(task, name, data, images, root):
    e.seed_all()
    surrogate = name=='SURROGATE'
    out = root/name
    out.mkdir(parents=True,exist_ok=True)
    epochs = 40 if surrogate else 200
    batch = 128 if task=='cifar10' else 32
    stats = normalization(data,crafting=surrogate)
    config = dict(protocol=PROTOCOL,upstream_commit=UPSTREAM,condition=name,task=task,
                  seed=0,epochs=epochs,batch_size=batch,optimizer='SGD',lr=.1,momentum=.9,
                  weight_decay=.0005,nesterov=surrogate,
                  schedule='MultiStepLR [14,24,35], gamma .1' if surrogate else 'CosineAnnealingLR T_max200',
                  normalization=stats,augmentation='official differentiable RandomTransform' if surrogate else
                  'PIL RandomCrop padding4 + horizontal flip',
                  selection='fixed final; medical clean-val BA best is supplementary',
                  test_use='after training only; no test monitoring or checkpoint selection',
                  model='official TAP crafting ResNet18' if surrogate and task=='cifar10' else
                  'official evaluation ResNet18' if task=='cifar10' else 'torchvision medical ResNet18',
                  torch=torch.__version__,host=os.environ.get('MEDUNLEARN_HOST','local'),gpu_uuid=os.environ.get('ASSIGNED_GPU_UUID'))
    if task!='cifar10':
        config['split_sha256']={s:e.digest(e.INPUTS/task/'splits_v1'/e.TASKS[task]/(s+'.csv'))
                                for s in ['train','validation','test']}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config,'Changed TAP config'
    e.atomic_json(out/'config.json',config)
    if (out/'complete.json').exists():
        return
    dataset = e.Images(images,data['train_y'],data['train_indices'],augment=not surrogate,normalization=stats)
    generator = torch.Generator().manual_seed(0)
    train_loader = e.loader(dataset,batch,True,generator)
    model = (crafting_model(task,data['classes']) if surrogate else downstream_model(task,data['classes'])).cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=surrogate)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[14,24,35],gamma=.1) if surrogate else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200))
    augmentation = transform(images.shape[-1]) if surrogate else None
    start,best,history=0,-1.,[]
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start,best,history=saved['epoch']+1,saved['best'],saved['history']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
    for epoch in range(start,epochs):
        begun=time.monotonic();model.train();total=correct=0;loss_sum=0.
        lr=optimizer.param_groups[0]['lr']
        for x,y,_ in train_loader:
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            if surrogate:x=augmentation(x)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=F.cross_entropy(logits,y)
            assert torch.isfinite(loss),'Nonfinite TAP training loss'
            loss.backward();optimizer.step()
            total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
        metrics={}
        if task!='cifar10' and not surrogate:
            val=e.Images(data['val_x'],data['val_y'],normalization=stats)
            _,metrics=e.predict(model,val,data['classes'],batch)
            if metrics['balanced_accuracy']>best:
                best=metrics['balanced_accuracy']
                e.checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=epoch,config=config))
        row=dict(epoch=epoch+1,train_loss=loss_sum/total,train_accuracy=correct/total,
                 validation_accuracy=metrics.get('accuracy'),
                 validation_balanced_accuracy=metrics.get('balanced_accuracy'),
                 lr=lr,seconds=time.monotonic()-begun)
        history.append(row);scheduler.step()
        e.checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),epoch=epoch,best=best,history=history,
                     python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                     loader_rng=generator.get_state()))
        e.atomic_json(out/'history.json',history)
        e.atomic_json(out/'status.json',dict(row,state='training',epochs=epochs))
        print(task,name,json.dumps(row),flush=True)
    reports={}
    if not surrogate:
        selections=[('fixed_final',out/'last.pt')]
        if task!='cifar10':selections.append(('validation_best',out/'best.pt'))
        test=e.Images(data['test_x'],data['test_y'],normalization=stats)
        for selection,path in selections:
            saved=torch.load(path,map_location='cpu');model.load_state_dict(saved['model'])
            predictions,metrics=e.predict(model,test,data['classes'],batch,data['test_ids'])
            if name=='T-AP':
                _,metrics=e.compare(e.read_predictions(root/'CLEAN'/f'{selection}_predictions.csv'),
                                    predictions,data['classes'])
            e.write_csv(out/f'{selection}_predictions.csv',predictions)
            reports[selection]=dict(metrics=metrics,selected_epoch=saved['epoch']+1)
        e.atomic_json(out/'metrics.json',reports)
    e.atomic_json(out/'complete.json',dict(protocol=PROTOCOL,reports=reports,completed_at=time.time()))
    e.atomic_json(out/'status.json',dict(state='completed',reports=reports))


def generate(task,data,root):
    out=root/'poisons';out.mkdir(exist_ok=True)
    destination=out/'T-AP.npy'
    if destination.exists() and destination.with_suffix('.json').exists():
        assert json.loads(destination.with_suffix('.json').read_text())['sha256']==e.digest(destination)
        return np.load(destination,mmap_mode='r')
    model=crafting_model(task,data['classes']).cuda()
    model.load_state_dict(torch.load(root/'SURROGATE/last.pt',map_location='cpu')['model'])
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    images=data['train_x'];batch=512 if task=='cifar10' else 32
    augmentation=transform(images.shape[-1]);stats=normalization(data,True)
    receipt=out/'restarts.json'
    scores=json.loads(receipt.read_text()) if receipt.exists() else []
    best=min([r['score'] for r in scores],default=float('inf'))
    candidate=out/'candidate.npy';best_path=out/'best_float.npy'
    for restart in range(len(scores),8):
        e.seed_all(restart)
        values=np.lib.format.open_memmap(candidate,mode='w+',dtype='float32',shape=images.shape)
        score=0.;batches=0
        for begin in range(0,len(images),batch):
            original=torch.from_numpy(np.array(images[begin:begin+batch],copy=True)).float()/255
            labels=torch.tensor(data['train_y'][begin:begin+batch],dtype=torch.long,device='cuda')
            delta=(torch.randn(original.shape)*(8/255)).clamp(-8/255,8/255).cuda()
            pixels=original.cuda()
            for step in range(250):
                delta,loss=attack_step(model,pixels,delta,labels,stats,augmentation)
                assert torch.isfinite(loss) and torch.isfinite(delta).all()
            poisoned=(pixels+delta).clamp(0,1)
            assert (poisoned-pixels).abs().max()<=8/255+1e-6
            values[begin:begin+len(original)]=poisoned.cpu().numpy()
            score+=loss.item();batches+=1
            e.atomic_json(out/'status.json',dict(state='generating',restart=restart+1,restarts=8,
                          done=min(begin+batch,len(images)),total=len(images),steps_per_batch=250))
            if begin%(batch*10)==0:print(task,'T-AP restart',restart+1,'samples',begin,flush=True)
        values.flush();del values
        score/=batches
        if score<best:
            best=score;candidate.replace(best_path)
        else:candidate.unlink()
        scores.append(dict(restart=restart+1,seed=restart,score=score))
        e.atomic_json(receipt,scores)
    best_float=np.load(best_path,mmap_mode='r')
    partial=destination.with_suffix('.partial.npy')
    quantized=np.lib.format.open_memmap(partial,mode='w+',dtype='uint8',shape=images.shape)
    maximum=quantized_maximum=0.
    for begin in range(0,len(images),32):
        pixels=np.asarray(images[begin:begin+32],dtype=np.float32)/255
        result=np.asarray(best_float[begin:begin+32])
        maximum=max(maximum,float(np.max(np.abs(result-pixels))))
        q=np.clip(result*255+.5,0,255).astype(np.uint8)
        quantized_maximum=max(quantized_maximum,float(np.max(np.abs(q.astype(np.float32)/255-pixels))))
        quantized[begin:begin+len(q)]=q
    assert maximum<=8/255+1e-6
    quantized.flush();del quantized;partial.replace(destination)
    e.atomic_json(destination.with_suffix('.json'),dict(method='T-AP',protocol=PROTOCOL,
                  upstream_commit=UPSTREAM,epsilon=8/255,norm='Linf',steps=250,restarts=scores,
                  selected_restart=min(scores,key=lambda r:r['score'])['restart'],
                  step_size=.05*(8/255),augmentation='official differentiable RandomTransform',
                  generation_batch=batch,surrogate_epochs=40,surrogate_sha256=e.digest(root/'SURROGATE/last.pt'),
                  numerical_order='batch-major PGD; independent images; seeded restarts 0..7',
                  observed_max_linf=maximum,observed_quantized_max_linf=quantized_maximum,
                  sha256=e.digest(destination),test_used=False,quantization='official PNG-style rounding'))
    return np.load(destination,mmap_mode='r')


def run(task):
    e.RECIPE='AR'  # Shared loader: all 50000 CIFAR images, medical train-only statistics.
    assert torch.cuda.device_count()==1
    root=e.WORK/task;root.mkdir(parents=True,exist_ok=True)
    data=e.load_data(task)
    train(task,'SURROGATE',data,data['train_x'],root)
    poison=generate(task,data,root)
    train(task,'CLEAN',data,data['train_x'],root)
    train(task,'T-AP',data,poison,root)
    e.protocol=lambda:PROTOCOL
    e.export(task,'T-AP',root)
    marker=dict(state='completed',task=task,method='T-AP',protocol=PROTOCOL,completed_at=time.time())
    if task=='cifar10':
        clean=json.loads((root/'CLEAN/metrics.json').read_text())['fixed_final']['metrics']['accuracy']
        ue=json.loads((root/'T-AP/metrics.json').read_text())['fixed_final']['metrics']['accuracy']
        marker.update(reference_gate_passed=clean>=.90 and ue<=.20,clean_accuracy=clean,poison_accuracy=ue,
                      gate_definition='CLEAN>=90%, T-AP<=20%; fixed before first run')
    e.atomic_json(root/'complete.json',marker)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=list(e.TASKS),required=True)
    parser.add_argument('--method',choices=['TAP'],required=True)
    args=parser.parse_args();run(args.task)
