"""Run an unchanged frozen downstream trainer with a mixed-training Dataset."""
import argparse
import os
import csv
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from util import save,sha,verify
from mixing import mixed_class


def compare_protocol(actual,reference,sgd):
    keys=['seed','epochs','lr','augmentation','weights','split_sha256']
    keys+=['optimizer','wd','batch','schedule','mean','std','pretrained','weighted'] if sgd else ['optimizer','weight_decay','batch_size','lr_schedule','input','selection']
    for key in keys:assert actual[key]==reference[key],('Training protocol mismatch',key,actual.get(key),reference.get(key))


def main(job,smoke=False):
    verify(job)
    plan=json.loads((job/'plan.json').read_text());dataset,method=plan['dataset'],plan['method']
    mask=np.load(job/'mask.npy',allow_pickle=False)
    assert mask.dtype==bool and int(mask.sum())==plan['poison_n']
    import benchmark
    benchmark.BASE=job
    original=benchmark.CachedDataset
    benchmark.CachedDataset=mixed_class(original,mask)
    sgd=dataset=='CIFAR-10' and method!='LSPAUG'
    reference=json.loads((job/'reference_config.json').read_text())
    assert reference['seed']==42 and reference['weights']=='random initialization'
    assert reference['split_sha256']['train']==sha(job/'splits_v1'/dataset/'train.csv')
    if sgd:
        assert reference['optimizer']=='sgd' and reference['epochs']==200
        assert reference['lr']==.1 and reference['batch']==128 and not reference['weighted'] and not reference['pretrained']
    else:
        assert reference['optimizer']=='Adam' and reference['weight_decay']==0
    assert torch.cuda.device_count()==1
    assert torch.__version__==reference['torch'],'Inspect environment change before training'
    if smoke:
        benchmark.seed_all(42)
        clean=original(dataset,'train');poison=job/'poisons'/dataset/(method+'.npy')
        full=original(dataset,'train',poison)
        mixed=benchmark.CachedDataset(dataset,'train',poison)
        for flag in [False,True]:
            candidates=np.flatnonzero(mask==flag)
            if len(candidates):
                i=int(candidates[0]);a=mixed[i];b=(full if flag else clean)[i]
                assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
        classes=json.loads((job/'splits_v1'/dataset/'classes.json').read_text())
        if sgd:
            cifar=importlib.import_module('benchmark_cifar_sgd')
            model=cifar.make_model(reference,len(classes),reference['mean'],reference['std']).cuda()
        else:model=benchmark.model_for(dataset,len(classes)).cuda()
        batch=min(4,len(mixed))
        x=torch.stack([mixed[i][0] for i in range(batch)]).cuda()
        y=torch.tensor([mixed[i][1] for i in range(batch)],device='cuda')
        loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        save(job/'smoke.json',dict(passed=True,loss=float(loss.detach()),clean_and_poison_rows_match=True,
            train_only_mixing=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        return
    if sgd:
        module=importlib.import_module('benchmark_cifar_sgd')
        sys.argv=['benchmark_cifar_sgd.py','--dataset',dataset,'--method',method]
        module.main()
        run=job/'runs_sgd'/dataset/method/'seed42'
    else:
        benchmark.run(SimpleNamespace(dataset=dataset,method=method,seed=42,
            epochs=reference['epochs'],batch_size=reference['batch_size'],lr=reference['lr']))
        run=job/'runs'/dataset/method/'seed42'
    config=json.loads((run/'config.json').read_text());compare_protocol(config,reference,sgd)
    complete=json.loads((run/'complete.json').read_text());metrics=complete['metrics']
    # Recompute paired metrics from the private saved predictions.
    from classwise_metrics import compare,read_predictions
    clean,paired=compare(read_predictions(job/'reference_predictions.csv'),
        read_predictions(run/'test_predictions.csv'),metrics['classes'])
    for key in ['accuracy','balanced_accuracy','macro_f1']:
        assert abs(paired[key]-metrics[key])<1e-12
    row=dict(dataset=plan['display_dataset'],method=method,target_ratio_pct=plan['ratio'],
        actual_ratio_pct=plan['actual_ratio_pct'],poison_n=plan['poison_n'],train_n=len(mask),
        status='completed',seed=42,mask_seed=42,accuracy_pct=100*paired['accuracy'],
        balanced_accuracy_pct=100*paired['balanced_accuracy'],macro_f1_pct=100*paired['macro_f1'],
        ba_drop_pp=paired['balanced_accuracy_drop_pp'],clean_baseline=plan['clean_baseline'],
        selected_epoch=complete['selected_epoch'],test_n=paired['n'],host=os.environ['UE_HOST'],gpu=0,
        mask_sha256=sha(job/'mask.npy'),input_manifest_sha256=sha(job/'inputs.json'),
        predictions_sha256=sha(run/'test_predictions.csv'),config_sha256=sha(run/'config.json'),
        generation_protocol=plan['generation_protocol'],
        definition='Full-train generator held fixed; perturbations applied only to the fixed selected training images')
    save(job/'public.json',dict(summary=row,classwise=[dict(dataset=plan['display_dataset'],method=method,
        target_ratio_pct=plan['ratio'],class_label=c,test_n=paired['support'][i],
        clean_recall_pct=100*clean['recall'][i],recall_pct=100*paired['recall'][i],
        recall_drop_pp=paired['recall_drop_pp'][i]) for i,c in enumerate(paired['classes'])]))
    save(job/'COMPLETE.json',dict(state='completed',selected_epoch=complete['selected_epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();main(a.job.resolve(),a.smoke)
