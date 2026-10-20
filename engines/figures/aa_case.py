"""AA retraining with the common trainer's optimizer, data and selection rule."""
import importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from util import save,sha,verify
from run_case import compare_protocol
from classwise_metrics import compare,read_predictions

CODE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main(job,smoke=False):
    verify(job)
    plan=json.loads((job/'plan.json').read_text());dataset=plan['dataset'];method=plan['method']
    reference=json.loads((job/'reference_config.json').read_text())
    assert reference['seed']==42 and reference['weights']=='random initialization'
    assert torch.__version__==reference['torch'] and torch.cuda.device_count()==1
    assert reference['split_sha256']['train']==sha(job/'splits_v1'/dataset/'train.csv')
    benchmark=load('benchmark',CODE/'aa_benchmark.py');benchmark.BASE=job
    sgd=dataset=='CIFAR-10' and method not in ['CLEAN_AUG','LSPAUG']
    classes=json.loads((job/'splits_v1'/dataset/'classes.json').read_text())
    if smoke:
        from aa_loss import training_loss
        benchmark.seed_all(42)
        poison=None if method in ['CLEAN','CLEAN_AUG'] else job/'poisons'/dataset/(method+'.npy')
        ds=benchmark.CachedDataset(dataset,'train',poison)
        if sgd:
            module=load('aa_cifar',CODE/'aa_cifar.py')
            model=module.make_model(reference,len(classes),reference['mean'],reference['std']).cuda()
        else:model=benchmark.model_for(dataset,len(classes)).cuda()
        batch=min(reference.get('batch',reference.get('batch_size')),len(ds))
        x=torch.stack([ds[i][0] for i in range(batch)]).cuda()
        y=torch.tensor([ds[i][1] for i in range(batch)],device='cuda')
        torch.cuda.reset_peak_memory_stats();loss,logits=training_loss(model,x,y,0)
        loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and logits.shape==(batch,len(classes))
        save(job/'smoke.json',dict(passed=True,batch=batch,repeat=5,loss=float(loss.detach()),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),validation_and_test_unaugmented=True))
        return
    if sgd:
        module=load('aa_cifar',CODE/'aa_cifar.py');sys.argv=['aa_cifar.py','--dataset',dataset,'--method',method];module.main()
        tree='runs_sgd'
    else:
        benchmark.run(SimpleNamespace(dataset=dataset,method=method,seed=42,
            epochs=reference['epochs'],batch_size=reference['batch_size'],lr=reference['lr']))
        tree='runs'
    run=job/tree/dataset/method/'seed42';config=json.loads((run/'config.json').read_text())
    compare_protocol(config,reference,sgd)
    from aa_loss import AA_CONFIG
    assert config['adversarial_augmentation']==AA_CONFIG
    complete=json.loads((run/'complete.json').read_text())
    prediction=read_predictions(run/'test_predictions.csv')
    clean,metrics=compare(read_predictions(job/'reference_predictions.csv'),prediction,classes)
    plain,recovery=compare(read_predictions(job/'plain_predictions.csv'),prediction,classes)
    baseline=plan['clean_baseline']
    control=job.parent/f'{plan["label"]}__{baseline}__AA'
    aa_control=control/tree/dataset/baseline/'seed42/test_predictions.csv'
    assert aa_control.exists(),'AA CLEAN control must finish before UE condition'
    aa_clean,protected=compare(read_predictions(aa_control),prediction,classes)
    for key in ['accuracy','balanced_accuracy','macro_f1']:
        assert abs(metrics[key]-complete['metrics'][key])<1e-12
    summary=dict(dataset=plan['display_dataset'],method=method,status='completed',seed=42,
        accuracy_pct=100*metrics['accuracy'],balanced_accuracy_pct=100*metrics['balanced_accuracy'],
        macro_f1_pct=100*metrics['macro_f1'],plain_accuracy_pct=100*plain['accuracy'],
        plain_balanced_accuracy_pct=100*plain['balanced_accuracy'],plain_macro_f1_pct=100*plain['macro_f1'],
        accuracy_recovery_pp=100*(metrics['accuracy']-plain['accuracy']),
        ba_recovery_pp=100*(metrics['balanced_accuracy']-plain['balanced_accuracy']),
        clean_aa_balanced_accuracy_pct=100*aa_clean['balanced_accuracy'],
        clean_aa_ba_drop_pp=protected['balanced_accuracy_drop_pp'],clean_baseline=baseline,
        selected_epoch=complete['selected_epoch'],epochs=config['epochs'],test_n=metrics['n'],
        host=os.environ.get('MEDUNLEARN_HOST','local'),gpu=0,aa_protocol=AA_CONFIG,generation_protocol=plan['generation_protocol'],
        predictions_sha256=sha(run/'test_predictions.csv'),plain_predictions_sha256=sha(job/'plain_predictions.csv'),
        aa_clean_predictions_sha256=sha(aa_control),input_manifest_sha256=sha(job/'inputs.json'),
        config_sha256=sha(run/'config.json'))
    children=[dict(dataset=plan['display_dataset'],method=method,class_label=c,test_n=metrics['support'][i],
        plain_recall_pct=100*plain['recall'][i],aa_recall_pct=100*metrics['recall'][i],
        recall_recovery_pp=100*(metrics['recall'][i]-plain['recall'][i]),
        clean_aa_recall_pct=100*aa_clean['recall'][i],clean_aa_recall_drop_pp=protected['recall_drop_pp'][i])
        for i,c in enumerate(classes)]
    save(job/'public.json',dict(summary=summary,classwise=children))
    save(job/'COMPLETE.json',dict(state='completed',selected_epoch=complete['selected_epoch']))
