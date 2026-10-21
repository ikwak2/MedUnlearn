"""Portable Figure5/6 cases following each method's selected Table3 protocol."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .io import load_module, save, sha, readcsv, writecsv
from .policy import REPO, POLICY, TASKS, engine_dataset, resolve
from .runtime import common_runtime, reference_environment, reference_inputs


def aa_source():
    source=REPO/'third_party/UEraser/ueraser.py'
    if not source.exists():raise ValueError('Run python -m medunlearn fetch-aa before AA experiments')
    return source


def patch(source,old,new):
    if source.count(old)!=1:raise ValueError('Frozen trainer does not match the reviewed AA transform')
    return source.replace(old,new)


def native_code(workspace,family):
    target=workspace/'figure_code/native'/family
    if (target/'snapshot_manifest.json').exists():return target
    shutil.copytree(REPO/'engines/reference'/family,target)
    source=(target/'experiment.py').read_text()
    source=patch(source,'import numpy as np','from aa_native import native_loss\nfrom aa_loss import AA_CONFIG\nimport numpy as np')
    source=patch(source,'            logits = model(x)\n            loss = F.cross_entropy(logits, y)',
                 "            loss, logits = native_loss(model, x, y, epoch, data['normalization'])")
    marker="    if (out/'config.json').exists():"
    source=patch(source,marker,"    config['adversarial_augmentation']=AA_CONFIG\n"+marker)
    (target/'experiment_aa.py').write_text(source)
    if family=='tap':
        source=(target/'tap_experiment.py').read_text()
        source=patch(source,'import numpy as np','from aa_native import native_loss\nfrom aa_loss import AA_CONFIG\nimport numpy as np')
        source=patch(source,'            logits=model(x);loss=F.cross_entropy(logits,y)',
                     '            loss,logits=native_loss(model,x,y,epoch,stats)')
        source=patch(source,marker,"    config['adversarial_augmentation']=AA_CONFIG\n"+marker)
        (target/'tap_experiment_aa.py').write_text(source)
    save(target/'snapshot_manifest.json',{str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()})
    return target


def adapter_code(workspace):
    target=workspace/'figure_code/adapters'
    target.mkdir(parents=True,exist_ok=True)
    for p in (REPO/'engines/figures').glob('*.py'):shutil.copyfile(p,target/p.name)
    for p in (REPO/'engines/common').glob('*.py'):
        if p.name=='benchmark_fixed.py':continue
        shutil.copyfile(p,target/p.name)
    marker="    if (out/'config.json').exists()"
    source=(target/'benchmark.py').read_text()
    source=patch(source,'import numpy as np','from aa_loss import training_loss,AA_CONFIG\nimport numpy as np')
    source=patch(source,'logits=model(x);loss=criterion(logits,y)','loss,logits=training_loss(model,x,y,epoch)')
    source=patch(source,marker,"    config['adversarial_augmentation']=AA_CONFIG\n"+marker)
    (target/'aa_benchmark.py').write_text(source)
    marker=" if (out/'config.json').exists()"
    source=(target/'benchmark_cifar_sgd.py').read_text()
    source=patch(source,'import numpy as np','from aa_loss import training_loss,AA_CONFIG\nimport numpy as np')
    source=patch(source,'logits=model(x);loss=criterion(logits,y)','loss,logits=training_loss(model,x,y,epoch)')
    source=patch(source,marker," config['adversarial_augmentation']=AA_CONFIG\n"+marker)
    (target/'aa_cifar.py').write_text(source)
    source=REPO/'third_party/UEraser/ueraser.py'
    if source.exists():shutil.copyfile(source,target/'official_ueraser.py')
    return target


def run_figure(workspace,dataset,method,mode,ratio=None,python=sys.executable,dry_run=False):
    p=resolve(dataset,method)
    if mode=='ratio' and ratio not in [20,40,60,80]:raise ValueError('Use ratio 20, 40, 60 or 80; 0/100 reuse Table3 controls/results')
    if dry_run:
        print(json.dumps(dict(p,mode=mode,ratio=ratio,training='fresh downstream model',generator='fixed Table3 poison'),indent=2));return
    workspace=Path(workspace).resolve();env=os.environ.copy();env['UE_HOST']='local'
    if mode=='aa':aa_source()
    adapters=adapter_code(workspace)
    if p['scope']=='paper_aligned':
        family=p['family'];reference=workspace/'reference'
        if dataset!='CIFAR-10':reference_inputs(workspace,dataset,python)
        inputs=reference/'inputs'
        if not inputs.exists():inputs.symlink_to(workspace/'inputs',target_is_directory=True)
        native=reference/family/p['task']
        if not (native/'complete.json').exists():raise ValueError('Complete the Table3 run for this method/dataset first')
        code=native_code(workspace,family)
        clean_job=workspace/'figures/native_aa'/f'{family}__{p["task"]}__clean'
        jobs=[(clean_job,'CLEAN'),(workspace/'figures/native_aa'/f'{family}__{p["task"]}__ue',p['condition'])] if mode=='aa' else [
            (workspace/'figures/native_ratio'/f'{family}__{p["task"]}__r{ratio:03d}',p['condition'])]
        env=reference_environment(workspace,family)
        for job,condition in jobs:
            if (job/'COMPLETE.json').exists():continue
            job.mkdir(parents=True,exist_ok=True)
            plan=dict(mode=mode,family=family,method=method,task=p['task'],display_dataset=dataset,
                condition=condition,primary_selection=p['selection'],reference_root=str(reference),
                frozen_code=str(code),gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0'),ratio=ratio,aa_clean_job=str(clean_job))
            save(job/'plan.json',plan)
            files=[f for folder in ['CLEAN',p['condition']] for f in (native/folder).glob('*')
                   if f.suffix in ['.csv','.json']]
            files += [native/'poisons'/(p['condition']+'.npy')]
            save(job/'inputs.json',{str(f):sha(f) for f in files})
            subprocess.run([python,'native_case.py',str(job),'--smoke'],cwd=adapters,env=env,check=True)
            subprocess.run([python,'native_case.py',str(job)],cwd=adapters,env=env,check=True)
        return
    runtime=common_runtime(workspace,dataset);name=engine_dataset(dataset);label=TASKS[dataset] if dataset=='X-ray(fixed)' else name
    baseline=p['clean_baseline'];tree='runs_sgd' if dataset=='CIFAR-10' and baseline=='CLEAN' else 'runs'
    clean=runtime/tree/name/baseline/'seed42';plain=runtime/tree/name/method/'seed42'
    if not (plain/'complete.json').exists():raise ValueError('Complete the corresponding Table3 run first')
    controls=[baseline,method] if mode=='aa' else [method]
    for condition in controls:
        job=workspace/'figures'/('common_aa' if mode=='aa' else 'common_ratio')/'jobs'/(
            f'{label}__{condition}__AA' if mode=='aa' else f'{label}__{method}__r{ratio:03d}')
        if (job/'COMPLETE.json').exists():continue
        job.mkdir(parents=True,exist_ok=True)
        for folder in ['splits_v1','cache','poisons']:
            target=job/folder
            if not target.exists():target.symlink_to(runtime/folder,target_is_directory=True)
        if dataset=='CIFAR-10':
            target=job/'runs_sgd/CIFAR-10/CIFAR-10_train_statistics.json';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(runtime/'runs_sgd/CIFAR-10/CIFAR-10_train_statistics.json',target)
        shutil.copyfile(clean/'config.json',job/'reference_config.json')
        shutil.copyfile(clean/'test_predictions.csv',job/'reference_predictions.csv')
        source=clean if condition in ['CLEAN','CLEAN_AUG'] else plain
        shutil.copyfile(source/'test_predictions.csv',job/'plain_predictions.csv')
        baseline_file=job/tree/name/baseline/'seed42/test_predictions.csv'
        baseline_file.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(clean/'test_predictions.csv',baseline_file)
        plan=dict(dataset=name,display_dataset=dataset,label=label,method=condition,clean_baseline=baseline,
                  generation_protocol='final common protocol',ratio=ratio)
        if mode=='ratio':
            import numpy as np
            module=load_module('nested_masks',REPO/'engines/figures/mixing.py')
            masks,details=module.make_masks([int(r['class_index']) for r in readcsv(runtime/'splits_v1'/name/'train.csv')])
            np.save(job/'mask.npy',masks[ratio]);plan.update(poison_n=details[ratio]['poison_n'],actual_ratio_pct=details[ratio]['actual_percent'])
        save(job/'plan.json',plan)
        # Hash all mutable references and masks. Underlying cache/poison metadata is checked by the engine.
        save(job/'inputs.json',{f.name:sha(f) for f in job.iterdir() if f.is_file() and f.name!='inputs.json'})
        program='from pathlib import Path\nfrom '+('run_case' if mode=='ratio' else 'aa_case')+' import main\nmain(Path('+repr(str(job))+'),smoke=True)\nmain(Path('+repr(str(job))+'))'
        subprocess.run([python,'-c',program],cwd=adapters,env=env,check=True)


def summarize_figures(workspace,output):
    workspace,output=Path(workspace).resolve(),Path(output).resolve();rows=[];children=[]
    for p in (workspace/'figures').rglob('public.json'):
        if not (p.parent/'COMPLETE.json').exists():continue
        value=json.loads(p.read_text());summary=value['summary']
        rows.extend(summary if isinstance(summary,list) else [summary]);children.extend(value['classwise'])
    output.mkdir(parents=True,exist_ok=True)
    writecsv(output/'comparison.csv',rows);writecsv(output/'classwise.csv',children)
    if rows:
        from .summarize import summarize
        base=summarize(workspace)
        plot_figures(output,rows,readcsv(base/'comparison.csv'))
    print('Collected',len(rows),'selection-labelled Figure5/6 results:',output)


def plot_figures(output,rows,main):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    selected=[r for r in rows if r.get('primary',True) not in [False,'False','false','0']]
    base={(r['dataset'],r['method']):r for r in main}
    for metric,label,clean_metric in [('accuracy_pct','Accuracy (%)','clean_accuracy_pct'),
            ('balanced_accuracy_pct','Balanced accuracy (%)','clean_balanced_accuracy_pct'),
            ('macro_f1_pct','Macro-F1 (%)','clean_macro_f1_pct')]:
        fig,axes=plt.subplots(2,2,figsize=(15,10),sharey=True)
        for axis,dataset in zip(axes.flat,POLICY['datasets']):
            for method in POLICY['methods']:
                b=base[dataset,method]
                def number(value):return float(value) if value not in ['',None] else np.nan
                values={0:number(b.get(clean_metric)),100:number(b.get(metric))}
                for r in selected:
                    if (r['dataset'],r['method'])!=(dataset,method):continue
                    if r.get('mode')=='aa' or r.get('aa_protocol'):continue
                    ratio=r.get('target_ratio_pct') or r.get('ratio')
                    if ratio not in [None,'']:values[int(ratio)]=float(r[metric])
                x=[0,20,40,60,80,100]
                axis.plot(x,[values.get(i,np.nan) for i in x],marker='o',markersize=3,label=method)
            axis.set_title(dataset);axis.set_ylim(0,105);axis.set_xlabel('Poisoned training samples (%)');axis.set_ylabel(label)
        axes.flat[-1].legend(fontsize=6,ncol=2)
        fig.suptitle('Figure5 | method-specific main protocols; gaps indicate unfinished conditions')
        fig.tight_layout();stem=metric.replace('_pct','')
        for extension in ['pdf','png']:fig.savefig(output/('figure5_'+stem+'.'+extension),dpi=190,bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(20,12),sharey=True)
        for axis,dataset in zip(axes.flat,POLICY['datasets']):
            plain=[];aa=[];control=[]
            for method in POLICY['methods']:
                p=resolve(dataset,method);b=base[dataset,method]
                plain.append(number(b.get(metric)))
                candidate=next((r for r in selected if r['dataset']==dataset and r['method']==method
                    and (r.get('mode')=='aa' or r.get('aa_protocol')) and r.get('condition')!='CLEAN'),None)
                aa.append(float(candidate[metric]) if candidate else np.nan)
                c=next((r for r in selected if r['dataset']==dataset and
                    ((p['scope']=='paper_aligned' and r.get('family')==p['family'] and r.get('condition')=='CLEAN') or
                     (p['scope']=='common' and r['method']==p['clean_baseline'] and r.get('aa_protocol')))),None)
                control.append(float(c[metric]) if c else np.nan)
            x=np.arange(len(POLICY['methods']))
            axis.bar(x-.25,plain,.25,label='UE');axis.bar(x,aa,.25,label='UE+AA');axis.bar(x+.25,control,.25,label='Matched CLEAN+AA')
            axis.set_xticks(x,POLICY['methods'],rotation=50,ha='right',fontsize=7)
            axis.set_ylim(0,105);axis.set_title(dataset);axis.set_ylabel(label)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('Figure6 | matched native/common protocols; missing bars indicate unfinished conditions')
        fig.tight_layout()
        for extension in ['pdf','png']:fig.savefig(output/('figure6_'+stem+'.'+extension),dpi=190,bbox_inches='tight')
        plt.close(fig)
