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
    """Render the final paper's Figure 5/6 layouts from anonymous metric rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    METHODS=list(POLICY['methods'])
    DATASETS=['CIFAR-10','X-ray(fixed)','MRI4','MRI15']
    LABELS={m:m.replace('HYPO_','HYPO ') for m in METHODS}
    LABELS.update(LSPNormal='LSP Normal',LSPAUG='LSP AUG')
    COLORS={m:plt.get_cmap('tab20')(i) for m,i in zip(METHODS,[0,2,4,6,8,10,12,14,16,18,1,5])}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':16,
        'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12})
    def display(dataset):return 'X-ray' if dataset=='X-ray(fixed)' else dataset
    def save(fig,stem):
        for extension in ['pdf','png']:fig.savefig(output/(stem+'.'+extension),dpi=300,bbox_inches='tight',pad_inches=.07)
        plt.close(fig)
    main={(r['dataset'],r['method']):r for r in main}
    primary=[]
    for raw in rows:
        if raw.get('primary',True) in [False,'False','false','0']:continue
        row=dict(raw)
        row['mode']=row.get('mode') or ('aa' if row.get('aa_protocol') else 'ratio')
        row['protocol_source']=row.get('protocol_source') or ('paper_aligned' if row.get('family') else 'common')
        primary.append(row)
    for metric,name,axislabel in [('accuracy_pct','accuracy','Accuracy (%)'),
                                 ('balanced_accuracy_pct','balanced_accuracy','Balanced accuracy (%)'),
                                 ('macro_f1_pct','macro_f1','Macro-F1 (%)')]:
        fig,axes=plt.subplots(2,2,figsize=(7.4,6.5))
        plotted=[]
        for ax,dataset in zip(axes.flat,DATASETS):
            for method in METHODS:
                row=main[dataset,method]
                values={0:float(row['clean_'+metric]),100:float(row[metric])}
                for r in primary:
                    if (r['dataset'],r['method'],r['mode'])!=(dataset,method,'ratio'):continue
                    ratio=r.get('target_ratio_pct') or r.get('ratio')
                    if ratio not in [None,'']:values[int(ratio)]=float(r[metric])
                x=[0,20,40,60,80,100];y=[values.get(i,np.nan) for i in x]
                ax.plot(x,y,marker='o',markersize=3.8,lw=1.7,color=COLORS[method],label=LABELS[method])
                plotted.append({'dataset':dataset,'method':method,'values':[None if np.isnan(v) else v for v in y]})
            ax.set_title(display(dataset),pad=8)
            ax.set_ylim(-2,103);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([0,20,40,60,80,100])
            ax.tick_params(axis='x',labelsize=11.5);ax.tick_params(axis='y',labelsize=12)
            ax.set_xlabel('Perturbed training images (%)',fontsize=12.5)
            ax.set_ylabel(axislabel,fontsize=13.5)
            ax.grid(axis='y',color='#d6d6d6',lw=.55,alpha=.6)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=11.5,
                   columnspacing=1,handlelength=1.8,handletextpad=.4,bbox_to_anchor=(.5,-.005))
        fig.tight_layout(rect=[0,.14,1,1],h_pad=1.6,w_pad=.8)
        save(fig,'figure5_ratio_'+name)

        fig,axes=plt.subplots(2,2,figsize=(7.6,6.6))
        plotted=[]
        for ax,dataset in zip(axes.flat,DATASETS):
            ue=[];aa=[];clean_aa=[]
            for method in METHODS:
                native=method in ['HYPO_P1','HYPO_P5','AR','T-AP']
                match=next((r for r in primary if r['dataset']==dataset and r['method']==method and r['mode']=='aa' and r.get('condition')!='CLEAN'),None)
                control=next((r for r in primary if r['dataset']==dataset and r['mode']=='aa' and (
                    native and r['method']==method and r.get('condition')=='CLEAN' or
                    not native and r['method']==('CLEAN_AUG' if method=='LSPAUG' else 'CLEAN') and r['protocol_source']=='common')),None)
                ue.append(float(main[dataset,method][metric]));aa.append(float(match[metric]) if match else np.nan)
                clean_aa.append(float(control[metric]) if control else np.nan)
            x=np.arange(len(METHODS))
            ax.bar(x-.26,ue,.26,color='#7c8290',label='UE')
            ax.bar(x,aa,.26,color='#2e7eaf',label='UE + AA')
            ax.bar(x+.26,clean_aa,.26,color='#d59145',label='Matched CLEAN + AA')
            ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=65,ha='right',fontsize=11.2)
            ax.tick_params(axis='x',pad=2);ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100])
            ax.set_title(display(dataset),pad=8);ax.set_ylabel(axislabel,fontsize=13.5)
            plotted.append({'dataset':dataset,'values':[[None if np.isnan(v) else v for v in values] for values in [ue,aa,clean_aa]]})
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=12,
                   columnspacing=1,handlelength=1.2,bbox_to_anchor=(.5,-.005))
        fig.tight_layout(rect=[0,.06,1,1],h_pad=1.6,w_pad=.9)
        save(fig,'figure6_aa_'+name)
