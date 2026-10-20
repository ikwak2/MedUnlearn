"""SSIM, generation cost and Figure4 feature extraction under the selected main policy."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import load_module, save, sha
from .policy import REPO, resolve, TASKS, engine_dataset
from .runtime import common_runtime, reference_environment


def native_modules(workspace,p):
    source=REPO/'engines/reference'/p['family'];sys.path.insert(0,str(source))
    for key,value in reference_environment(workspace,p['family']).items():
        if key.startswith(('HYPO_','MEDUNLEARN_','NUMBA_')):os.environ[key]=value
    e=load_module('experiment',source/'experiment.py');sys.modules['experiment']=e
    if p['family'] in ['ar','tap']:e.RECIPE='AR'
    module=load_module('tap_experiment',source/'tap_experiment.py') if p['family']=='tap' else e
    return e,module


def ssim(workspace,dataset,method,output,examples=None):
    import numpy as np
    from skimage.metrics import structural_similarity
    workspace=Path(workspace).resolve();p=resolve(dataset,method)
    if p['scope']=='paper_aligned':
        e,_=native_modules(workspace,p);data=e.load_data(p['task']);clean=data['train_x'];indices=data['train_indices']
        poison=np.load(workspace/'reference'/p['family']/p['task']/'poisons'/(p['condition']+'.npy'),mmap_mode='r')
        quantize=poison.dtype!=np.uint8
    else:
        root=common_runtime(workspace,dataset);name=engine_dataset(dataset)
        clean=np.load(root/'cache'/name/'train.npy',mmap_mode='r');indices=range(len(clean))
        poison=np.load(root/'poisons'/name/(method+'.npy'),mmap_mode='r');quantize=False
    values=[]
    for i in indices:
        original=np.asarray(clean[i],dtype=np.float64)/255
        array=np.asarray(poison[i])
        if quantize:array=(array*255).astype(np.uint8)
        perturbed=np.asarray(array,dtype=np.float64)/(255 if array.dtype==np.uint8 else 1)
        score=structural_similarity(original,perturbed,data_range=1,channel_axis=0,gaussian_weights=True,
                                    sigma=1.5,win_size=11,use_sample_covariance=False)
        values.append(1-float(score))
    save(output,dict(dataset=dataset,method=method,protocol=p['scope'],n=len(values),
        ssim_loss_mean=float(np.mean(values)),ssim_loss_std=float(np.std(values)),
        median=float(np.median(values)),p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95)),
        effective_quantization='Native float poisons truncated through uint8 before SSIM' if quantize else 'Stored effective poison representation',
        split='Actual training indices only; native CIFAR holdout excluded'))
    if examples is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        position=int(np.argmin(np.abs(np.asarray(values)-np.median(values))))
        i=indices[position]
        original=np.asarray(clean[i],dtype=np.float64)/255
        array=np.asarray(poison[i])
        if quantize:array=(array*255).astype(np.uint8)
        perturbed=np.asarray(array,dtype=np.float64)/(255 if array.dtype==np.uint8 else 1)
        panels=[('Clean',original),('Poisoned',perturbed),('Difference ×12; neutral gray=.5',
                  np.clip(.5+12*(perturbed-original),0,1))]
        fig,axes=plt.subplots(1,3,figsize=(11,4))
        for axis,(title,array) in zip(axes,panels):axis.imshow(array.transpose(1,2,0));axis.set_title(title);axis.axis('off')
        fig.suptitle(dataset+' / '+method+' | nearest-to-median training SSIM loss')
        fig.tight_layout();examples=Path(examples);examples.mkdir(parents=True,exist_ok=True)
        fig.savefig(examples/'perturbation.png',dpi=220,bbox_inches='tight')
        fig.savefig(examples/'perturbation.pdf',bbox_inches='tight');plt.close(fig)


def profile(workspace,dataset,method,output):
    import torch
    workspace=Path(workspace).resolve();p=resolve(dataset,method)
    target=workspace/'profiles'/p['task']/method
    if target.exists():raise ValueError('Fresh generation profiling requires a new profile directory')
    target.mkdir(parents=True)
    if p['scope']=='paper_aligned':
        e,module=native_modules(workspace,p);data=e.load_data(p['task'])
        original=workspace/'reference'/p['family']/p['task']
        for name in ['CLEAN','SURROGATE']:
            if (original/name).exists():(target/name).symlink_to(original/name,target_is_directory=True)
        if p['family']=='tap':generation=lambda:module.generate(p['task'],data,target)
        elif p['family']=='ar':generation=lambda:e.generate_ar(p['task'],data,target)
        else:generation=lambda:e.generate(p['task'],p['condition'],data,target)
    else:
        original=common_runtime(workspace,dataset)
        for f in original.glob('*.py'):(target/f.name).symlink_to(f)
        for name in ['splits_v1','cache','vendor','official','runs','runs_sgd']:
            if (original/name).exists():(target/name).symlink_to(original/name,target_is_directory=True)
        sys.path.insert(0,str(target))
        # benchmark.BASE derives from __file__, so load a real code copy at the profile root.
        (target/'benchmark.py').unlink();(target/'benchmark.py').write_bytes((original/'benchmark.py').read_bytes())
        b=load_module('benchmark',target/'benchmark.py');sys.modules['benchmark']=b
        if method=='GUE':
            from .policy import POLICY
            save(target/'assignment.json',dict(dataset=engine_dataset(dataset),generation_batch=POLICY['gue']['generation_batch'][dataset]))
            module=load_module('profile_gue',target/'generate_gue.py');generation=module.main
        else:
            script='generate_simple.py' if method in ['LSPNormal','LSPAUG','HYPO_P3'] else 'generate_advanced.py'
            module=load_module('profile_generator',target/script)
            generation=lambda:module.generate(engine_dataset(dataset),method)
    assert torch.cuda.is_available(),'CUDA required for the recorded generation-cost protocol'
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begun=time.monotonic()
    generation();torch.cuda.synchronize();seconds=time.monotonic()-begun
    save(output,dict(dataset=dataset,method=method,protocol=p['scope'],generation_seconds=seconds,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        clean_pretraining_included=False,downstream_training_included=False,
        scope='Fresh generator invocation; existing required crafting/surrogate checkpoints reused',
        shared_generator='LSPAUG shares LSPNormal' if method=='LSPAUG' else None))


def tsne(workspace,dataset,output):
    if dataset=='CIFAR-10':raise ValueError('Figure4 uses the three medical datasets')
    workspace=Path(workspace).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new feature-extraction output directory')
    output.mkdir(parents=True)
    root=common_runtime(workspace,dataset);name=engine_dataset(dataset)
    files={'test.npy':root/'cache'/name/'test.npy','test.csv':root/'splits_v1'/name/'test.csv',
           'classes.json':root/'splits_v1'/name/'classes.json'}
    for method in ['CLEAN','REM']:
        run=root/'runs'/name/method/'seed42'
        if not (run/'complete.json').exists():raise ValueError('Complete CLEAN and REM before Figure4')
        files.update({method+'.pt':run/'best.pt',method+'_config.json':run/'config.json',
                      method+'_predictions.csv':run/'test_predictions.csv'})
    for target,source in files.items():(output/target).symlink_to(source)
    save(output/'task.json',dict(dataset=name,display_dataset=dataset,
        evaluation_split='complete clean test',selection='maximum clean validation BA',
        patient_independence='verified for X-ray(fixed); unverified for MRI'))
    files['task.json']=output/'task.json'
    save(output/'inputs.json',{target:sha(source) for target,source in files.items()})
    code=REPO/'engines/artifacts'
    program='from pathlib import Path\nfrom tsne import run\nrun(Path('+repr(str(output))+'))'
    subprocess.run([sys.executable,'-c',program],cwd=code,check=True)
