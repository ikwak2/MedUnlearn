"""Stage portable, immutable runtime inputs and invoke the final experimental engines."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .io import load_module, save, sha
from .policy import REPO, POLICY, TASKS, engine_dataset, resolve


def copy_immutable(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target): raise ValueError('Runtime code changed; use a new workspace: ' + str(target))
    else: shutil.copyfile(source, target)


def common_runtime(workspace, dataset):
    workspace = Path(workspace).resolve()
    data = json.loads((workspace / 'data_config.json').read_text())
    name = engine_dataset(dataset)
    manifest_source = Path(data['manifest_root']) / name
    if not manifest_source.is_dir():
        raise FileNotFoundError('Prepared manifests are missing; run prepare or import-manifests: ' + str(manifest_source))
    root = workspace / 'common' / TASKS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    for source in (REPO / 'engines/common').glob('*.py'):
        if source.name == 'benchmark_fixed.py': continue
        if source.name == 'benchmark.py' and dataset == 'X-ray(fixed)':
            source = source.with_name('benchmark_fixed.py')
        target_name = 'benchmark.py' if source.name == 'benchmark_fixed.py' else source.name
        copy_immutable(source, root / target_name)
    split = root / 'splits_v1' / name
    split.parent.mkdir(exist_ok=True)
    if not split.exists() and not split.is_symlink(): split.symlink_to(manifest_source, target_is_directory=True)
    if dataset == 'X-ray(fixed)':
        audit = Path(data['manifest_root']) / 'xray_fixed_audit.json'
        if not audit.exists(): raise ValueError('The fixed 12-class split audit is required')
        copy_immutable(audit, root / 'splits_v1/COMPLETE.json')
    vendor = root / 'vendor/Unlearn2_source'
    vendor.parent.mkdir(exist_ok=True)
    if not vendor.exists(): vendor.symlink_to(REPO / 'third_party/Unlearn2', target_is_directory=True)
    official = root / 'official'
    if not official.exists(): official.symlink_to(REPO / 'third_party/GUE', target_is_directory=True)
    save(root / 'source_manifest.json', {p.name:sha(p) for p in root.glob('*.py')})
    return root


def cache(workspace, dataset, python=sys.executable):
    root = common_runtime(workspace, dataset)
    name = engine_dataset(dataset)
    program = 'from benchmark import cache_split\nfor split in ["train","validation","test"]:cache_split(' + repr(name) + ',split)'
    subprocess.run([python, '-c', program], cwd=root, check=True)
    return root


def reference_inputs(workspace, dataset, python=sys.executable):
    workspace = Path(workspace).resolve()
    common = cache(workspace, dataset, python)
    target = workspace / 'inputs' / TASKS[dataset]
    target.mkdir(parents=True, exist_ok=True)
    for directory in ['splits_v1', 'cache']:
        dest = target / directory
        if not dest.exists(): dest.symlink_to(common / directory, target_is_directory=True)
    return target


def reference_environment(workspace, family):
    workspace = Path(workspace).resolve()
    data = json.loads((workspace / 'data_config.json').read_text())
    env = os.environ.copy()
    env.update(HYPO_WORK=str(workspace / 'reference' / family),
               HYPO_INPUTS=str(workspace / 'inputs'),
               MEDUNLEARN_CIFAR_ROOT=data['data_root'], PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR=str(workspace / 'numba_cache'))
    return env


def common_train(root, dataset, method, python):
    sgd = dataset == 'CIFAR-10' and method not in ['LSPAUG','CLEAN_AUG']
    script = 'benchmark_cifar_sgd.py' if sgd else 'benchmark.py'
    cmd = [python, script, '--dataset', engine_dataset(dataset), '--method', method]
    if not sgd:
        cmd += ['--epochs','100','--batch-size','128' if dataset == 'CIFAR-10' else '32',
                '--lr','0.001','--seed','42']
        if dataset == 'X-ray(fixed)': cmd += ['--class-weighting','none']
    subprocess.run(cmd, cwd=root, check=True)


def run(workspace, dataset, method, python=sys.executable, generation_python=None, dry_run=False):
    selected = resolve(dataset, method)
    if dry_run:
        print(json.dumps(dict(selected, workspace=str(Path(workspace).resolve()), python=python,
                              generation_python=generation_python or python),indent=2)); return
    workspace = Path(workspace).resolve()
    generation_python = generation_python or python
    if selected['scope'] == 'paper_aligned':
        if dataset != 'CIFAR-10': reference_inputs(workspace, dataset, python)
        family = selected['family']
        source = REPO / 'engines/reference' / family
        script = 'tap_experiment.py' if family == 'tap' else 'experiment.py'
        command = [python, str(source / script), '--task', selected['task'], '--method',
                   'TAP' if family == 'tap' else selected['condition']]
        subprocess.run(command, cwd=source, env=reference_environment(workspace,family), check=True)
        return
    root = cache(workspace, dataset, python)
    common_train(root, dataset, selected['clean_baseline'], python)
    # The original common CIFAR AP/P3 crafting model used the Adam CLEAN,
    # distinct from the SGD downstream CLEAN. Preserve that surrogate.
    if dataset == 'CIFAR-10' and method in ['UT-AP','HYPO_P3']:
        subprocess.run([python,'benchmark.py','--dataset','CIFAR-10','--method','CLEAN',
                        '--epochs','100','--batch-size','128','--lr','0.001','--seed','42'],cwd=root,check=True)
    if method == 'GUE':
        save(root / 'assignment.json', dict(dataset=engine_dataset(dataset),
            generation_batch=POLICY['gue']['generation_batch'][dataset]))
        command = [generation_python,'generate_gue.py']
    else:
        script = 'generate_simple.py' if method in ['LSPNormal','LSPAUG','HYPO_P3'] else 'generate_advanced.py'
        command = [generation_python,script,'--dataset',engine_dataset(dataset),'--method',method]
    subprocess.run(command,cwd=root,check=True)
    common_train(root,dataset,method,python)
