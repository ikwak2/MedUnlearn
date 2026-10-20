"""The published split/filter recipes, with user-controlled filesystem roots."""
import json
from pathlib import Path
import shutil
from .io import load_module, save
from .policy import REPO


def prepare(data_root, workspace):
    data_root, workspace = Path(data_root).resolve(), Path(workspace).resolve()
    manifest = workspace / 'manifests'
    if manifest.exists(): raise ValueError('Use a new workspace; prepared manifests are immutable')
    parent = workspace / 'preparation/grouped_splits'
    module = load_module('split_recipe', REPO / 'data_tools/prepare_splits.py')
    module.ROOT = data_root
    module.main(parent)
    single = workspace / 'preparation/xray_single_annotation'
    module = load_module('single_annotation', REPO / 'data_tools/filter_single_annotation.py')
    module.prepare(parent / 'X-ray', data_root / 'xray/Data_Entry_2017.csv', single)
    fixed = workspace / 'preparation/xray_fixed'
    module = load_module('rare_filter', REPO / 'data_tools/filter_rare_classes.py')
    audit = module.prepare(single, fixed, minimum_train_n=50)
    manifest.mkdir()
    for dataset in ['MRI4', 'MRI15', 'CIFAR-10']:
        shutil.copytree(parent / dataset, manifest / dataset)
    shutil.copytree(fixed / 'X-ray', manifest / 'X-ray')
    shutil.copyfile(fixed / 'COMPLETE.json', manifest / 'xray_fixed_audit.json')
    save(workspace / 'data_config.json', dict(data_root=str(data_root), manifest_root=str(manifest)))
    expected = {'train':8843,'validation':3758,'test':8417}
    actual = {s:v['n'] for s,v in audit['splits'].items()}
    if actual != expected:
        raise ValueError('Prepared a different X-ray cohort; original source memberships are required for the published benchmark: ' + str(actual))
    from .runtime import cache
    for dataset in ['MRI4','MRI15','X-ray(fixed)','CIFAR-10']:cache(workspace,dataset)
    return manifest


def import_manifests(source, data_root, workspace):
    """Accept an already authorized local copy; individual identities never ship with the repo."""
    source, data_root, workspace = Path(source).resolve(), Path(data_root).resolve(), Path(workspace).resolve()
    target = workspace / 'manifests'
    if target.exists(): raise ValueError('Use a new workspace')
    target.mkdir(parents=True)
    for dataset in ['MRI4','MRI15','CIFAR-10','X-ray']:
        shutil.copytree(source / dataset, target / dataset)
    audit = source / 'xray_fixed_audit.json'
    if audit.exists(): shutil.copyfile(audit, target / audit.name)
    elif (source / 'COMPLETE.json').exists():
        value = json.loads((source / 'COMPLETE.json').read_text())
        if value.get('data_version') == 'xray_singlelabel12_v3':
            save(target / 'xray_fixed_audit.json', value)
    save(workspace / 'data_config.json', dict(data_root=str(data_root),manifest_root=str(target)))
    from .runtime import cache
    for dataset in ['MRI4','MRI15','X-ray(fixed)','CIFAR-10']:cache(workspace,dataset)
    return target
