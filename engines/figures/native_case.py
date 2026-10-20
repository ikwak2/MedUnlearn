"""Run frozen native trainers: mixed-training data or train-only official UEraser."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from util import save, sha
from mixing import make_masks


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


class MixedArray:
    def __init__(self, clean, poison, mask):
        self.clean, self.poison, self.mask = clean, poison, mask
        self.shape, self.dtype = clean.shape, clean.dtype

    def __getitem__(self, index):
        return (self.poison if self.mask[index] else self.clean)[index]


def setup(job):
    plan = json.loads((job / 'plan.json').read_text())
    source = Path(plan['reference_root'])
    frozen = Path(plan['frozen_code'])
    for name, digest in json.loads((frozen / 'snapshot_manifest.json').read_text()).items():
        assert sha(frozen / name) == digest, 'Changed frozen native source: ' + name
    for name, digest in json.loads((job / 'inputs.json').read_text()).items():
        assert sha(name) == digest, 'Changed experiment input: ' + name
    sys.path.insert(0, str(frozen))
    e = load('experiment', frozen / ('experiment_aa.py' if plan['mode'] == 'aa' else 'experiment.py'))
    e.INPUTS = source / 'inputs'
    e.RECIPE = 'AR' if plan['family'] in ['ar', 'tap'] else 'HYPO'
    task = plan['task']; data = e.load_data(task)
    module = e
    if plan['family'] == 'tap':
        module = load('tap_experiment', frozen / ('tap_experiment_aa.py' if plan['mode'] == 'aa' else 'tap_experiment.py'))
    native = source / plan['family'] / task
    condition = plan['condition']
    if condition == 'CLEAN':
        images = data['train_x']
    else:
        poison = np.load(native / 'poisons' / (condition + '.npy'), mmap_mode='r')
        assert poison.shape == data['train_x'].shape
        images = poison
        if plan['mode'] == 'ratio':
            indices = np.array(data['train_indices'])
            masks, details = make_masks(data['train_y'][indices], seed=42)
            mask = np.zeros(len(data['train_y']), dtype=bool)
            mask[indices] = masks[plan['ratio']]
            # Native P1/P5 CIFAR holds out poisoned, augmented validation images.
            # Preserve that validation protocol; the x-axis varies training images only.
            if data['val_indices'] is not None: mask[data['val_indices']] = True
            np.save(job / 'mask.npy', mask)
            save(job / 'mask_summary.json', dict(**details[plan['ratio']], seed=42,
                holdout_policy='Native CIFAR P1/P5 poison holdout remains fully poisoned; only train ratio varies'))
            images = MixedArray(data['train_x'], poison, mask)
    clean = job / 'CLEAN'
    if condition != 'CLEAN' and not clean.exists():
        clean.symlink_to(Path(plan['aa_clean_job']) / 'CLEAN' if plan['mode'] == 'aa' else native / 'CLEAN', target_is_directory=True)
    return plan, e, module, data, images


def main(job, smoke=False):
    plan, e, module, data, images = setup(job)
    assert torch.cuda.device_count() == 1 and torch.__version__ == '2.0.1'
    if smoke:
        e.seed_all()
        batch = 128 if plan['task'] == 'cifar10' else 32
        ds = e.Images(images, data['train_y'], data['train_indices'], augment=True, normalization=data['normalization'])
        x = torch.stack([ds[i][0] for i in range(batch)]).cuda()
        y = torch.tensor([ds[i][1] for i in range(batch)], device='cuda')
        model = (module.downstream_model(plan['task'], data['classes']) if plan['family'] == 'tap'
                 else e.model_for(plan['task'], data['classes'])).cuda()
        torch.cuda.reset_peak_memory_stats()
        if plan['mode'] == 'aa':
            from aa_native import native_loss
            loss, logits = native_loss(model, x, y, 0, data['normalization'])
        else:
            logits = model(x); loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward(); torch.cuda.synchronize()
        assert torch.isfinite(loss) and logits.shape == (batch, len(data['classes']))
        save(job / 'smoke.json', dict(passed=True, batch=batch, loss=float(loss.detach()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0), repeat=5 if plan['mode'] == 'aa' else 1))
        return
    module.train(plan['task'], plan['condition'], data, images, job)
    run = job / plan['condition']
    config = json.loads((run / 'config.json').read_text())
    native = Path(plan['reference_root']) / plan['family'] / plan['task']
    reference = json.loads((native / plan['condition'] / 'config.json').read_text())
    # Only hardware provenance and the explicitly added AA objective may differ.
    for key, expected in reference.items():
        if key not in ['host', 'gpu_uuid']: assert config[key] == expected, ('Native config changed', key)
    if plan['mode'] == 'aa':
        from aa_loss import AA_CONFIG
        assert config['adversarial_augmentation'] == AA_CONFIG
    from classwise_metrics import read_predictions, compare
    reports = json.loads((run / 'metrics.json').read_text())
    public = []
    children = []
    for selection, report in reports.items():
        predictions = read_predictions(run / (selection + '_predictions.csv'))
        plain_path = native / plan['condition'] / (selection + '_predictions.csv')
        plain, recovery = compare(read_predictions(plain_path), predictions, data['classes'])
        control_path = (job / 'CLEAN' if plan['condition'] != 'CLEAN' else run) / (selection + '_predictions.csv')
        clean, paired = compare(read_predictions(control_path), predictions, data['classes'])
        m = report['metrics']
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1']: assert abs(m[key] - paired[key]) < 1e-10
        row = dict(dataset=plan['display_dataset'], method=plan['method'], condition=plan['condition'],
            family=plan['family'], mode=plan['mode'], ratio=plan.get('ratio'), selection=selection,
            primary=selection == plan['primary_selection'], protocol=config['protocol'],
            accuracy_pct=100*m['accuracy'], balanced_accuracy_pct=100*m['balanced_accuracy'],
            macro_f1_pct=100*m['macro_f1'], clean_balanced_accuracy_pct=100*clean['balanced_accuracy'],
            ba_drop_pp=paired['balanced_accuracy_drop_pp'], plain_balanced_accuracy_pct=100*plain['balanced_accuracy'],
            ba_recovery_pp=100*(m['balanced_accuracy']-plain['balanced_accuracy']),
            selected_epoch=report['selected_epoch'], seed=0, test_n=m['n'],
            config_sha256=sha(run/'config.json'), predictions_sha256=sha(run/(selection+'_predictions.csv')),
            input_manifest_sha256=sha(job/'inputs.json'), gpu=plan['gpu'], host=os.environ.get('MEDUNLEARN_HOST','local'))
        public.append(row)
        children.extend(dict(dataset=plan['display_dataset'],method=plan['method'],condition=plan['condition'],
            mode=plan['mode'],ratio=plan.get('ratio'),selection=selection,primary=row['primary'],
            class_label=label,test_n=m['support'][i],clean_recall_pct=100*clean['recall'][i],
            recall_pct=100*m['recall'][i],recall_drop_pp=paired['recall_drop_pp'][i],
            plain_recall_pct=100*plain['recall'][i]) for i,label in enumerate(data['classes']))
    save(job/'public.json',dict(summary=public,classwise=children))
    save(job/'COMPLETE.json',dict(state='completed',time=__import__('time').time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();main(a.job,a.smoke)
