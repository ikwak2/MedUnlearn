import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if not (REPO/'configs/table3.json').exists():
    REPO = next(p for p in [Path(sys.prefix)/'share/medunlearn', REPO/'share/medunlearn']
                if (p/'configs/table3.json').exists())
POLICY = json.loads((REPO / 'configs/table3.json').read_text())
TASKS = {'CIFAR-10': 'cifar10', 'MRI4': 'mri4', 'MRI15': 'mri15', 'X-ray(fixed)': 'xrayfixed'}


def engine_dataset(dataset):
    return 'X-ray' if dataset == 'X-ray(fixed)' else dataset


def resolve(dataset, method):
    if dataset not in POLICY['datasets'] or method not in POLICY['methods']:
        raise ValueError('Unknown final-paper dataset or method')
    native = POLICY['paper_aligned'].get(method)
    return dict(dataset=dataset, engine_dataset=engine_dataset(dataset), method=method,
                task=TASKS[dataset], scope='paper_aligned' if native else 'common',
                seed=0 if native else 42, selection=native['selection'] if native else 'validation_best',
                clean_baseline='CLEAN_AUG' if method == 'LSPAUG' else 'CLEAN',
                **({k:v for k,v in native.items() if k!='selection'} if native else {}))
