"""Remove classes using training support only, preserving all parent split memberships."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

VERSION = 'xray_singlelabel12_v3'
SPLITS = ('train', 'validation', 'test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def prepare(source, out, minimum_train_n=50):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError('Use a new immutable output directory')
    if minimum_train_n < 1:
        raise ValueError('Minimum support must be positive')
    parent = json.loads((source/'COMPLETE.json').read_text())
    if parent['data_version'] != 'xray_singlelabel_v2':
        raise ValueError('Expected the original single-annotation v2 subset')
    original_classes = json.loads((source/'X-ray/classes.json').read_text())
    records = {}
    for split in SPLITS:
        path = source/'X-ray'/(split+'.csv')
        if sha(path) != parent['split_sha256'][split]:
            raise ValueError('Parent manifest changed')
        records[split] = read(path)
        if any(original_classes[int(r['class_index'])] != r['label'] for r in records[split]):
            raise ValueError('Parent label/index mismatch')
    train_counts = Counter(r['label'] for r in records['train'])
    classes = [c for c in original_classes if train_counts[c] >= minimum_train_n]
    if len(classes) < 2:
        raise ValueError('Fewer than two retained classes')
    removed = [c for c in original_classes if c not in classes]
    mapping = {c: str(i) for i, c in enumerate(classes)}
    kept, exclusions, composition, stats = {}, [], [], {}
    sample_ids = set()
    for split, rows in records.items():
        kept[split] = []
        for row in rows:
            if row['sample_id'] in sample_ids:
                raise ValueError('Duplicate sample identity')
            sample_ids.add(row['sample_id'])
            if row['label'] not in mapping:
                exclusions.append(dict(split=split, sample_id=row['sample_id'], class_label=row['label'],
                                       reason='training_class_support_below_'+str(minimum_train_n)))
            else:
                kept[split].append(dict(row, class_index=mapping[row['label']]))
        counts = Counter(r['label'] for r in kept[split])
        if set(counts) != set(classes):
            raise ValueError('A retained class has no split support')
        n = len(kept[split])
        stats[split] = dict(source_n=len(rows), n=n, excluded_rare_class_n=len(rows)-n,
                            patient_n=len({r['patient_id'] for r in kept[split]}),
                            unique_content_n=len({r['sha256'] for r in kept[split]}),
                            no_finding_percent=100*counts.get('No Finding', 0)/n)
        composition.extend(dict(dataset='X-ray', split=split, class_label=c, n=counts[c],
                                percent=100*counts[c]/n) for c in classes)
    overlaps = {}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlaps[a+'/'+b] = {field: len({r[field] for r in kept[a]} & {r[field] for r in kept[b]})
                             for field in ['patient_id', 'sha256', 'sample_id']}
        if any(overlaps[a+'/'+b].values()):
            raise ValueError('Patient/content/identity overlap')
    directory = out/'X-ray'
    directory.mkdir(parents=True)
    (directory/'classes.json').write_text(json.dumps(classes, indent=2)+'\n')
    for split, rows in kept.items():
        write(directory/(split+'.csv'), rows)
    write(out/'class_composition.csv', composition)
    write(out/'exclusions.csv', exclusions, ['split', 'sample_id', 'class_label', 'reason'])
    audit = dict(data_version=VERSION, dataset='X-ray', parent_data_version=parent['data_version'],
                 classes=classes, excluded_classes=removed, minimum_train_n=minimum_train_n,
                 class_selection='Single-annotation TRAIN image count >= threshold; no validation/test performance used',
                 train_counts_before_filter=dict(train_counts), splits=stats, overlaps=overlaps,
                 split_assignment='Unchanged patient-disjoint v2 membership; no reshuffle; contiguous class indices rebuilt',
                 annotation_limit=parent['annotation_limit'],
                 original_metadata_sha256=parent['original_metadata_sha256'],
                 parent_audit_sha256=sha(source/'COMPLETE.json'), parent_manifest_sha256=parent['split_sha256'],
                 split_sha256={s: sha(directory/(s+'.csv')) for s in SPLITS})
    (out/'COMPLETE.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--minimum-train-n', type=int, default=50)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.out, a.minimum_train_n), indent=2))
