"""Filter all three existing patient-disjoint splits using original NIH annotations."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

VERSION = 'xray_singlelabel_v2'
SPLITS = ('train', 'validation', 'test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def prepare(source, metadata, out):
    source, metadata, out = Path(source), Path(metadata), Path(out)
    if out.exists():
        raise ValueError('Use a new immutable split directory')
    classes = json.loads((source / 'classes.json').read_text())
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('Invalid canonical classes')
    mapping = {c: i for i, c in enumerate(classes)}
    meta = {}
    for r in read_csv(metadata):
        key = r['Image Index']
        if not key or key in meta:
            raise ValueError('Empty or duplicate metadata image ID')
        meta[key] = r
    kept, excluded, stats, seen = {}, [], {}, set()
    composition = []
    for split in SPLITS:
        original = read_csv(source / (split + '.csv'))
        selected = []
        for r in original:
            if not r['sample_id'] or r['sample_id'] in seen:
                raise ValueError('Duplicate sample ID across source splits')
            seen.add(r['sample_id'])
            annotation = meta[Path(r['path']).name]
            labels = [x.strip() for x in annotation['Finding Labels'].split('|')]
            if not labels or any(x not in mapping for x in labels) or len(set(labels)) != len(labels):
                raise ValueError('Invalid original annotation')
            if not annotation['Patient ID'] or annotation['Patient ID'] != r['patient_id']:
                raise ValueError('Patient mapping mismatch')
            if r['label'] != labels[0] or int(r['class_index']) != mapping[r['label']]:
                raise ValueError('Source label mapping mismatch')
            if 'No Finding' in labels and len(labels) != 1:
                raise ValueError('No Finding combined with a disease')
            if len(labels) != 1:
                excluded.append(dict(split=split, sample_id=r['sample_id'], reason='multiple_original_finding_labels'))
                continue
            selected.append(dict(r, label=labels[0], class_index=str(mapping[labels[0]])))
        counts = Counter(r['label'] for r in selected)
        if set(counts) != set(classes):
            raise ValueError('Every canonical class must have support in every split')
        kept[split] = selected
        stats[split] = dict(source_n=len(original), n=len(selected), excluded_multilabel_n=len(original)-len(selected),
                            patient_n=len({r['patient_id'] for r in selected}),
                            unique_content_n=len({r['sha256'] for r in selected}),
                            no_finding_percent=100*counts.get('No Finding', 0)/len(selected))
        composition.extend(dict(dataset='X-ray', split=split, class_label=c, n=counts[c], percent=100*counts[c]/len(selected)) for c in classes)
    overlaps = {}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        pair = {field: len({r[field] for r in kept[a]} & {r[field] for r in kept[b]}) for field in ['patient_id', 'sha256']}
        if any(pair.values()):
            raise ValueError('Patient or exact-content overlap between splits')
        overlaps[a+'/'+b] = pair
    directory = out / 'X-ray'
    directory.mkdir(parents=True)
    (directory/'classes.json').write_text(json.dumps(classes, indent=2)+'\n')
    for split, rows in kept.items():
        write_csv(directory/(split+'.csv'), rows)
    write_csv(out/'class_composition.csv', composition)
    write_csv(out/'exclusions.csv', excluded, ['split', 'sample_id', 'reason'])
    audit = dict(data_version=VERSION, dataset='X-ray', classes=classes, splits=stats,
                 selection='Original Finding Labels has exactly one entry: one disease or No Finding',
                 split_assignment='Inherited unchanged from patient-disjoint splits_v1; no reshuffle',
                 annotation_limit='Single annotated label does not establish absence of other clinical conditions',
                 overlaps=overlaps, original_metadata_sha256=sha(metadata),
                 parent_manifest_sha256={s: sha(source/(s+'.csv')) for s in SPLITS},
                 split_sha256={s: sha(directory/(s+'.csv')) for s in SPLITS})
    (out/'COMPLETE.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.metadata, args.out), indent=2))
