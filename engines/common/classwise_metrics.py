"""Single-label evaluation, with explicit labels and paired clean/UE samples.
CSV columns: sample_id,y_true,y_pred (semantic labels, not inferred indices).
Run: python3 classwise_metrics.py --classes classes.json --clean clean.csv
     --ue REM=rem.csv --out results
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def evaluate(rows, classes):
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('Classes must be nonempty and unique')
    index = {name: i for i, name in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    seen = set()
    for r in rows:
        sid = r['sample_id']
        if not sid or sid in seen:
            raise ValueError('Empty or duplicate sample_id: ' + sid)
        seen.add(sid)
        if r['y_true'] not in index or r['y_pred'] not in index:
            raise ValueError('Unknown label: ' + str(r))
        cm[index[r['y_true']]][index[r['y_pred']]] += 1
    support = [sum(r) for r in cm]
    if any(n == 0 for n in support):
        raise ValueError('All declared classes must have test support; absent classes cannot be silently dropped')
    recall = [cm[i][i] / support[i] for i in range(len(classes))]
    f1 = [2 * cm[i][i] / (support[i] + sum(r[i] for r in cm)) for i in range(len(classes))]
    return {'n': sum(support), 'accuracy': sum(cm[i][i] for i in range(len(classes))) / sum(support),
            'balanced_accuracy': sum(recall) / len(classes), 'macro_f1': sum(f1) / len(classes),
            'classes': classes, 'support': support, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm}


def read_predictions(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if not {'sample_id', 'y_true', 'y_pred'} <= set(reader.fieldnames or []):
            raise ValueError('Required columns: sample_id,y_true,y_pred')
        return list(reader)


def compare(clean_rows, ue_rows, classes):
    clean, ue = evaluate(clean_rows, classes), evaluate(ue_rows, classes)
    identity = lambda rows: {r['sample_id']: r['y_true'] for r in rows}
    if identity(clean_rows) != identity(ue_rows):
        raise ValueError('Clean and UE must use exactly the same sample IDs and true labels')
    ue['recall_drop_pp'] = [100 * (c-u) for c, u in zip(clean['recall'], ue['recall'])]
    ue['balanced_accuracy_drop_pp'] = 100 * (clean['balanced_accuracy'] - ue['balanced_accuracy'])
    return clean, ue


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classes', required=True, help='JSON list of semantic labels in model output order')
    p.add_argument('--clean', required=True)
    p.add_argument('--ue', action='append', default=[], help='METHOD=prediction.csv; one seed/run per invocation')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    classes = json.loads(Path(a.classes).read_text())
    clean_rows = read_predictions(a.clean)
    reports = {'CLEAN': evaluate(clean_rows, classes)}
    inputs = {'CLEAN': a.clean}
    for spec in a.ue:
        name, path = spec.split('=', 1)
        if not name or name in reports:
            raise ValueError('Duplicate or empty method name')
        reports[name] = compare(clean_rows, read_predictions(path), classes)[1]
        inputs[name] = path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
    write_csv(out/'summary.csv', [{'method': m, 'n': r['n'], 'accuracy_pct': 100*r['accuracy'],
        'balanced_accuracy_pct': 100*r['balanced_accuracy'], 'macro_f1_pct': 100*r['macro_f1']} for m,r in reports.items()])
    write_csv(out/'classwise.csv', [{'method': m, 'class_label': c, 'test_n': r['support'][i],
        'recall_pct': 100*r['recall'][i], 'clean_recall_pct': 100*reports['CLEAN']['recall'][i],
        'recall_drop_pp': r.get('recall_drop_pp', [0]*len(classes))[i]} for m,r in reports.items() for i,c in enumerate(classes)])
    (out/'provenance.json').write_text(json.dumps({'inputs': {m:{'path':str(Path(f).resolve()),
        'sha256':hashlib.sha256(Path(f).read_bytes()).hexdigest()} for m,f in inputs.items()},
        'classes':classes, 'units':'JSON fractions; CSV percent and percentage points'}, indent=2))

if __name__ == '__main__':
    main()
