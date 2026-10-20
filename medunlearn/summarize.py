"""Compute Table3 and class aggregates from the selected, paired saved predictions."""
import json
from pathlib import Path
from .io import load_module, readcsv, save, sha, writecsv
from .policy import REPO, POLICY, TASKS, engine_dataset, resolve


def summarize(workspace, output=None):
    workspace = Path(workspace).resolve()
    output = Path(output).resolve() if output else workspace / 'results'
    metrics = load_module('paired_metrics', REPO / 'engines/common/classwise_metrics.py')
    rows, class_rows, reports = [], [], []
    for dataset in POLICY['datasets']:
        for method in POLICY['methods']:
            p = resolve(dataset,method)
            if p['scope'] == 'paper_aligned':
                root = workspace/'reference'/p['family']/p['task']
                run, clean = root/p['condition'],root/'CLEAN'
                filename = p['selection']+'_predictions.csv'
                complete = run/'complete.json'
            else:
                root=workspace/'common'/TASKS[dataset]
                tree='runs_sgd' if dataset=='CIFAR-10' and method!='LSPAUG' else 'runs'
                run=root/tree/engine_dataset(dataset)/method/'seed42'
                clean=root/tree/engine_dataset(dataset)/p['clean_baseline']/'seed42'
                filename='test_predictions.csv';complete=run/'complete.json'
            row=dict(dataset=dataset,method=method,status='pending',protocol_source=p['scope'],
                     seed=p['seed'],selection=p['selection'],clean_baseline=p.get('family',p['clean_baseline'])+':CLEAN',
                     accuracy_pct='',balanced_accuracy_pct='',macro_f1_pct='',ba_drop_pp='')
            if complete.exists():
                config=json.loads((run/'config.json').read_text())
                classes=json.loads((run/'metrics.json').read_text())
                if p['scope']=='paper_aligned':
                    stored=classes[p['selection']];classes=stored['metrics']['classes'];epoch=stored['selected_epoch']
                else:
                    classes=classes['classes'];epoch=json.loads(complete.read_text())['selected_epoch']
                if method=='GUE':
                    meta=json.loads((root/'poisons'/engine_dataset(dataset)/'GUE.npy.json').read_text())
                    if meta.get('generation_protocol')!='gue_official_generation_common_eval_v2':
                        raise ValueError('Superseded GUE outputs cannot enter final Table3')
                predictions=metrics.read_predictions(run/filename)
                if p['scope']=='common' or dataset!='CIFAR-10':
                    split_root=(workspace/'inputs'/p['task'] if p['scope']=='paper_aligned' else root)/'splits_v1'/engine_dataset(dataset)
                    expected_hashes={s:sha(split_root/(s+'.csv')) for s in ['train','validation','test']}
                    clean_config=json.loads((clean/'config.json').read_text())
                    if config['split_sha256']!=expected_hashes or clean_config['split_sha256']!=expected_hashes:
                        raise ValueError('Stored CLEAN/UE config does not match the declared splits')
                    truth={r['sample_id']:r['label'] if 'label' in r else classes[int(r['class_index'])] for r in readcsv(split_root/'test.csv')}
                    if {r['sample_id']:r['y_true'] for r in predictions}!=truth:
                        raise ValueError('Predictions do not match the complete declared test set')
                c,m=metrics.compare(metrics.read_predictions(clean/filename),predictions,classes)
                stored_metrics=stored['metrics'] if p['scope']=='paper_aligned' else json.loads(complete.read_text())['metrics']
                for key in ['accuracy','balanced_accuracy','macro_f1']:
                    if abs(m[key]-stored_metrics[key])>1e-9:raise ValueError('Stored metric mismatch: '+key)
                row.update(status='completed',accuracy_pct=100*m['accuracy'],balanced_accuracy_pct=100*m['balanced_accuracy'],
                    macro_f1_pct=100*m['macro_f1'],ba_drop_pp=m['balanced_accuracy_drop_pp'],
                    clean_accuracy_pct=100*c['accuracy'],clean_balanced_accuracy_pct=100*c['balanced_accuracy'],clean_macro_f1_pct=100*c['macro_f1'],
                    selected_epoch=epoch,test_n=m['n'],config_sha256=sha(run/'config.json'),
                    predictions_sha256=sha(run/filename),clean_predictions_sha256=sha(clean/filename))
                reports.append(dict(dataset=dataset,method=method,classes=classes,clean_baseline=row['clean_baseline'],
                    confusion_matrix=m['confusion_matrix'],clean_confusion_matrix=c['confusion_matrix'],
                    support=m['support'],predicted=[sum(r[j] for r in m['confusion_matrix']) for j in range(len(classes))],
                    n=m['n'],accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],
                    recall=m['recall'],provenance=dict(config_sha256=row['config_sha256'],predictions_sha256=row['predictions_sha256'])))
                class_rows.extend(dict(dataset=dataset,method=method,selection=p['selection'],class_label=label,
                    clean_baseline=row['clean_baseline'],test_n=m['support'][i],clean_recall_pct=100*c['recall'][i],
                    recall_pct=100*m['recall'][i],recall_drop_pp=m['recall_drop_pp'][i]) for i,label in enumerate(classes))
            rows.append(row)
    output.mkdir(parents=True,exist_ok=True)
    writecsv(output/'comparison.csv',rows);writecsv(output/'classwise.csv',class_rows)
    save(output/'aggregates.json',reports)
    print('Completed:',len(reports),'/48 UE conditions; aggregate results:',output)
    return output
