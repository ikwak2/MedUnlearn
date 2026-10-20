"""Versioned corrected splits, canonical labels and exclusions; never mutates source data."""
import csv, hashlib, json, pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path('.')
EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}

def dumpcsv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): raise RuntimeError('Immutable split directory exists; use a new version')
    excluded=[]; report={}; allcounts=[]
    metadata={r['Image Index']:r['Patient ID'] for r in csv.DictReader((ROOT/'xray/Data_Entry_2017.csv').open())}
    for dataset,folder in [('MRI4','mri'),('MRI15','mri_15'),('X-ray','xray'),('CIFAR-10','cifar-10-batches-py')]:
        pools={}
        for split in ['train','test']:
            rows=[]
            if dataset=='CIFAR-10':
                with (ROOT/folder/'batches.meta').open('rb') as f: names=[s.decode() for s in pickle.load(f,encoding='bytes')[b'label_names']]
                for batch in ([f'data_batch_{i}' for i in range(1,6)] if split=='train' else ['test_batch']):
                    with (ROOT/folder/batch).open('rb') as f: data=pickle.load(f,encoding='bytes')
                    for i,(im,y) in enumerate(zip(data[b'data'],data[b'labels'])):
                        rows.append(dict(sample_id=f'{folder}/{batch}:{i}',path=str(ROOT/folder/batch),index=i,label=names[y],patient_id='',sha256=hashlib.sha256(im.tobytes()).hexdigest()))
            else:
                if dataset=='X-ray': source=[(ROOT/folder/split/r['Image'],r['label'],metadata[r['Image']]) for r in csv.DictReader((ROOT/folder/(split+'.csv')).open())]
                else: source=[(f, 'NORMAL' if f.parent.name=='_NORMAL' else f.parent.name,'') for f in sorted((ROOT/folder/split).rglob('*')) if f.is_file() and f.suffix.lower() in EXT]
                for f,label,patient in source:
                    if '_noisy' in f.name.lower():
                        excluded.append(dict(dataset=dataset,sample_id=str(f.relative_to(ROOT)),reason='generated_noisy_demo_in_source_train',original_split=split)); continue
                    rows.append(dict(sample_id=str(f.relative_to(ROOT)),path=str(f),index=-1,label=label,patient_id=patient,sha256=sha(f)))
            pools[split]=rows
        test=pools['test']; test_hash={r['sha256'] for r in test}; test_patients={r['patient_id'] for r in test if r['patient_id']}
        pool=[]
        for r in pools['train']:
            reason='patient_overlaps_test' if r['patient_id'] and r['patient_id'] in test_patients else ('exact_duplicate_of_test' if r['sha256'] in test_hash else None)
            if reason: excluded.append(dict(dataset=dataset,sample_id=r['sample_id'],reason=reason,original_split='train'))
            else: pool.append(r)
        classes=sorted({r['label'] for r in pool}); mapping={c:i for i,c in enumerate(classes)}
        assert set(classes)=={r['label'] for r in test}
        # Union identical-content records and (where known) all images of a patient.
        parent=list(range(len(pool)))
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b): parent[find(a)]=find(b)
        first={}
        for i,r in enumerate(pool):
            keys=['hash:'+r['sha256']]+(['patient:'+r['patient_id']] if r['patient_id'] else [])
            for key in keys:
                if key in first: union(i,first[key])
                else: first[key]=i
        groups=np.array([find(i) for i in range(len(pool))]); y=np.array([mapping[r['label']] for r in pool]); target=np.bincount(y,minlength=len(classes))*.3
        best=None
        for trainidx,validx in GroupShuffleSplit(n_splits=256,test_size=.3,random_state=42).split(y,y,groups):
            vc=np.bincount(y[validx],minlength=len(classes)); tc=np.bincount(y[trainidx],minlength=len(classes))
            if min(vc)==0 or min(tc)==0: continue
            score=float(np.mean(((vc-target)/np.maximum(target,1))**2)+((len(validx)-len(pool)*.3)/(len(pool)*.3))**2)
            if best is None or score<best[0]: best=(score,trainidx,validx)
        if best is None: raise ValueError('Cannot preserve all classes in group-disjoint splits')
        splits={'train':[pool[i] for i in best[1]],'validation':[pool[i] for i in best[2]],'test':test}
        directory=out/dataset; directory.mkdir(exist_ok=True)
        for split,rows in splits.items():
            for r in rows: r['class_index']=mapping[r['label']]
            rows.sort(key=lambda r:r['sample_id']); dumpcsv(directory/(split+'.csv'),rows)
            counter=Counter(r['label'] for r in rows)
            allcounts.extend(dict(dataset=dataset,split=split,class_label=c,n=counter[c],percent=100*counter[c]/len(rows)) for c in classes)
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not ({r['sha256'] for r in splits[a]} & {r['sha256'] for r in splits[b]})
            assert not ({r['patient_id'] for r in splits[a] if r['patient_id']} & {r['patient_id'] for r in splits[b] if r['patient_id']})
        (directory/'classes.json').write_text(json.dumps(classes,indent=2))
        hashes=defaultdict(set)
        for r in pool+test: hashes[r['sha256']].add(r['label'])
        report[dataset]={'sizes':{s:len(r) for s,r in splits.items()},'classes':classes,'exclusions':dict(Counter(r['reason'] for r in excluded if r['dataset']==dataset)),
            'test_unique_content':len(test_hash),'cross_label_identical_hashes':sum(len(v)>1 for v in hashes.values()),
            'patient_grouping':dataset=='X-ray','train_validation_group_selection':'256 group shuffle candidates, closest class balance to 30% using only train pool; seed 42',
            'split_sha256':{s:sha(directory/(s+'.csv')) for s in splits}}
        print(dataset,report[dataset],flush=True)
    dumpcsv(out/'class_composition.csv',allcounts)
    if excluded: dumpcsv(out/'exclusions.csv',excluded)
    (out/'COMPLETE.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
