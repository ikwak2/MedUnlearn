"""CLEAN versus REM: identical complete clean test samples and replay verification."""
import csv
import inspect
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torchvision.models import resnet18

from common import save, sha, verify


def replay_audit(logits,expected):
    predicted=logits.argmax(1)
    bad=np.flatnonzero(predicted!=expected)
    relative_gap=(logits[bad,predicted[bad]]-logits[bad,expected[bad]])/(np.abs(logits[bad]).max(1)+1e-12)
    # Small cross-GPU discrepancies are recorded, never substituted into Table 3.
    assert len(bad)<=max(1,int(len(expected)*.001)), 'Substantial prediction replay mismatch'
    assert not len(bad) or np.max(relative_gap)<=.01, 'Mismatch is not near a decision boundary'
    return dict(exact_match=len(bad)==0,mismatches=len(bad),n=len(expected),
        max_relative_logit_gap=float(relative_gap.max()) if len(bad) else 0.,
        numerical_guard='at most 0.1% samples (minimum one); relative logit gap <= 1%; original performance metrics retained')


def run(task):
    plan=json.loads((task/'task.json').read_text());manifest=verify(task)
    rows=list(csv.DictReader((task/'test.csv').open()))
    classes=json.loads((task/'classes.json').read_text())
    images=np.load(task/'test.npy',mmap_mode='r')
    assert len(images)==len(rows) and images.dtype==np.uint8
    y=np.array([int(r['class_index']) for r in rows])
    features={};reports={}
    for method in ['CLEAN','REM']:
        config=json.loads((task/f'{method}_config.json').read_text())
        assert config['split_sha256']['test']==sha(task/'test.csv')
        assert config['augmentation']=='none'
        assert config['weights']=='random initialization'
        model=resnet18(weights=None,num_classes=len(classes)).cuda().eval()
        checkpoint=torch.load(task/f'{method}.pt',map_location='cpu',weights_only=False)
        assert checkpoint['classes']==classes
        model.load_state_dict(checkpoint['model'])
        saved=list(csv.DictReader((task/f'{method}_predictions.csv').open()))
        expected={r['sample_id']:r['y_pred'] for r in saved}
        assert len(expected)==len(rows) and set(expected)=={r['sample_id'] for r in rows}
        vectors=[];predictions=[];collected=[];all_logits=[]
        hook=model.avgpool.register_forward_hook(lambda module,inputs,out:collected.append(out.detach().flatten(1).cpu().numpy()))
        with torch.inference_mode():
            for start in range(0,len(images),32):
                x=torch.from_numpy(np.array(images[start:start+32],copy=True)).float().div(255).cuda()
                logits=model(x).cpu().numpy();all_logits.append(logits)
                predictions.extend(logits.argmax(1).tolist())
        hook.remove();features[method]=np.concatenate(collected)
        expected_indices=np.array([classes.index(expected[r['sample_id']]) for r in rows])
        audit=replay_audit(np.concatenate(all_logits),expected_indices)
        assert np.isfinite(features[method]).all()
        np.save(task/f'{method}_features.npy',features[method])
        idx=np.random.default_rng(42).choice(len(y),min(2000,len(y)),replace=False)
        reports[method]=dict(checkpoint_epoch=checkpoint['epoch']+1,
            prediction_replay=audit,test_accuracy=float(np.mean(np.array(predictions)==y)),
            original_test_accuracy=float(np.mean(expected_indices==y)),
            feature_shape=list(features[method].shape),
            silhouette_512=float(silhouette_score(features[method][idx],y[idx])),
            feature_sha256=sha(task/f'{method}_features.npy'))
        del model;torch.cuda.empty_cache()
    # Fit separately: the coordinate systems of independently trained networks differ.
    # Identical sample IDs, seeds and t-SNE settings; panel positions are not comparable.
    public=task/'public';public.mkdir(exist_ok=True)
    for seed in [42,0,1]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for axis,method in zip(axes,['CLEAN','REM']):
            z=PCA(n_components=50,random_state=42).fit_transform(features[method])
            params=dict(n_components=2,perplexity=30,init='pca',learning_rate='auto',
                random_state=seed,method='barnes_hut',angle=.5,n_jobs=4)
            params['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1000
            embedding=TSNE(**params).fit_transform(z)
            np.save(task/f'{method}_tsne_seed{seed}.npy',embedding)
            for c,label in enumerate(classes):
                points=embedding[y==c]
                axis.scatter(points[:,0],points[:,1],s=4,alpha=.65,color=plt.get_cmap('tab20')(c),label=label,rasterized=True)
            axis.set_title(method);axis.set_xticks([]);axis.set_yticks([])
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=6,markerscale=2)
        fig.suptitle(plan['display_dataset']+' — identical clean test samples')
        fig.tight_layout()
        fig.savefig(public/f'tsne_seed{seed}.png',dpi=220,bbox_inches='tight')
        fig.savefig(public/f'tsne_seed{seed}.pdf',bbox_inches='tight');plt.close(fig)
    save(public/'summary.json',dict(**plan,n=len(y),models=reports,input_hashes=manifest,
        seeds=[42,0,1],perplexity=30,pca_components=50,iterations=1000,
        interpretation='Separate t-SNE fits; axis positions, distances, and apparent density cannot be compared across panels; qualitative only',
        patient_independence='verified for X-ray(fixed); unverified for MRI',torch=torch.__version__))
    save(task/'complete.json',dict(state='completed',time=time.time()))
