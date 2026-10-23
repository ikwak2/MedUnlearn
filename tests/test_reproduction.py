import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medunlearn.policy import POLICY, REPO, resolve
from medunlearn.io import load_module


class ReproductionTests(unittest.TestCase):
    def test_exact_main_policy_and_48_conditions(self):
        rows=[resolve(d,m) for d in POLICY['datasets'] for m in POLICY['methods']]
        self.assertEqual(len(rows),48)
        self.assertEqual(sum(r['scope']=='paper_aligned' for r in rows),16)
        self.assertEqual({r['method'] for r in rows if r['scope']=='paper_aligned'}, {'HYPO_P1','HYPO_P5','AR','T-AP'})
        self.assertEqual(resolve('MRI4','T-AP')['selection'],'fixed_final')
        self.assertEqual(resolve('MRI4','HYPO_P5')['selection'],'validation_best')
        self.assertEqual(resolve('X-ray(fixed)','GUE')['scope'],'common')
        self.assertEqual(resolve('CIFAR-10','LSPAUG')['clean_baseline'],'CLEAN_AUG')
        with self.assertRaises(ValueError):resolve('X-ray','GUE')

    def test_imbalanced_confusion_and_pairing(self):
        metrics=load_module('metrics_test',REPO/'engines/common/classwise_metrics.py')
        rows=[dict(sample_id=str(i),y_true='normal' if i<9 else 'disease',y_pred='normal') for i in range(10)]
        result=metrics.evaluate(rows,['normal','disease'])
        self.assertEqual(result['accuracy'],.9);self.assertEqual(result['balanced_accuracy'],.5)
        altered=[dict(r) for r in rows];altered[0]['sample_id']='different'
        with self.assertRaises(ValueError):metrics.compare(rows,altered,['normal','disease'])

    def test_reported_results_are_complete_and_consistent(self):
        rows=json.loads((REPO/'reported_results/aggregates.json').read_text())
        status=json.loads((REPO/'reported_results/policy.json').read_text())
        self.assertEqual(len(rows),status['completed_ue'])
        self.assertEqual(len(rows),48)
        self.assertFalse(any(r['dataset']=='X-ray' for r in rows))
        for r in rows:
            cm=r['confusion_matrix'];n=sum(map(sum,cm))
            self.assertEqual(n,r['n'])
            self.assertAlmostEqual(sum(cm[i][i]/sum(cm[i]) for i in range(len(cm)))/len(cm),r['balanced_accuracy'])

    def test_final_auxiliary_protocols_and_complete_figure_grid(self):
        base=REPO/'reported_results'
        def read(name):
            with (base/name).open() as stream:return list(csv.DictReader(stream))
        ssim=read('auxiliary/ssim.csv');cost=read('auxiliary/cost.csv')
        expected={(d,m) for d in POLICY['datasets'] for m in POLICY['methods']}
        self.assertEqual({(r['dataset'],r['method']) for r in ssim},expected)
        self.assertEqual(len(ssim),48)
        self.assertEqual(len(cost),33)
        for row in ssim+cost:
            native=row['method'] in {'HYPO_P1','HYPO_P5','AR','T-AP'}
            self.assertEqual(row['protocol_source'],'paper_aligned' if native else 'common')
            self.assertNotIn('host',row)
        for method in ['HYPO_P1','HYPO_P5']:
            row=next(r for r in ssim if r['dataset']=='CIFAR-10' and r['method']==method)
            self.assertEqual(int(row['n']),49000)
            self.assertIn('uint8',row['representation'])
        figures=read('figures56/comparison.csv')
        primary=[r for r in figures if r['primary']=='True']
        ratios={(r['dataset'],r['method'],int(r.get('target_ratio_pct') or r['ratio']))
                for r in primary if r['mode']=='ratio'
                and int(r.get('target_ratio_pct') or r['ratio']) in [20,40,60,80]}
        self.assertEqual(ratios,{(d,m,n) for d,m in expected for n in [20,40,60,80]})
        aa={(r['dataset'],r['method']) for r in primary if r['mode']=='aa' and r.get('condition')!='CLEAN'}
        self.assertTrue(expected.issubset(aa))
        progress=json.loads((base/'figures56/progress.json').read_text())
        self.assertEqual(progress['figure6_training_completed'],72)

    def test_release_source_hashes_and_minimal_vendoring(self):
        for name,entry in json.loads((REPO/'configs/source_provenance.json').read_text()).items():
            self.assertEqual(hashlib.sha256((REPO/name).read_bytes()).hexdigest(),entry['release_sha256'],name)
        for name in ['vgg.py','densenet.py','wideresnet.py']:
            self.assertFalse((REPO/'third_party/GUE/models'/name).exists())

    def test_no_machine_specific_paths_in_entrypoints(self):
        for p in (REPO/'medunlearn').glob('*.py'):
            text=p.read_text()
            for forbidden in ['/home/user/','/disk1/','/var/data/']:
                self.assertNotIn(forbidden,text,str(p))


if __name__=='__main__':unittest.main()
