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

    def test_reported_partial_results_are_explicit(self):
        rows=json.loads((REPO/'reported_results/aggregates.json').read_text())
        status=json.loads((REPO/'reported_results/policy.json').read_text())
        self.assertEqual(len(rows),status['completed_ue'])
        self.assertLessEqual(len(rows),48)
        self.assertFalse(any(r['dataset']=='X-ray' for r in rows))
        for r in rows:
            cm=r['confusion_matrix'];n=sum(map(sum,cm))
            self.assertEqual(n,r['n'])
            self.assertAlmostEqual(sum(cm[i][i]/sum(cm[i]) for i in range(len(cm)))/len(cm),r['balanced_accuracy'])

    def test_no_machine_specific_paths_in_entrypoints(self):
        for p in (REPO/'medunlearn').glob('*.py'):
            text=p.read_text()
            for forbidden in ['/home/user/','/disk1/','/var/data/']:
                self.assertNotIn(forbidden,text,str(p))


if __name__=='__main__':unittest.main()
