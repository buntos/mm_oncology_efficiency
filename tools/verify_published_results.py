#!/usr/bin/env python3
"""Read-only CPU verification of published predictions and resource tables.
Requires NumPy. No weights, images, torch, training or network access.
Reuses the hash-verified Stage 32 statistical functions; not an independent
implementation or a verification of model inference.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import sys
sys.dont_write_bytecode=True
import hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
STAGE32_SHA='9b55353056dca798e3e480f08f5ac65ff9fd83b9a55c004e45147ac240cf9fab'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)
def main():
    path=ROOT/'scripts/32_evaluate_patch_budget_test.py'
    require(sha(path)==STAGE32_SHA,'Stage 32 code hash mismatch')
    spec=importlib.util.spec_from_file_location('frozen_stage32',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name,h in m.CODE_HASHES.items():require(sha(ROOT/'scripts'/name)==h,'Code hash mismatch: '+name)
    source=ROOT/'results'/m.SOURCE_NAME
    require(sha(source/'protocol.json')==m.STAGE31_PROTOCOL_SHA,'Stage31 protocol mismatch')
    bundle=m.load_stage29().load_bundle(ROOT)
    out=ROOT/'results'/m.OUTPUT_NAME
    completion=m.read_json(out/'completion.json')
    require(completion['status']=='COMPLETE','Stage32 not complete')
    for name,h in completion['files_sha256'].items():require(sha(out/name)==h,'Stage32 output hash mismatch: '+name)
    labels={p:int(r['class_label']=='LUSC') for p,r in bundle['patients'].items()}
    ids=sorted(p for p,r in bundle['patients'].items() if r['split']=='test')
    require(len(ids)==186,'Wrong test patient count')
    predictions=m.read_csv(out/'test_predictions.csv')
    require(len(predictions)==9*186,'Incomplete test matrix')
    require({(int(r['budget']),int(r['seed'])) for r in predictions}==set(m.MATRIX),'Unexpected run keys')
    scores={};actual=[];models={}
    for k,s in m.MATRIX:
        name=m.run_name(k,s);best=m.read_json(source/'runs'/name/'best_model.json')
        require(best['sha256']==m.REVIEWED_MODELS[name]['sha256'] and best['epoch']==m.REVIEWED_MODELS[name]['epoch'],'Selected model metadata mismatch')
        rows=[r for r in predictions if int(r['budget'])==k and int(r['seed'])==s]
        require(all(int(r['selected_epoch'])==best['epoch'] for r in rows),'Selected epoch mismatch')
        indexed=m.prediction_map(rows,ids,labels)
        y=np.array([labels[p] for p in ids]);p=np.array([indexed[i]['p_LUSC'] for i in ids]);z=np.array([indexed[i]['logit_diff_LUSC_minus_LUAD'] for i in ids])
        scores[k,s]=p;models[name]=best
        actual.append(dict(budget=k,seed=s,selected_epoch=best['epoch'],**m.metrics(y,p,z)))
    def compare(actual,filename,keys):
        expected=m.read_csv(out/filename)
        a={tuple(str(r[k]) for k in keys):r for r in actual}
        b={tuple(r[k] for k in keys):r for r in expected}
        require(a.keys()==b.keys(),'Row keys mismatch: '+filename)
        for key,row in a.items():
            for field,value in row.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    require(close(value,b[key][field]),'Value mismatch: '+filename+'/'+field)
    compare(actual,'metrics_by_run.csv',('budget','seed'))
    grouped=m.summarize(actual,m.PLAN['descriptive_metrics'])
    compare(grouped,'metrics_by_budget.csv',('budget',))
    print('PASS: frozen inputs, patient alignment and metrics for 9 x 186 predictions.',flush=True)
    boot=m.bootstrap_auroc(y,scores,m.PLAN['bootstrap']['replicates'],m.PLAN['bootstrap']['seed'])
    pairs=[]
    for k in (16,32):
        delta=np.mean([boot[k,s]-boot[64,s] for s in m.SEEDS],axis=0)
        lo,hi=np.quantile(delta,[.025,.975],method='linear')
        observed=np.mean([next(r['auroc'] for r in actual if r['budget']==k and r['seed']==s)-next(r['auroc'] for r in actual if r['budget']==64 and r['seed']==s) for s in m.SEEDS])
        pairs.append(dict(comparison=f'K{k}-K64',mean_paired_auroc_difference=float(observed),ci95_lower=float(lo),ci95_upper=float(hi)))
    compare(pairs,'paired_auroc_comparisons.csv',('comparison',))
    resources=m.resource_rows({'source':source,'models':models})
    compare(resources,'training_resources_by_run.csv',('budget','seed'))
    print('PASS: 10,000 paired bootstrap replicates and per-run training resource totals.',flush=True)
    print('No original files modified. Model inference, GPU timings and data reconstruction were not rerun.')
if __name__=='__main__':
    try:main()
    except Exception as exc:print('FAIL:',str(exc),file=sys.stderr);sys.exit(1)
