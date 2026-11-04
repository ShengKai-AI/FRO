import argparse,hashlib,json
from pathlib import Path
import yaml
from .runtime.paths import ROOT,asset

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--run',required=True)
    ap.add_argument('--models',nargs='+');ap.add_argument('--methods',nargs='+');ap.add_argument('--ratios',type=float,nargs='+')
    ap.add_argument('--pairs',type=Path);ap.add_argument('--cache',type=Path);ap.add_argument('--pair-ids',nargs='+')
    ap.add_argument('--gpus');ap.add_argument('--execute',action='store_true');ap.add_argument('--shard-size',type=int,default=8);a=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'exp/baseline/configs/experiment.yaml').read_text())
    if a.dataset not in cfg['datasets']:raise ValueError('Dataset is not in experiment plan')
    models=a.models or cfg['models'];methods=a.methods or ['full_tokens'];ratios=a.ratios or cfg['fro_anchor_retentions']
    if not set(models)<=set(cfg['models']) or not set(methods)<=set(cfg['methods']):raise ValueError('Unknown model/method')
    if not all(0<r<=1 for r in ratios) or a.shard_size<1:raise ValueError('Invalid ratio/shard size')
    pairs=(a.pairs or asset('data')/'pairs'/(a.dataset+'.json')).resolve()
    bundles=json.loads(pairs.read_text());ids=[q['id'] for b in bundles for q in b['pairs']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate pair IDs')
    if a.pair_ids:
        if not set(a.pair_ids)<=set(ids):raise ValueError('Unknown requested pair ID')
        ids=[i for i in ids if i in a.pair_ids]
    folder=asset('runs')/a.run;folder.mkdir(parents=True,exist_ok=True);jobs=[]
    cache=(a.cache or asset('data')/'processed'/a.dataset).resolve()
    for model in models:
        for method in methods:
            for ratio in ([1.] if method=='full_tokens' else ratios):
                from .metrics.flops import solve_reference_retention
                model_config=json.loads((ROOT/'exp/baseline/configs/models'/(model+'.json')).read_text())['thinker_config']['text_config']
                reference_ratio,reference_target=solve_reference_retention(model_config,method,ratio)
                for offset in range(0,len(ids),a.shard_size):
                    identity=f'{a.dataset}_{model}_{method}_{ratio}_{offset:06d}'
                    job={'model':model,'method':method,'ratio':reference_ratio,'fro_anchor_retention':None if method=='full_tokens' else ratio,'reference_target_flops':reference_target,'actual_input_calibration_complete':method=='full_tokens','pairs':str(pairs),'cache':str(cache),'pair_ids':ids[offset:offset+a.shard_size],
                         'output':str(folder/'outputs'/identity),'budget_mode':cfg['budget_mode'],'plan_sha256':hashlib.sha256((ROOT/'docs/experiment_plan.pdf').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'exp/baseline/configs/experiment.yaml').read_bytes()).hexdigest()}
                    path=folder/(identity+'.json');text=json.dumps(job,indent=2)
                    if path.exists() and path.read_text()!=text:raise ValueError('Run name already used for a different plan')
                    path.write_text(text);jobs.append(path)
    print(json.dumps({'jobs':len(jobs),'pairs':len(ids),'run':str(folder),'execute':a.execute}))
    if a.execute and any(m!='full_tokens' for m in methods):raise ValueError('Complete actual-input auxiliary-cost calibration before compressed main runs')
    if a.execute:
        from .scheduler.queue import run_queue
        run_queue(jobs,folder,a.gpus,cfg['scheduler']['poll_seconds'])
if __name__=='__main__':main()
