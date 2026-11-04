"""Start the v5 Full Tokens stage over a complete frozen dataset manifest.

CPU preparation and GPU inference have separate slots. Unimplemented input cases
remain explicit blocked pairs; they are never removed from the denominator.
Compressed stages remain gated on actual-input auxiliary-cost calibration.
"""
import argparse,collections,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from .runtime.paths import ROOT,model_path
from .scheduler.monitor import snapshot,permitted

def atomic(path,data):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def model_ready(name):
    path=model_path(name);index=path/'model.safetensors.index.json'
    if not index.exists():return False
    try:return all((path/f).is_file() for f in set(json.loads(index.read_text())['weight_map'].values()))
    except (OSError,ValueError):return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--media-map',type=Path,required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--prep-workers',type=int,default=6);ap.add_argument('--gpus');ap.add_argument('--models',nargs='+',default=['7B','3B']);a=ap.parse_args()
    run=ROOT/'runs'/a.run;run.mkdir(parents=True,exist_ok=True);cache=ROOT/'data/processed'/a.run;cache.mkdir(parents=True,exist_ok=True)
    (run/'jobs').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    lock=(run/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bundles=json.loads(a.pairs.read_text());media=json.loads(a.media_map.read_text());pairs=[p for b in bundles for p in b['pairs']]
    if len(pairs)!=len({p['id'] for p in pairs}):raise ValueError('Duplicate pair IDs')
    scored=[q['id'] for p in pairs for q,m in zip(p['questions'],p['score_mask']) if m]
    if len(scored)!=len(set(scored)):raise ValueError('Duplicate scored question IDs')
    # Label-blind length order makes ready short inputs available while long adapters await repair.
    def duration(p):return sum(media[s['video_id']].get('duration_seconds',100000) for s in p['streams'])
    pairs.sort(key=lambda p:(duration(p),p['id']))
    config={'pairs_sha256':hashlib.sha256(a.pairs.read_bytes()).hexdigest(),'media_map_sha256':hashlib.sha256(a.media_map.read_bytes()).hexdigest(),
            'plan_sha256':hashlib.sha256((ROOT/'docs/experiment_plan.pdf').read_bytes()).hexdigest(),'models':a.models,
            'source_pairs':len(pairs),'source_questions':len(scored),'stage':'full_tokens','budget_mode':'matched_flops_dagger','validation_frame_cap':None}
    conf=run/'campaign.json'
    if conf.exists() and json.loads(conf.read_text())!=config:raise ValueError('Campaign config changed; use a new name')
    atomic(conf,config)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    gpu_allow=permitted(snapshot(),a.gpus)
    if not gpu_allow:raise ValueError('No allocated GPU')
    prep=collections.deque(pairs);preparing={};gpu_active={};leases={};ready=[];blocked={};done=set();failed={};attempts=collections.Counter()
    statefile=run/'progress.json'
    if statefile.exists():
        old=json.loads(statefile.read_text());blocked=old.get('blocked_preparation',{});failed=old.get('failed_inference',{})
    donefile=run/'done.json'
    if donefile.exists():done=set(json.loads(donefile.read_text()))
    # Recover atomic completions written just before a supervisor restart.
    for output in (run/'outputs').glob('*/*.json'):
        result=json.loads(output.read_text())
        if result.get('status')=='passed' and result.get('plan_sha256')==config['plan_sha256']:
            done.add(result['pair_id']+'_'+result['model'])
    completed_gpu=collections.Counter();validated=bool(done);single_progress=0;previous_counts={}
    for p in pairs:
        if (cache/(p['id']+'.pt')).exists():ready.append(p)
    prep=collections.deque(p for p in prep if p['id'] not in blocked and not (cache/(p['id']+'.pt')).exists())
    try:
        while prep or preparing or ready or gpu_active:
            for ident,(proc,p,log) in list(preparing.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del preparing[ident]
                if code==0 and (cache/(ident+'.pt')).exists():ready.append(p)
                else:
                    blocked[ident]={'returncode':code,'log':str(Path(log.name).relative_to(ROOT)),'reason_tail':Path(log.name).read_text(errors='replace')[-1400:]}
            for gpu,(proc,key,log) in list(gpu_active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del gpu_active[gpu];leases.pop(gpu).close()
                if code==0:
                    done.add(key);completed_gpu[gpu]+=1;validated=True;atomic(donefile,sorted(done))
                elif attempts[key]>=2:
                    failed[key]={'returncode':code,'log':str(Path(log.name).relative_to(ROOT))}
            # Keep prepared inputs for all later methods; a disk guard stops new cache growth.
            free=shutil.disk_usage(cache).free
            ready_models=[m for m in a.models if model_ready(m)]
            runnable_backlog=sum(any(p['id']+'_'+m not in done|set(failed) for m in ready_models) for p in ready)
            while prep and len(preparing)<a.prep_workers and runnable_backlog+len(preparing)<24 and free>40*2**30:
                p=prep.popleft();ident=p['id'];log=(run/'logs'/('prepare_'+ident+'.log')).open('a')
                proc=subprocess.Popen([sys.executable,'-m','exp.baseline.prepare','--dataset',p['dataset'],'--pairs',str(a.pairs),'--media-map',str(a.media_map),'--pair-ids',ident,'--output',str(cache)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                preparing[ident]=(proc,p,log)
            devices={g['uuid']:g for g in snapshot()};available_models=[m for m in a.models if model_ready(m)]
            running_keys={x[1] for x in gpu_active.values()}
            for gpu in gpu_allow:
                uuid=gpu['uuid'];g=devices[uuid]
                if uuid in gpu_active or g['processes'] or int(g['memory_used_mib'])>64:continue
                if not validated and gpu_active:break  # one short launch validation, then all cards immediately
                pending=next(((p,m,p['id']+'_'+m) for p in ready for m in available_models if p['id']+'_'+m not in done|set(failed)|running_keys),None)
                if not pending:break
                p,m,key=pending
                lease_dir=Path('/tmp/fro-gpu-leases');lease_dir.mkdir(exist_ok=True);lease=(lease_dir/(uuid+'.lock')).open('a')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                leases[uuid]=lease;attempts[key]+=1
                job={'method':'full_tokens','model':m,'ratio':1.,'budget_mode':'matched_flops_dagger','plan_sha256':config['plan_sha256'],
                     'pairs':str(a.pairs),'cache':str(cache),'pair_ids':[p['id']],'output':str(run/'outputs'/key)}
                jobpath=run/'jobs'/(key+'.json');atomic(jobpath,job)
                log=(run/'logs'/(key+f'.attempt{attempts[key]}.log')).open('a')
                proc=subprocess.Popen([sys.executable,'-m','exp.baseline.worker','--job',str(jobpath)],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT)
                gpu_active[uuid]=(proc,key,log);running_keys.add(key)
            ready=[p for p in ready if any(p['id']+'_'+m not in done|set(failed) for m in a.models)]
            progress={'time':time.time(),'config':config,'prepared_waiting':len(ready),'preparing':len(preparing),'unprepared':len(prep),
                      'passed_pairs_by_model':dict(collections.Counter(k.rsplit('_',1)[1] for k in done)),
                      'completed_per_gpu':dict(completed_gpu),'failed_inference':failed,'blocked_preparation':blocked,
                      'waiting_models':[m for m in a.models if m not in available_models],
                      'active':{u:{'pid':p.pid,'job':k,'log':str(Path(f.name).relative_to(ROOT))} for u,(p,k,f) in gpu_active.items()},
                      'gpus':list(devices.values()),'disk_free_gib':free/2**30,'single_card_reason':None if validated else 'initial real-input validation',
                      'compressed_stage':'blocked_pending_complete_per_input_auxiliary_cost_calibration'}
            atomic(statefile,progress)
            with (run/'monitor.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in progress.items() if k not in ('blocked_preparation','failed_inference')})+'\n')
            print(json.dumps({k:progress[k] for k in ('passed_pairs_by_model','preparing','unprepared','prepared_waiting','active','waiting_models')}),flush=True)
            if not gpu_active and not preparing and not ready and blocked and not prep:break
            if free<=40*2**30 and not gpu_active and not preparing:raise OSError('Cache disk guard reached; inputs retained')
            if prep or preparing or ready or gpu_active:time.sleep(5)
    finally:
        for proc,_,log in list(preparing.values())+list(gpu_active.values()):
            if proc.poll() is None:proc.terminate()
            log.close()
        for lease in leases.values():lease.close()
    atomic(run/'stage_result.json',{'complete':len(done)==len(pairs)*len(a.models),'passed':len(done),'blocked_pairs':len(blocked),'failed_jobs':len(failed)})
if __name__=='__main__':main()
