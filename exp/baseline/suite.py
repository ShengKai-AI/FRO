"""Resumable full-method campaign: bounded shared inputs, resident GPU consumers."""
import argparse,collections,concurrent.futures,fcntl,hashlib,json,multiprocessing,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
from .runtime.paths import ROOT
from .scheduler.monitor import snapshot,permitted
from .worker import atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,default=ROOT/'exp/baseline/configs/experiment.yaml');ap.add_argument('--dataset',required=True);ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--media-map',type=Path,required=True);ap.add_argument('--run',required=True);ap.add_argument('--cache',type=Path);ap.add_argument('--gpus');ap.add_argument('--window',type=int,default=48);ap.add_argument('--prep-workers',type=int,default=16);ap.add_argument('--batch-pairs',type=int,default=8);ap.add_argument('--reuse-run',type=Path,action='append',default=[]);ap.add_argument('--exclude-reused-methods',nargs='*',default=[]);a=ap.parse_args()
    run=ROOT/'runs'/a.run;cache=a.cache or ROOT/'data/processed'/a.run
    for p in (run,cache,run/'jobs',run/'outputs',run/'logs',run/'resident',run/'retired_inputs'):p.mkdir(parents=True,exist_ok=True)
    lock=(run/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=yaml.safe_load(a.plan.read_text());ph=hashlib.sha256((ROOT/'docs/experiment_plan.pdf').read_bytes()).hexdigest()
    native=plan.get('evaluation_protocol')=='native_two_video_mc'
    if not native:raise ValueError('Only native two-video MC evaluation is supported')
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for folder in ('src','exp') for p in sorted((ROOT/folder).rglob('*.py')))).hexdigest()
    cfg={'stage':'native_pilot' if native else 'all_methods','runtime_plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'dataset':a.dataset,'plan_sha256':ph,'budget_mode':plan['budget_mode'],'pairs_sha256':hashlib.sha256(a.pairs.read_bytes()).hexdigest(),'media_map_sha256':hashlib.sha256(a.media_map.read_bytes()).hexdigest(),'models':plan['models'],'methods':plan['methods'],'source_sha256':source_hash,'reuse_runs':[str(p) for p in a.reuse_run]}
    if (run/'campaign.json').exists() and json.loads((run/'campaign.json').read_text())!=cfg:raise ValueError('Frozen campaign changed; use a new campaign directory')
    atomic(run/'campaign.json',cfg)
    pairs=[p for b in json.loads(a.pairs.read_text()) for p in b['pairs']];lookup={p['id']:p for p in pairs}
    if native and not all(p.get('protocol')=='native_two_video_mc' and len(p['questions'])==1 and len(p['streams'])==2 for p in pairs):raise ValueError('Invalid native two-video manifest')
    if len(lookup)!=len(pairs):raise ValueError('Duplicate pair IDs')
    combinations=[(m,method,r) for method in plan['methods'] for m in plan['models'] for r in ([1.] if method=='full_tokens' else plan['fro_anchor_retentions'])]
    configs=set(combinations);done=set();records={};failed={};blocked={};attempts=collections.Counter();workers={};active={};preparing={};finished_per_gpu=collections.Counter();stopping=False
    compatible=set(json.loads((ROOT/'docs/plan_metadata.json').read_text()).get('compatible_prior_plan_sha256',[]))|{ph}
    def rowkey(row):return (row['pair_id'],row['model'],row['method'],1. if row['method']=='full_tokens' else row.get('fro_anchor_retention'))
    for folder in a.reuse_run+[run]:
        superseded=json.loads((folder/'superseded_methods.json').read_text()) if (folder/'superseded_methods.json').exists() else []
        for path in (folder/'outputs').glob('*/*.json'):
            row=json.loads(path.read_text());key=rowkey(row)
            if row['method'] in superseded:continue
            if folder!=run and row['method'] in a.exclude_reused_methods:continue
            if row.get('status')!='passed' or row.get('plan_sha256') not in compatible or row.get('dataset')!=a.dataset or key[1:] not in configs:continue
            if row.get('pairs_sha256')!=cfg['pairs_sha256']:raise ValueError('Incompatible pair manifest in reused result')
            if row['method']!='full_tokens' and (not row.get('flops',{}).get('total_complete') or abs(row.get('flops_relative_error',1))>.005):raise ValueError('Incomplete prior calibration')
            if key in done:raise ValueError('Duplicate result across supplied runs: '+str(key))
            done.add(key);records[key]=row
    def complete(ident):return all((ident,*c) in done for c in combinations)
    pending=collections.deque(p for p in sorted(pairs,key=lambda p:(not (cache/(p['id']+'.pt')).exists(),p.get('duration_seconds',0),p['id'])) if not complete(p['id']))
    window={};allowed=permitted(snapshot(),a.gpus)
    if not allowed:raise ValueError('No allocated GPUs')
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM'):os.environ[k]=env[k]
    from .native_prepare import initialize,prepare_pair
    from .runtime.cache_lifecycle import retire_cache
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=a.prep_workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.dataset,str(a.pairs),str(a.media_map),str(cache)))
    cleanup=concurrent.futures.ThreadPoolExecutor(max_workers=1);retiring={}
    def close(uuid,kill=False):
        w=workers[uuid];p=w['process']
        if p.poll() is None:
            if kill:os.killpg(p.pid,signal.SIGTERM)
            else:
                try:p.stdin.write('{"stop":true}\n');p.stdin.flush()
                except BrokenPipeError:pass
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        w['log'].close();w['lease'].close();del workers[uuid]
    try:
        while True:
            now=time.time()
            for ident,future in list(preparing.items()):
                if not future.done():continue
                try:result=future.result()
                except Exception as e:result={'ok':False,'error':str(e)}
                del preparing[ident]
                if not result['ok']:blocked[ident]=result;window.pop(ident,None)
                with (run/'preparation.jsonl').open('a') as f:f.write(json.dumps({'pair_id':ident,**result})+'\n')
            for uuid,j in list(active.items()):
                w=workers[uuid];new=0
                for key in j['keys']:
                    path=j['output']/(key[0]+'.json')
                    if key not in done and path.exists():
                        row=json.loads(path.read_text())
                        if row.get('status')=='passed':
                            done.add(key);records[key]=row;finished_per_gpu[uuid]+=1;new+=1
                if new:j['last_progress']=now
                timedout=now-j['last_progress']>1200
                if not j['receipt'].exists() and w['process'].poll() is None and not timedout:continue
                result=json.loads(j['receipt'].read_text()) if j['receipt'].exists() else {'code':w['process'].poll() or 124}
                for key in j['keys']:
                    if key not in done and attempts[key]>=2:failed[key]={'log':w['log'].name,'returncode':result['code'],'timeout':timedout}
                del active[uuid]
                if result['code'] or timedout:close(uuid,kill=timedout)
            for ident in list(window):
                if complete(ident) and ident not in retiring:
                    # Every required model/method/band has consumed this exact cached input.
                    if (cache/(ident+'.pt')).exists():
                        row=records[(ident,*combinations[-1])]
                        retiring[ident]=cleanup.submit(retire_cache,cache/(ident+'.pt'),run/'retired_inputs'/(ident+'.json'),row)
                    else:window.pop(ident)
            for ident,f in list(retiring.items()):
                if f.done():
                    try:f.result()
                    except Exception as e:blocked[ident]={'cache_retirement':str(e)}
                    window.pop(ident,None);del retiring[ident]
            while not stopping and pending and len(window)<a.window:
                p=pending.popleft();window[p['id']]=p
            for ident,p in window.items():
                if stopping or len(preparing)>=a.prep_workers:break
                if ident not in preparing and not (cache/(ident+'.pt')).exists() and ident not in retiring:preparing[ident]=pool.submit(prepare_pair,p)
            running={key for j in active.values() for key in j['keys']};ready=[i for i in window if i not in preparing and i not in retiring and (cache/(i+'.pt')).exists()]
            for gpu_index,gpu in enumerate(allowed):
                uuid=gpu['uuid']
                if stopping or uuid in active:continue
                candidates={c:[i for i in ready if (i,*c) not in done|set(failed)|running] for c in combinations}
                current=workers.get(uuid,{}).get('configuration')
                preferred=plan['models'][gpu_index%len(plan['models'])]
                ordered=[c for c in combinations if c[0]==preferred]+[c for c in combinations if c[0]!=preferred]
                c=current if candidates.get(current) else next((c for c in ordered if candidates[c]),None)
                if c is None:continue
                if uuid in workers and current!=c:close(uuid)
                if uuid not in workers:
                    actual=next(g for g in snapshot() if g['uuid']==uuid)
                    if actual['processes'] or int(actual['memory_used_mib'])>64:continue
                    directory=Path('/tmp/fro-gpu-leases');directory.mkdir(exist_ok=True);lease=(directory/(uuid+'.lock')).open('a')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    log=(run/'logs'/('resident_'+uuid+'.log')).open('a');proc=subprocess.Popen([sys.executable,'-u','-m','exp.baseline.resident','--state-dir',str(run/'resident'/uuid)],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=uuid),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
                    workers[uuid]={'process':proc,'configuration':c,'log':log,'lease':lease}
                batch=candidates[c][:a.batch_pairs];keys=[(i,*c) for i in batch]
                for key in keys:attempts[key]+=1
                ident=hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:20];jobpath=run/'jobs'/(ident+'.json');output=run/'outputs'/ident;receipt=run/'resident'/(ident+'.receipt.json');receipt.unlink(missing_ok=True)
                job={'model':c[0],'method':c[1],'ratio':c[2],'fro_anchor_retention':None if c[1]=='full_tokens' else c[2],'budget_mode':plan['budget_mode'],'plan_sha256':ph,'pairs':str(a.pairs),'cache':str(cache),'pair_ids':batch,'output':str(output),'max_new_tokens':plan['decode']['max_new_tokens'],'flops_decode_tokens':plan.get('flops_decode_tokens',100),'evaluation_protocol':plan.get('evaluation_protocol','paired_questions')}
                atomic(jobpath,job);w=workers[uuid];w['process'].stdin.write(json.dumps({'job':str(jobpath),'receipt':str(receipt)})+'\n');w['process'].stdin.flush()
                active[uuid]={'keys':keys,'output':output,'receipt':receipt,'last_progress':now};running.update(keys)
            counts=collections.Counter('|'.join(map(str,k[1:])) for k in done);devices=snapshot()
            progress={'time':time.time(),'config':cfg,'controller_pid':os.getpid(),'total_pairs':len(pairs),'total_evaluations':len(pairs)*len(combinations),'passed':len(done),'completed_by_configuration':dict(counts),'completed_per_gpu':dict(finished_per_gpu),'window':len(window),'pending_inputs':len(pending),'preparing':len(preparing),'failed':{'|'.join(map(str,k)):v for k,v in failed.items()},'blocked_inputs':blocked,'gpus':devices,'active':{u:{'pid':workers[u]['process'].pid,'configuration':workers[u]['configuration'],'pairs':[k[0] for k in j['keys']],'log':workers[u]['log'].name} for u,j in active.items()}}
            atomic(run/'progress.json',progress)
            with (run/'monitor.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            if len(done)==len(pairs)*len(combinations):break
            if stopping and not active and not preparing:break
            runnable=any((i,*c) not in done|set(failed) for i in ready for c in combinations)
            if not active and not preparing and not runnable and not retiring:
                # Drop exhausted windows, preserving explicit failure denominator.
                window.clear()
                if not pending:break
            time.sleep(3)
    finally:
        pool.shutdown(wait=True,cancel_futures=True);cleanup.shutdown(wait=True)
        for uuid in list(workers):close(uuid)
    atomic(run/'suite_result.json',{'complete':len(done)==len(pairs)*len(combinations),'passed':len(done),'expected':len(pairs)*len(combinations),'stopped':stopping,'failed':len(failed),'blocked':blocked})
    if len(done)!=len(pairs)*len(combinations):raise SystemExit(1)
if __name__=='__main__':main()
