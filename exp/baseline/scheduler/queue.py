"""Ready-input queue with one physical GPU per worker and atomic shard outputs."""
import collections,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .monitor import snapshot,permitted
from exp.baseline.runtime.paths import ROOT

def run_queue(jobs,run_dir,gpus=None,poll_seconds=15):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    lock=(run_dir/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=collections.deque(jobs);active={};failures=[];done=collections.Counter();attempts=collections.Counter();leases={}
    allowed=permitted(snapshot(),gpus)
    if not allowed:raise ValueError('No GPU in authorized allocation')
    # Only ready jobs enter the execution slots. Download/preprocessing does not hold a GPU.
    for path in jobs:
        job=json.loads(Path(path).read_text())
        missing=[p for p in job['pair_ids'] if not (Path(job['cache'])/(p+'.pt')).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} unprepared pairs in {path}; prepare inputs first')
    try:
        while queue or active:
            devices={d['uuid']:d for d in snapshot()}
            for uuid,(process,path,log) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();del active[uuid]
                fcntl.flock(leases.pop(uuid),fcntl.LOCK_UN)
                if code==0:done[uuid]+=1
                elif attempts[str(path)]<2:queue.append(path)
                else:failures.append({'job':str(path),'exit_code':code})
            for gpu in allowed:
                uuid=gpu['uuid'];current=devices.get(uuid)
                if not queue or uuid in active or not current:continue
                if current['processes'] or int(current['memory_used_mib'])>64:continue
                lease_dir=Path('/tmp/fro-gpu-leases');lease_dir.mkdir(exist_ok=True)
                lease=(lease_dir/(uuid+'.lock')).open('a')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                leases[uuid]=lease
                path=Path(queue.popleft());attempts[str(path)]+=1
                logpath=run_dir/(path.stem+f'.attempt{attempts[str(path)]}.log');log=logpath.open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),OMP_NUM_THREADS='4')
                process=subprocess.Popen([sys.executable,'-m','exp.baseline.worker','--job',str(path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[uuid]=(process,path,log)
            event={'time':time.time(),'gpus':list(devices.values()),'active':{u:{'pid':p.pid,'job':str(j),'log':f.name} for u,(p,j,f) in active.items()},'completed_shards':dict(done),'queued':len(queue),'failed':failures}
            with (run_dir/'monitor.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps({k:event[k] for k in ('active','completed_shards','queued','failed')}),flush=True)
            if queue or active:time.sleep(poll_seconds)
    except BaseException:
        for process,_,_ in active.values():
            if process.poll() is None:process.terminate()
        raise
    finally:
        for lease in leases.values():lease.close()
    (run_dir/'scheduler_result.json').write_text(json.dumps({'completed':dict(done),'failed':failures},indent=2))
    if failures:raise SystemExit(1)
