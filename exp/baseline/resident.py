"""One model/configuration per physical GPU; jobs arrive over a private pipe."""
import argparse,json,sys,time,traceback,os
from pathlib import Path
from .worker import execute,atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state-dir',type=Path,required=True);ap.add_argument('--no-prefetch',action='store_true');a=ap.parse_args()
    a.state_dir.mkdir(parents=True,exist_ok=True);models={};jobs=0
    atomic(a.state_dir/'worker.json',{'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'status':'waiting','jobs':0})
    for line in sys.stdin:
        message=json.loads(line)
        if message.get('stop'):break
        path=Path(message['job']);started=time.time();code=0;error=None
        atomic(a.state_dir/'worker.json',{'pid':os.getpid(),'status':'running','job':str(path),'jobs':jobs,'time':started})
        try:execute(path,resident_models=models,prefetch=not a.no_prefetch)
        except SystemExit as e:code=int(e.code or 0)
        except Exception:code=1;error=traceback.format_exc();print(error,flush=True)
        jobs+=1
        atomic(Path(message['receipt']),{'job':str(path),'pid':os.getpid(),'code':code,'error':error,'started':started,'finished':time.time(),'jobs_in_process':jobs})
        atomic(a.state_dir/'worker.json',{'pid':os.getpid(),'status':'waiting','jobs':jobs,'time':time.time()})
        # Failed CUDA work may leave allocator or hooks in an unknown state.
        if code:raise SystemExit(code)

if __name__=='__main__':main()
