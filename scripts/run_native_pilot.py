"""Sequential datasets, four-GPU ready queues, resumable baseline-only pilot."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
from exp.baseline.runtime.paths import ROOT
from exp.baseline.worker import atomic
from scripts.native_pilot_results import score

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='native-baseline-pilot-20261105');ap.add_argument('--window',type=int,default=200);ap.add_argument('--prep-workers',type=int,default=24);ap.add_argument('--batch-pairs',type=int,default=16);ap.add_argument('--scheduler-file',type=Path);a=ap.parse_args()
    parent=ROOT/'runs'/a.run;parent.mkdir(parents=True,exist_ok=True)
    controller_lock=(parent/'controller.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=ROOT/'exp/baseline/configs/native_pilot.yaml';cfg=yaml.safe_load(plan.read_text())
    assert 'fro' not in cfg['methods']
    child=None;stopping=False;results={}
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT),FRO_FFMPEG=os.environ.get('FRO_FFMPEG','/root/seamless-media-tools/bin/ffmpeg'),FRO_FFPROBE=os.environ.get('FRO_FFPROBE','/root/seamless-media-tools/bin/ffprobe'))
    atomic(parent/'campaign.json',{'datasets':cfg['datasets'],'models':cfg['models'],'methods':cfg['methods'],'max_questions_per_dataset':200,'budget_anchors':cfg['fro_anchor_retentions'],'source_root':str(ROOT),'controller_pid':os.getpid(),'started':time.time()})
    for name in cfg['datasets']:
        if stopping:break
        folder=ROOT/'data/native_pilot'/name;questions=json.loads((folder/'questions.json').read_text())[0]['pairs']
        assert len(questions)<=200 and all(q['protocol']=='native_two_video_mc' for q in questions)
        run_name=a.run+'-'+name;run=ROOT/'runs'/run_name;log=(parent/(name+'.log')).open('a')
        cache=Path('/dev/shm/fro-native-pair-cache')/run_name
        cmd=[sys.executable,'-u','-m','exp.baseline.suite','--plan',str(plan),'--dataset',name,'--pairs',str(folder/'questions.json'),'--media-map',str(folder/'media_map.json'),'--run',run_name,'--cache',str(cache),'--window',str(a.window),'--prep-workers',str(a.prep_workers),'--batch-pairs',str(a.batch_pairs)]
        if a.scheduler_file:
            # Explicit orchestration overlay: imports still resolve to the frozen source.
            cmd[2:4]=[str(ROOT/'scripts/launch_native_scheduler.py'),str(a.scheduler_file.resolve())]
        for attempt in range(2):
            if stopping:break
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                score(run,folder/'gold.json',folder/'questions.json')
                atomic(parent/'status.json',{'state':'stopping' if stopping else 'running','dataset':name,'dataset_run':str(run),'controller_pid':os.getpid(),'child_pid':child.pid,'attempt':attempt+1,'results':results,'time':time.time()})
                time.sleep(15)
            if child.returncode==0 or stopping:break
        score(run,folder/'gold.json',folder/'questions.json');log.close()
        result=json.loads((run/'suite_result.json').read_text()) if (run/'suite_result.json').exists() else {'complete':False,'returncode':child.returncode}
        results[name]=result
        atomic(parent/'status.json',{'state':'stopped' if stopping else 'between_datasets','results':results,'time':time.time()})
    atomic(parent/'status.json',{'state':'stopped' if stopping else 'complete' if len(results)==len(cfg['datasets']) and all(x.get('complete') for x in results.values()) else 'incomplete','results':results,'time':time.time()})
if __name__=='__main__':main()
