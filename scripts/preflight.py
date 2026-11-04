import json,os,shutil
from pathlib import Path
from exp.baseline.scheduler.monitor import snapshot,permitted
from exp.baseline.runtime.paths import ROOT

def main():
    gpus=snapshot();constraints={}
    for path in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us']:
        if Path(path).exists():constraints[path]=Path(path).read_text().strip()
    print(json.dumps({'gpus':gpus,'authorized_gpu_uuids':[x['uuid'] for x in permitted(gpus)],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'constraints':constraints,'disk_free_gib':shutil.disk_usage(ROOT).free/2**30},indent=2))
if __name__=='__main__':main()
