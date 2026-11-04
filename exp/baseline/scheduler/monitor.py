import csv,io,os,subprocess

def snapshot():
    command=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    rows=list(csv.reader(io.StringIO(subprocess.check_output(command,text=True))))
    devices=[dict(zip(('index','uuid','name','memory_total_mib','memory_used_mib','utilization'),[v.strip() for v in r])) for r in rows]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[{'pid':r[0].strip(),'uuid':r[1].strip(),'memory_mib':r[2].strip()} for r in csv.reader(io.StringIO(apps)) if r]
    for gpu in devices:gpu['processes']=[p for p in processes if p['uuid']==gpu['uuid']]
    return devices

def permitted(devices,requested=None):
    inherited=os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed=devices if inherited is None else [d for d in devices if d['index'] in inherited.split(',') or d['uuid'] in inherited.split(',')]
    if requested is not None:
        wanted=requested.split(',');selected=[d for d in allowed if d['index'] in wanted or d['uuid'] in wanted]
        if len(selected)!=len(set(wanted)):raise ValueError('Requested GPU is outside inherited allocation or absent')
        allowed=selected
    return allowed
