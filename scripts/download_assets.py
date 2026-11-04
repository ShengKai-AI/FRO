"""Pinned Hugging Face downloads with resume, space checks and receipts."""
import argparse,fnmatch,hashlib,json,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from exp.baseline.runtime.paths import ROOT,asset

def download(kind,name,patterns,execute):
    manifest=json.loads((ROOT/'manifests'/(kind+'.yaml')).read_text());entry=manifest[name]
    if not entry.get('revision'):raise ValueError('Source revision unresolved: '+name)
    files=[f for f in entry['files'] if any(fnmatch.fnmatch(f['path'],p) for p in patterns)]
    if not files:raise ValueError('No matching assets')
    destination=asset('weights')/('Qwen2.5-Omni-'+name) if kind=='models' else asset('data')/'raw'/name
    destination.mkdir(parents=True,exist_ok=True)
    missing=sum(f.get('size') or 0 for f in files if not (destination/f['path']).exists())
    plan={'repo_id':entry['repo_id'],'revision':entry['revision'],'destination':str(destination),'files':len(files),'missing_bytes':missing,'execute':execute}
    print(json.dumps(plan))
    if not execute:return
    if shutil.disk_usage(destination).free<missing+2*2**30:raise OSError('Insufficient disk space; complete/archive the current dataset first')
    records=[]
    for f in files:
        path=Path(hf_hub_download(repo_id=entry['repo_id'],filename=f['path'],revision=entry['revision'],repo_type=entry['repo_type'],local_dir=destination))
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if f.get('sha256') and sha!=f['sha256']:raise ValueError('SHA256 mismatch: '+str(path))
        if f.get('size') is not None and path.stat().st_size!=f['size']:raise ValueError('Size mismatch: '+str(path))
        records.append({'path':f['path'],'sha256':sha,'bytes':path.stat().st_size})
        (destination/'download_receipt.json').write_text(json.dumps({**plan,'completed':records},indent=2))

def main(kind):
    ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--patterns',nargs='+',default=['*.json','*.parquet'] if kind=='datasets' else ['*.json','*.safetensors','*.txt','*.model','*.tiktoken']);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    download(kind,a.name,a.patterns,a.execute)
