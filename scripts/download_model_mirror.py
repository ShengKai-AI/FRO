"""Download weight bytes from a mirror, verified against the pinned HF manifest."""
import argparse,concurrent.futures,hashlib,json,time,urllib.parse,urllib.request
from pathlib import Path
from exp.baseline.runtime.paths import ROOT,asset

def main():
    ap=argparse.ArgumentParser();ap.add_argument('name');a=ap.parse_args()
    entry=json.loads((ROOT/'manifests/models.yaml').read_text())[a.name]
    target=asset('weights')/('Qwen2.5-Omni-'+a.name);target.mkdir(parents=True,exist_ok=True)
    base='https://modelscope.cn/api/v1/models/'+entry['repo_id']
    listing=json.load(urllib.request.urlopen(base+'/repo/files?Revision=master&Recursive=true',timeout=60))
    sources={f['Path']:f for f in listing['Data']['Files']}
    files=[f for f in entry['files'] if Path(f['path']).suffix in ('.json','.txt','.safetensors')]
    for f in files:
        s=sources[f['path']]
        if s['Size']!=f['size'] or (f.get('sha256') and f['sha256']!=s['Sha256']):raise ValueError('Mirror differs from pinned HF asset: '+f['path'])
    def one(f):
        path=target/f['path'];part=path.with_suffix(path.suffix+'.mirror-partial');s=sources[f['path']]
        expected=f.get('sha256') or s['Sha256']
        if path.exists():
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size==f['size'] and actual==expected:return
            raise ValueError('Existing asset differs: '+str(path))
        url=base+'/repo?'+urllib.parse.urlencode({'Revision':s['Revision'],'FilePath':f['path']})
        for attempt in range(8):
            offset=part.stat().st_size if part.exists() else 0
            if offset==f['size']:break
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={offset}-'}),timeout=60) as response:
                    if offset and response.status!=206:raise ValueError('Server ignored resume Range')
                    with part.open('ab') as out:
                        last=offset
                        while chunk:=response.read(4*1024**2):
                            out.write(chunk)
                            if out.tell()-last>=256*1024**2:
                                last=out.tell();print(f['path'],last,'/',f['size'],flush=True)
                break
            except Exception as error:
                print(f['path'],'retry',attempt,type(error).__name__,flush=True)
                if attempt==7:raise
                time.sleep(3)
        with part.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if part.stat().st_size!=f['size'] or actual!=expected:raise ValueError('Downloaded hash/size mismatch')
        part.replace(path);print('VERIFIED',f['path'],flush=True)
    # Publish the index last, so the scheduler cannot observe incomplete weights.
    index=[f for f in files if f['path']=='model.safetensors.index.json']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,[f for f in files if f not in index]))
    for f in index:one(f)
    (target/'mirror_receipt.json').write_text(json.dumps({'hf_revision':entry['revision'],'sources':sources,'verified':True},indent=2))

if __name__=='__main__':main()
