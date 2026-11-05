"""Question-independent processor outputs shared by CPU workers and both models."""
import fcntl,hashlib,json,os,shutil,time
from pathlib import Path
import torch
from .preprocessing import prepare_stream
from .paths import model_path

class StreamCache:
    def __init__(self,folder,max_gib=32):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.limit=max_gib*2**30
        root=model_path('7B')
        parts=[(root/n).read_bytes() for n in ('preprocessor_config.json','tokenizer_config.json','special_tokens_map.json')]
        parts += [Path(__file__).with_name(n).read_bytes() for n in ('preprocessing.py','timeline.py')]
        self.version=hashlib.sha256(b''.join(parts)).hexdigest()
    @staticmethod
    def identity(path):
        p=Path(path).resolve();s=p.stat();return [str(p),s.st_size,s.st_mtime_ns]
    def get(self,processor,source,question,cap,**kwargs):
        identity={'version':self.version,'source':self.identity(source),'cap':cap,'kwargs':kwargs}
        if kwargs.get('audio_path'):identity['audio_source']=self.identity(kwargs['audio_path'])
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=self.folder/(key+'.pt')
        started=time.perf_counter();lock=(self.folder/(key+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            hit=path.exists()
            if hit:value=torch.load(path,map_location='cpu',weights_only=False);os.utime(path,None)
            else:
                value=prepare_stream(processor,source,'Describe the video.',cap,**kwargs)
                if shutil.disk_usage(self.folder).free>4*2**30:
                    tmp=path.with_suffix('.partial');torch.save(value,tmp);tmp.replace(path)
            value=dict(value,record=dict(value['record'],question=question,stream_cache_key=key,stream_cache_hit=hit))
        finally:lock.close()
        event={'time':time.time(),'pid':os.getpid(),'key':key,'hit':hit,'seconds':time.perf_counter()-started}
        with (self.folder/'events.jsonl').open('a') as out:
            fcntl.flock(out,fcntl.LOCK_EX);out.write(json.dumps(event)+'\n')
        self.trim(exclude=key)
        return value
    def trim(self,exclude):
        guard=(self.folder/'eviction.lock').open('a')
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:guard.close();return
        try:
            entries=[]
            for p in self.folder.glob('*.pt'):
                try:s=p.stat();entries.append((s.st_mtime,s.st_size,p))
                except FileNotFoundError:pass
            total=sum(x[1] for x in entries)
            for _,size,path in sorted(entries):
                if total<=self.limit:break
                if path.stem==exclude:continue
                lock=path.with_suffix('.lock').open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lock.close();continue
                try:path.unlink(missing_ok=True);total-=size
                finally:lock.close()
        finally:guard.close()
