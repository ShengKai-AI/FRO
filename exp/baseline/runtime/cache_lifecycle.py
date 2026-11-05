"""Retire derived inputs after every model/method consumer finishes."""
import fcntl,hashlib,json,time
from pathlib import Path
from exp.baseline.worker import atomic

def retire_cache(path,receipt,row):
    """Only derived inputs consumed by both models; retain exact sampling metadata."""
    import torch
    lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        if not path.exists():return 0
        payload=torch.load(path,map_location='cpu',weights_only=False)
        with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum!=row['cache_sha256']:raise ValueError('Completed input cache changed')
        record={'pair_id':payload['pair_id'],'protocol':payload['protocol'],'stream_records':[s['record'] for s in payload['streams']],
                'cache_sha256':checksum,'source_sha256':row['source_sha256'],'reason':'all configured consumers passed; original media and results preserved','time':time.time()}
        atomic(receipt,record);size=path.stat().st_size
        if path.is_symlink():
            target=path.resolve()
            if not target.is_relative_to(Path('/dev/shm/fro-pair-cache')):raise ValueError('Unexpected cache symlink target')
            path.unlink();target.unlink()
        else:path.unlink()
        return size
    finally:lock.close()
