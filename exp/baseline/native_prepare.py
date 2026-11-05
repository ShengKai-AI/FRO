"""Prepare one native cross-video question with shared, cached visual inputs."""
import fcntl,hashlib,json,time,traceback
from pathlib import Path
import torch
from transformers import Qwen2_5OmniProcessor
from .runtime.paths import ROOT,model_path
from .runtime.stream_cache import StreamCache
from .runtime.native_prompt import assemble
_CONTEXT=None

def initialize(dataset,pairs_path,media_map_path,folder):
    global _CONTEXT
    torch.set_num_threads(2)
    cfg=next(d for d in json.loads((ROOT/'docs/native_dual_video_parameters.json').read_text())['datasets'] if d['name']==dataset)
    media_path=Path(media_map_path)
    _CONTEXT={'p':Qwen2_5OmniProcessor.from_pretrained(model_path('7B'),local_files_only=True,use_fast=False),'cfg':cfg,'media':json.loads(media_path.read_text()),'media_hash':hashlib.sha256(media_path.read_bytes()).hexdigest(),'folder':Path(folder),'cache':StreamCache(Path('/dev/shm/fro-native-stream-cache')/dataset,max_gib=24)}
    Path(folder).mkdir(parents=True,exist_ok=True)

def prepare_pair(q):
    started=time.perf_counter()
    try:
        c=_CONTEXT;path=c['folder']/(q['id']+'.pt');lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        signature=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=False)
            assert old['protocol']['pair_sha256']==signature
            return {'ok':True,'pair_id':q['id'],'cached':True}
        cfg=c['cfg'];scale=1.
        for attempt in range(8):
            streams=[]
            for i,s in enumerate(q['streams']):
                spec=c['media'][s['video_id']]
                value=c['cache'].get(c['p'],spec['path'],q['question'],cfg['frames_per_stream'],start=s.get('start'),end=s.get('end'),audio_enabled=False,fixed_frames=cfg['sampling']=='uniform_full_valid_interval',native_frames=cfg['sampling']=='all_native_frames_up_to_cap_then_uniform',max_pixels=cfg.get('max_pixels',100352),long_edge=cfg.get('long_edge'),sampling_scale=scale)
                value['record']['source']=i+1;value['record']['timeline_version']=1
                streams.append(value)
            joint=assemble(c['p'],streams,q);tokens=joint['joint']['input_ids'].numel()
            if tokens<=32256:break
            scale*=min(.9,31000/tokens)
        else:raise ValueError('Native input exceeds shared context capacity')
        protocol={'version':'v6_native_two_video_single_choice','pair_sha256':signature,'media_map_sha256':c['media_hash'],'frame_cap_used':cfg['frames_per_stream'],'sampling_scale':scale,'dataset_parameters':cfg,'max_pair_tokens':32256,'timeline_version':1,'audio_enabled':False,'max_new_tokens':64,'flops_decode_tokens':64}
        payload={'pair_id':q['id'],'streams':streams,'protocol':protocol}
        tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)
        return {'ok':True,'pair_id':q['id'],'cached':False,'tokens':tokens,'seconds':time.perf_counter()-started}
    except Exception:return {'ok':False,'pair_id':q['id'],'error':traceback.format_exc(),'seconds':time.perf_counter()-started}
