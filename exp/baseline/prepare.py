import argparse,json,hashlib,fcntl
from pathlib import Path
import torch,yaml
from transformers import Qwen2_5OmniProcessor
from .runtime.paths import ROOT,asset,model_path
from .runtime.preprocessing import prepare_stream
from .runtime.pairing import assemble

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--media-map',type=Path,required=True)
    ap.add_argument('--pair-ids',nargs='+');ap.add_argument('--output',type=Path);ap.add_argument('--validation-frame-cap',type=int);a=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'exp/baseline/configs/datasets'/(a.dataset+'.yaml')).read_text())
    if a.validation_frame_cap is not None and a.validation_frame_cap<2:raise ValueError('Frame cap must be >=2')
    media=json.loads(a.media_map.read_text());bundles=json.loads(a.pairs.read_text());folder=a.output or asset('data')/'processed'/a.dataset;folder.mkdir(parents=True,exist_ok=True)
    processor=Qwen2_5OmniProcessor.from_pretrained(model_path('7B'),local_files_only=True,use_fast=False)
    for bundle in bundles:
        for q in bundle['pairs']:
            if a.pair_ids and q['id'] not in a.pair_ids:continue
            path=folder/(q['id']+'.pt');lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                expected={'pair_sha256':hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),'media_map_sha256':hashlib.sha256(a.media_map.read_bytes()).hexdigest(),'validation_frame_cap':a.validation_frame_cap}
                if any(old['protocol'].get(k)!=v for k,v in expected.items()):raise ValueError('Cache configuration mismatch; use another output directory')
                lock.close();continue
            cap=a.validation_frame_cap or cfg['max_frames']
            while True:
                streams=[]
                for index,stream in enumerate(q['streams']):
                    spec=media[stream['video_id']];spec={'path':spec} if isinstance(spec,str) else spec
                    p=prepare_stream(processor,spec['path'],q['question'],cap,start=stream.get('start',spec.get('start')),end=stream.get('end',spec.get('end')),audio_path=spec.get('audio_path'))
                    p['record']['source']=index+1;streams.append(p)
                joint=assemble(processor,streams,q)
                if joint['joint']['input_ids'].numel()<=32256:break
                if cap<=2:raise ValueError('Pair exceeds budget even at 2 frames; audio resampling adaptation required')
                cap=max(2,2*int(cap*.75/2))
            protocol={'version':'v4','validation_frame_cap':a.validation_frame_cap,'frame_cap_used':cap,'pair_sha256':hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),
                      'media_map_sha256':hashlib.sha256(a.media_map.read_bytes()).hexdigest(),'max_pair_tokens':32256,'fps_candidate':2,'max_audio_seconds':300,'subtitles':False}
            payload={'pair_id':q['id'],'streams':streams,'protocol':protocol}
            tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path);lock.close()
            print(q['id'],joint['joint']['input_ids'].numel(),flush=True)
if __name__=='__main__':main()
