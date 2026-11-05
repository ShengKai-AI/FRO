"""Freeze at most 200 native MC questions per ready dataset, without consulting labels."""
import argparse,collections,concurrent.futures,hashlib,json,re,subprocess
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image
from exp.baseline.datasets.native_two_video import BASE,ROOT,atomic
from exp.baseline.runtime.preprocessing import tool

def probe(path):
    p=Path(path)
    if p.is_dir():return {'duration':None,'fps':None,'frames':len(list(p.glob('*.jpg')))}
    d=json.loads(subprocess.check_output([tool('ffprobe'),'-v','error','-show_streams','-show_format','-of','json',str(p)]))
    v=next(s for s in d['streams'] if s['codec_type']=='video')
    return {'duration':float(d['format']['duration']),'fps':float(Fraction(v['avg_frame_rate'])),'frames':int(v.get('nb_frames',0))}

def build(name,limit=200,seed=20261031):
    qs=json.loads((BASE/name/'questions.json').read_text());groups=collections.defaultdict(list)
    for q in qs:groups[q['task']].append(q)
    quotas={k:len(v)*min(limit,len(qs))//len(qs) for k,v in groups.items()}
    for k in sorted(groups,key=lambda k:(-(len(groups[k])*min(limit,len(qs))%len(qs)),k))[:min(limit,len(qs))-sum(quotas.values())]:quotas[k]+=1
    chosen=[]
    for k,v in groups.items():chosen.extend(sorted(v,key=lambda q:hashlib.sha256(f"{seed}:{q['id']}".encode()).hexdigest())[:quotas[k]])
    paths={v['path'] for q in chosen for v in q['streams']};dest=ROOT/'data/native_pilot'/name;dest.mkdir(parents=True,exist_ok=True)
    media={p:{'path':str(BASE/name/'media'/p)} for p in sorted(paths)}
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for p,d in zip(media,pool.map(probe,[v['path'] for v in media.values()])):media[p].update(d)
    items=[];gold={}
    cfg=next(d for d in json.loads((ROOT/'docs/native_dual_video_parameters.json').read_text())['datasets'] if d['name']==name)
    for original in chosen:
        q=dict(original);question=q['question'];options=[re.sub(r'^[A-Z][.)]\s*','',str(v)) for v in q['options'].values()];labels=list(q['options']);streams=[]
        for v in q['streams']:
            stream={'video_id':v['path']}
            if 'time_range' in v:stream.update(start=v['time_range'][0],end=v['time_range'][1])
            streams.append(stream)
        if '{fps}' in question:
            rates={media[v['path']]['fps'] for v in q['streams']}
            if len(rates)!=1:raise ValueError('Ambiguous FPS placeholder')
            question=question.replace('{fps}',format(rates.pop(),'g'))
        if q.get('grounding_required'):
            refs=sorted(set(re.findall(r'\{([AB][1-5])\}',question+' '+ ' '.join(options))));lines=[]
            for ref in refs:
                stream=q['streams'][0 if ref[0]=='A' else 1];obj=q['original']['objects'][int(ref[1])-1]
                boxes=next(x['bbox'] for x in json.loads((BASE/name/'media'/stream['bbox_path']).read_text()) if x['id']==obj['id'])
                files=sorted((BASE/name/'media'/stream['path']).glob('*.jpg'));indices=np.linspace(0,len(files)-1,cfg['frames_per_stream'],dtype=int)
                appearances=[(j,int(index),boxes[str(index)]) for j,index in enumerate(indices) if boxes.get(str(index)) is not None and not boxes[str(index)].get('outside',False)]
                if not appearances:raise ValueError('Grounded object absent from sampled frames: '+q['id']+' '+ref)
                j,index,box=appearances[0]
                from qwen_vl_utils.vision_process import smart_resize
                with Image.open(files[index]) as im:
                    scale=min(1,cfg['long_edge']/max(im.height,im.width));h,w=smart_resize(max(28,int(im.height*scale)),max(28,int(im.width*scale)),factor=28,min_pixels=3136,max_pixels=cfg['long_edge']**2)
                    xy=[round(box[k]*(w/im.width if k in ('xtl','xbr') else h/im.height)) for k in ('xtl','ytl','xbr','ybr')]
                lines.append(f"{ref} ({obj['label']}) appears in sampled frame {j+1} of Video {1 if ref[0]=='A' else 2} with bbox [left, top, right, bottom] = {xy}.")
            question+='\nView A is Video 1; view B is Video 2. Object identification:\n'+'\n'.join(lines)
        options=[f'{k}. {v}' for k,v in zip(labels,options)]
        durations=[(s.get('end',media[s['video_id']]['duration']) or 1e9)-(s.get('start',0) or 0) for s in streams]
        ident=name+'_'+hashlib.sha256(q['id'].encode()).hexdigest()[:16]
        item={'id':ident,'source_id':q['id'],'dataset':name,'protocol':'native_two_video_mc','question':question,'questions':[{'id':q['id'],'question':question,'options':options,'option_labels':labels}],'streams':streams,'score_mask':[True],'task':q['task'],'duration_seconds':sum(durations)}
        items.append(item);gold[q['id']]={'answer':q['answers'][0],'task':q['task']}
    items.sort(key=lambda q:(q['duration_seconds'],q['id']))
    atomic(dest/'questions.json',[{'pairs':items}]);atomic(dest/'gold.json',gold);atomic(dest/'media_map.json',media)
    summary={'dataset':name,'available_questions':len(qs),'selected_questions':len(items),'seed':seed,'selection':'task-proportional deterministic hash; then schedule by ascending duration','task_counts':quotas,'median_total_video_seconds':float(np.median([q['duration_seconds'] for q in items])) if all(q['duration_seconds']<1e9 for q in items) else None,'parameters':cfg}
    atomic(dest/'selection.json',summary);print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('datasets',nargs='+');ap.add_argument('--limit',type=int,default=200);a=ap.parse_args()
    for name in a.datasets:build(name,a.limit)
