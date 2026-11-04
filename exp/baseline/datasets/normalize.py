"""Normalize released annotation formats; gold stays in a separate file."""
import argparse,json,re
from pathlib import Path
import yaml
from exp.baseline.runtime.paths import ROOT
from .pair import make_pairs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-subset',action='store_true');a=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'exp/baseline/configs/datasets'/(a.dataset+'.yaml')).read_text())
    files=list(a.annotations.rglob('*.parquet')) if a.annotations.is_dir() else [a.annotations]
    if a.dataset=='MVBench' and a.annotations.is_dir():files=sorted(a.annotations.glob('*.json'))
    rows=[];gold={}
    for path in files:
        if path.suffix=='.parquet':
            import pyarrow.parquet as pq
            values=pq.read_table(path).to_pylist()
        else:values=json.loads(path.read_text())
        if not isinstance(values,list):raise ValueError('Annotation root must be a list')
        for i,x in enumerate(values):
            d=a.dataset;extra={};task=x.get('task_type',x.get('question_category',x.get('Type')))
            if d=='Video-MME':qid=x['question_id'];video=x['videoID'];q=x['question'];options=x['options'];answer=x['answer']
            elif d=='WorldSense':qid=x['index'];video=x['video'];q=x['question'];options=x['candidates'];answer=x['answer'];extra={'video_path':x.get('video_path'),'audio_path':x.get('audio_path')}
            elif d=='AVUT':
                if 'human' not in path.name.lower():raise ValueError('AVUT requires Human annotations only')
                qid=x.get('QA_id',i);video=Path(x['video_path']).name;q=x['question'];options=[x['option_'+c] for c in 'ABCD'];answer=x['answer']
            elif d=='MVBench':
                qid=path.stem+':'+str(i);namespaces=json.loads((ROOT/'exp/baseline/configs/datasets/mvbench_media_namespaces.json').read_text());video=namespaces[path.stem]+'/'+x['video'];q=x['question'];options=x['candidates'];answer=x['answer'];task=path.stem;extra={'start':x.get('start'),'end':x.get('end'),'media_namespace':path.stem}
            elif d=='DailyOmni':qid=i;video=x['video_id'];q=x['Question'];options=x['Choice'];answer=x['Answer']
            elif d=='LongVideoBench':
                if 'val' not in path.name:raise ValueError('LongVideoBench requires validation ground truth')
                qid=x['id'];video=x['video_path'];q=x['question'];options=x['candidates'];answer=chr(65+int(x['correct_choice']))
            else:raise ValueError(d)
            options=[re.sub(r'^[A-Z][.)]\s*','',s) for s in options]
            answer=str(answer).strip()
            if answer not in [chr(65+j) for j in range(len(options))]:
                matched=[j for j,o in enumerate(options) if o.strip()==answer]
                answer=chr(65+matched[0]) if len(matched)==1 else None
            ident=f'{d}:{cfg["split"]}:{qid}'
            if ident in gold:raise ValueError('Duplicate source question')
            rows.append({'id':ident,'dataset':d,'video_id':str(video),'question':q,'options':options,'task':task,**extra})
            gold[ident]={'answer':answer,'task':task,'options_count':len(options)}
    if not a.allow_subset and len(rows)!=cfg['expected_questions']:raise ValueError(f'Expected {cfg["expected_questions"]}, got {len(rows)}')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/(a.dataset+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (a.output/(a.dataset+'.json')).write_text(json.dumps(make_pairs(rows),ensure_ascii=False,indent=2))
    (a.output/(a.dataset+'.gold.json')).write_text(json.dumps(gold,ensure_ascii=False,indent=2))
    print(json.dumps({'questions':len(rows),'unresolved_gold':sum(x['answer'] is None for x in gold.values()),'subset':a.allow_subset}))
if __name__=='__main__':main()
