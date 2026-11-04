"""Offline scoring; inference workers never open gold labels."""
import argparse,collections,json,re
from pathlib import Path

def parse(answer,index,count):
    matches=re.findall(rf'(?im)^\s*Q{index}\s*:\s*([A-Z])\s*[.)]?\s*$',answer)
    return matches[0] if len(matches)==1 and ord(matches[0])-65 in range(count) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--gold',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--require-full-coverage',action='store_true');a=ap.parse_args()
    gold=json.loads(a.gold.read_text());groups=collections.defaultdict(list)
    for job_path in a.run.glob('*.json'):
        job=json.loads(job_path.read_text())
        if 'pair_ids' not in job:continue
        key=(job['model'],job['method'],job['ratio']);pairs=json.loads(Path(job['pairs']).read_text());lookup={p['id']:p for b in pairs for p in b['pairs']}
        for pair_id in job['pair_ids']:
            path=Path(job['output'])/(pair_id+'.json')
            if not path.exists():raise ValueError('Missing result: '+pair_id)
            row=json.loads(path.read_text())
            if row['status']!='passed':raise ValueError('Non-passing result: '+pair_id+' '+row['status'])
            for i,(question,score) in enumerate(zip(lookup[pair_id]['questions'],row['score_mask']),1):
                if not score:continue
                ident=question['id'];truth=gold[ident]
                if truth['answer'] is None:raise ValueError('Unresolved gold: '+ident)
                groups[key].append({'id':ident,'task':truth.get('task'),'correct':parse(row['answer'],i,len(question['options']))==truth['answer']})
    report=[]
    for key,rows in groups.items():
        counts=collections.Counter(r['id'] for r in rows)
        if any(n!=1 for n in counts.values()):raise ValueError('Duplicate scored IDs')
        if a.require_full_coverage and set(counts)!=set(gold):raise ValueError('Scored IDs do not cover the complete gold manifest')
        by_task=collections.defaultdict(list)
        for r in rows:by_task[r['task']].append(r['correct'])
        report.append({'model':key[0],'method':key[1],'ratio':key[2],'scored':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
                       'task_accuracy':{str(k):sum(v)/len(v) for k,v in by_task.items()},'scope':'complete provided gold manifest' if a.require_full_coverage else 'exact selected job IDs; not necessarily full benchmark'})
    if not report:raise ValueError('No results')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
