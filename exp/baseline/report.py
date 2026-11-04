"""Export measured outcomes; never fill missing FLOPs or unrun accuracies."""
import argparse,csv,json,collections
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows=[]
    for path in a.run.glob('outputs/*/*.json'):
        row=json.loads(path.read_text());rows.append({k:row.get(k) for k in ['pair_id','model','method','target_ratio','status','physical_gpu','final_av_ratio','elapsed_seconds','peak_allocated_gib']})
    if not rows:raise ValueError('No results')
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(collections.Counter(r['status'] for r in rows))))
if __name__=='__main__':main()
