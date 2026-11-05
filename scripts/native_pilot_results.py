"""Score one-label native MC outputs, keeping decode failures explicit."""
import argparse,collections,csv,json,re
from pathlib import Path
from exp.baseline.worker import atomic

def parse_native(text,labels):
    text=text.strip()
    explicit=re.findall(r'(?im)^\s*(?:Answer|Final answer)\s*:\s*([A-Z0-9]+)\s*[.)]?\s*$',text)
    tagged=re.findall(r'<answer>\s*([A-Z0-9]+)\s*</answer>',text,re.I)
    matches=explicit+tagged
    if matches:return matches[0].upper() if len(set(x.upper() for x in matches))==1 and matches[0].upper() in labels else None
    m=re.fullmatch(r'\s*\(?([A-Z0-9]+)\)?[.)]?\s*',text)
    return m[1] if m and m[1] in labels else None

def score(run,gold_path,pairs_path):
    gold=json.loads(Path(gold_path).read_text());pairs={q['id']:q for b in json.loads(Path(pairs_path).read_text()) for q in b['pairs']};groups=collections.defaultdict(dict)
    for path in Path(run).glob('outputs/*/*.json'):
        r=json.loads(path.read_text())
        if r.get('status')!='passed':continue
        q=pairs[r['pair_id']]['questions'][0];choice=parse_native(r['answer'],q['option_labels']);key=(r['model'],r['method'],r.get('fro_anchor_retention') or 1.)
        if q['id'] in groups[key]:raise ValueError('Duplicate native result')
        groups[key][q['id']]={'correct':choice==gold[q['id']]['answer'],'unparsed':choice is None}
    rows=[]
    for key,values in sorted(groups.items()):
        n=len(values);rows.append({'model':key[0],'method':key[1],'budget_anchor':key[2],'completed':n,'total':len(pairs),'accuracy':sum(x['correct'] for x in values.values())/n,'unparsed_ratio':sum(x['unparsed'] for x in values.values())/n})
    atomic(Path(run)/'accuracy.json',rows);return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--dataset-dir',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(score(a.run,a.dataset_dir/'gold.json',a.dataset_dir/'questions.json'),indent=2))
