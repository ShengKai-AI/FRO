"""Select native two-input multiple-choice tasks; never manufacture question pairs."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import threading
import requests
from exp.baseline.runtime.paths import ROOT

BASE = ROOT / 'data/native_two_video'
SOURCES = BASE / 'sources'
_HTTP_LOCAL = threading.local()
_HTTP_LOCK = threading.Lock()
_HTTP_NEXT = 0.0

def download_session():
    if not hasattr(_HTTP_LOCAL, 'session'):
        _HTTP_LOCAL.session = requests.Session()
    return _HTTP_LOCAL.session

def rate_limit():
    global _HTTP_NEXT
    rate=float(os.environ.get('FRO_DOWNLOAD_REQUESTS_PER_SECOND', '8'))
    with _HTTP_LOCK:
        now=time.monotonic();delay=max(0,_HTTP_NEXT-now);_HTTP_NEXT=max(now,_HTTP_NEXT)+1/rate
    if delay:time.sleep(delay)

NAMES = ['CrossVid', 'X-Stream', 'CVBench', 'MVU-Eval', 'MVPBench', 'SYNCR']

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def manifest():
    return json.loads((ROOT/'manifests/native_two_video.json').read_text())

def fetch(url, path, size=None, sha256=None, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        if sha256 is None or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == sha256:
            return path
    if size and shutil.disk_usage(path.parent).free < size+5*2**30:
        raise OSError('Insufficient space with 5 GiB reserve')
    temp = path.with_suffix(path.suffix+'.partial')
    for attempt in range(12):
        try:
            if '/resolve/' in url:rate_limit()
            with download_session().get(url, headers=headers, stream=True, timeout=(15, 90)) as r:
                if r.status_code == 429:
                    time.sleep(max(float(r.headers.get("Retry-After", 1)), min(2**attempt,60)))
                r.raise_for_status()
                digest = hashlib.sha256(); count = 0
                with temp.open('wb') as f:
                    for block in r.iter_content(1024*1024):
                        f.write(block); digest.update(block); count += len(block)
                if size is not None and count != size: raise IOError('Size mismatch: '+str(path))
                if sha256 and digest.hexdigest() != sha256: raise IOError('SHA256 mismatch: '+str(path))
                temp.replace(path)
                return path
        except (requests.RequestException, IOError):
            if attempt == 11: raise
            time.sleep(min(2**attempt,30))

def hf_url(entry, path):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
    return f"{endpoint}/datasets/{entry['repo_id']}/resolve/{entry['revision']}/{path}"

def prepare_sources(name):
    entry = manifest()['datasets'][name]
    if name == 'MVPBench':
        from .remote_archive import HTTPRangeFile
        import zipfile
        for a in entry['archives']:
            folder = SOURCES/'MVPBench'/a['id']; annotation = folder/'annotations'/a['annotation']
            if annotation.exists() and hashlib.sha256(annotation.read_bytes()).hexdigest()==a['annotation_sha256'] and (folder/'index.json').exists(): continue
            url = f"https://drive.usercontent.google.com/download?id={a['id']}&export=download&confirm=t"
            with HTTPRangeFile(url,a['size']) as remote, zipfile.ZipFile(remote) as z:
                data=z.read(a['annotation'])
                if hashlib.sha256(data).hexdigest()!=a['annotation_sha256']:raise ValueError('MVPBench annotation revision changed')
                annotation.parent.mkdir(parents=True,exist_ok=True);annotation.write_bytes(data)
                atomic(folder/'index.json',{'name':a['name'],'url':url,'size':a['size'],'members':[{'path':i.filename,'size':i.file_size,'compressed_size':i.compress_size,'offset':i.header_offset} for i in z.infolist()]})
        return
    if not entry.get('repo_id'): return
    folder = SOURCES/entry['source_folder']
    for path in entry['annotation_files']:
        fetch(hf_url(entry, path), folder/path, **entry.get('annotation_checksums', {}).get(path, {}))
    index = SOURCES/(entry['source_folder']+'.repo.json')
    if not index.exists():
        url = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')+'/api/datasets/'+entry['repo_id']+'/revision/'+entry['revision']+'?blobs=true'
        fetch(url, index)

def options_and_answer(raw_options, raw_answer):
    repaired = False
    if isinstance(raw_options, dict):
        keys = list(raw_options); values = list(raw_options.values())
    elif isinstance(raw_options, list):
        values = list(raw_options)
        if len(values) == 1 and isinstance(values[0], str):
            pieces = re.split(r',\s*(?=[A-Z]\.\s)', values[0])
            if len(pieces) > 1: values = pieces; repaired = True
        keys = [chr(65+i) for i in range(len(values))]
    else:
        return None
    if len(values) < 2: return None
    def choice(value):
        text = str(value).strip()
        if text in keys: return text
        m = re.match(r'^([A-Z])[.)]\s', text)
        if m and m[1] in keys: return m[1]
        clean = lambda v: re.sub(r'^[A-Z][.)]\s*', '', str(v)).strip().rstrip('.').casefold()
        hits = [k for k,v in zip(keys, values) if clean(v) == clean(text)]
        return hits[0] if len(hits) == 1 else None
    answers = raw_answer if isinstance(raw_answer, list) else [raw_answer]
    labels = [choice(v) for v in answers]
    if not labels or any(v is None for v in labels): return None
    return dict(zip(keys, values)), labels, repaired

def select(name):
    prepare_sources(name)
    entry = manifest()['datasets'][name]; folder = SOURCES/entry.get('source_folder', name)
    candidates = []; raw_total = 0; two_total = 0
    def add(ident, task, question, videos, opts, answer, raw, **extra):
        nonlocal raw_total, two_total
        raw_total += 1
        if len(videos) != 2: return
        two_total += 1
        parsed = options_and_answer(opts, answer)
        if parsed is None: return
        options, labels, repaired = parsed
        candidates.append({'id': f'{name}:{ident}', 'dataset':name, 'task':task, 'question':question,
            'streams':videos, 'options':options, 'answers':labels,
            'answer_type':'single_choice' if len(labels)==1 else 'multiple_select',
            'options_repaired':repaired, 'original':raw, **extra})
    if name == 'CrossVid':
        for p in sorted((folder/'QA').glob('*.json')):
            task = p.stem
            for r in json.loads(p.read_text()):
                videos = [{'path':'videos/'+v} for v in r.get('videos',[])]
                if task in ['MOC','MSR']:
                    videos = [{'path':f"uav/frames/{v}/{r['vid']}-{v}", 'kind':'frames', 'bbox_path':f"uav/bbox/{v}/{r['vid']}.json"} for v in [1,2]]
                elif task == 'PI':
                    videos = [{'path':'videos/'+r['video'],'time_range':r[k]} for k in ['beginning','ending']]
                elif 'video A' in r:
                    videos = [{'path':'videos/'+r[k]} for k in ['video A','video B']]
                elif task == 'PSS':
                    videos = [{'path':'videos/'+r['video'],'segments':s} for s in r['segments'].values()]
                question = r.get('question', 'Given the beginning and ending of a movie clip, what is most likely to happen in the middle?' if task=='PI' else '')
                add(f"{task}:{r['id']}",task,question,videos,r.get('options'),r.get('answer'),r,
                    input_relation='same_source_two_clips' if task=='PI' else 'two_views' if task in ['MOC','MSR'] else 'distinct_videos', grounding_required=task in ['MOC','MSR'])
    elif name == 'MVU-Eval':
        for ident,r in json.loads((folder/'MVU_Eval_QAs.json').read_text()).items():
            add(ident,r['task'],r['question'],[{'path':v} for v in r['video_paths']],r['options'],r['ground_truth'],r)
    elif name == 'CVBench':
        import pyarrow.parquet as pq
        for r in pq.read_table(folder/'data/test-00000-of-00001.parquet').to_pylist():
            videos = [{'path':'CVBench/'+r['video_'+str(i)]} for i in range(1,5) if r.get('video_'+str(i))]
            add(r['id'],r['task_type'],r['question'],videos,r['options'],r['answer'],r)
    elif name == 'SYNCR':
        for i,line in enumerate((folder/'test/metadata.jsonl').read_text().splitlines()):
            r=json.loads(line); ranges=r.get('video_ranges') or []
            videos=[{'path':v, **({'time_range':ranges[j]} if ranges else {})} for j,v in enumerate(r['videos'])]
            add(i,r['task'],r['question'],videos,r['options'],r['answer'],r)
    elif name == 'X-Stream':
        for i,line in enumerate((folder/'eval_relative.json').read_text().splitlines()):
            r=json.loads(line)
            for j,q in enumerate(r['verified_responses']):
                if q.get('is_keep') != 1: continue
                videos=[{'path':v,'encoded_path':r['encoded_video_path'][k]} for k,v in enumerate(r['original_video_path'])]
                add(f'{i}:{j}',q['time_type'],q['user_query'],videos,q.get('options'),q.get('correct_answer'),q,
                    timestamp_question=q['timestamp_question'],timestamp_proactive=q.get('timestamp_proactive'),audio_status='pending_media_probe')
    elif name == 'MVPBench':
        for a in entry['archives']:
            index=json.loads((folder/a['id']/'index.json').read_text())
            paths={m['path'] for m in index['members'] if not m['path'].startswith('__MACOSX/')}
            rows=[json.loads(line) for line in (folder/a['id']/'annotations'/a['annotation']).read_text().splitlines() if line.strip()]
            for i,r in enumerate(rows):
                n=r['query'].count('<video>') if 'query' in r else len(r.get('video_order',[]))
                videos=[{'path':'unselected:'+str(j)} for j in range(n)]
                if n==2:
                    videos=[]
                    for v in r['video_options']:
                        prefix='EPIC_Skills2018_videos/'+r['id']+'/' if a['name'].startswith('Daily life') else 'AQA-7/'
                        path=prefix+v
                        if path not in paths:raise ValueError('Unresolved MVPBench media '+path)
                        videos.append({'path':a['id']+'/'+path,'archive_id':a['id'],'archive_member':path})
                opts=r.get('options')
                if opts==['1','2']:opts={'1':'Video 1','2':'Video 2'}
                add(a['id']+':'+str(i),a['name'].removesuffix('.zip'),r.get('query',r.get('question','')),videos,opts,r.get('answer'),r)
    assert len({r['id'] for r in candidates})==len(candidates)
    # IDs remain source-derived. Detect exact repeated semantic questions separately.
    semantic = collections.Counter(json.dumps([r['streams'],r['question'],r['options'],r.get('timestamp_question'),r.get('timestamp_proactive'),r['original'].get('objects')],sort_keys=True) for r in candidates)
    summary={'dataset':name,'source_revision':entry.get('revision') or 'metadata-sha256:'+hashlib.sha256(json.dumps(entry.get('archives',[]),sort_keys=True).encode()).hexdigest(),'raw_questions':raw_total,'exact_two_input_questions':two_total,
        'selected_mc_questions':len(candidates),'single_choice_questions':sum(r['answer_type']=='single_choice' for r in candidates),
        'multiple_select_questions':sum(r['answer_type']=='multiple_select' for r in candidates),
        'exact_duplicate_extra_rows':sum(n-1 for n in semantic.values()),'task_counts':dict(collections.Counter(r['task'] for r in candidates)),
        'option_repairs':sum(r['options_repaired'] for r in candidates),
        'unique_source_paths':len({v['path'] for r in candidates for v in r['streams']}),
        'audio_filter':'pending_media_probe' if name=='X-Stream' else 'not_required'}
    atomic(BASE/name/'questions.candidates.json',candidates)
    if name!='X-Stream':atomic(BASE/name/'questions.json',candidates)
    if name=='X-Stream' and (BASE/name/'questions.json').exists():
        final=json.loads((BASE/name/'questions.json').read_text())
        if all(q['id'] in {r['id'] for r in candidates} for q in final):
            summary.update(audio_filter='complete',audio_selected_questions=len(final),both_streams_audio_questions=sum(all(v.get('has_audio',False) for v in q['streams']) for q in final))
    atomic(BASE/name/'selection_summary.json',summary)
    return candidates,summary

def direct_assets(name, questions):
    entry=manifest()['datasets'][name]; repo=json.loads((SOURCES/(entry['source_folder']+'.repo.json')).read_text())
    wanted={v['path'] for q in questions for v in q['streams']}
    wanted.update(v['bbox_path'] for q in questions for v in q['streams'] if 'bbox_path' in v)
    prefixes={v['path']+'/' for q in questions for v in q['streams'] if v.get('kind')=='frames'}
    files=[f for f in repo['siblings'] if f['rfilename'] in wanted or any(f['rfilename'].startswith(p) for p in prefixes)]
    assert all(any(f['rfilename']==p or f['rfilename'].startswith(p+'/') for f in files) for p in wanted), 'Unresolved source paths'
    atomic(BASE/name/'assets.json',files)
    return files

def download_direct(name, questions, workers=16):
    entry=manifest()['datasets'][name];files=direct_assets(name,questions);dest=BASE/name/'media'
    dest.mkdir(parents=True,exist_ok=True)
    needed=sum(f['size'] for f in files if not (dest/f['rfilename']).exists())
    if shutil.disk_usage(dest).free < needed+5*2**30: raise OSError(f'Need {needed/2**30:.1f} GiB plus reserve for {name}')
    errors=[];done=0
    def work(f, action=None):
        return fetch(action['href'] if action else hf_url(entry,f['rfilename']),dest/f['rfilename'],f['size'],f.get('lfs',{}).get('sha256'),action.get('header') if action else None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={}
        if name=='CrossVid' and os.environ.get('FRO_USE_LFS_BATCH')=='1':
            # Git LFS batch resolves hundreds of content-addressed URLs per request.
            endpoint=os.environ.get('FRO_LFS_ENDPOINT','https://huggingface.co').rstrip('/')
            proxy=os.environ.get('FRO_LFS_PROXY');proxies={'http':proxy,'https':proxy} if proxy else None
            pending=[]
            for f in files:
                path=dest/f['rfilename']
                if path.exists() or 'lfs' not in f:
                    futures[pool.submit(work,f)]=f
                else:pending.append(f)
            for start in range(0,len(pending),256):
                batch=pending[start:start+256]
                response=requests.post(endpoint+'/datasets/'+entry['repo_id']+'.git/info/lfs/objects/batch',json={'operation':'download','transfers':['basic'],'objects':[{'oid':f['lfs']['sha256'],'size':f['size']} for f in batch]},headers={'Accept':'application/vnd.git-lfs+json'},proxies=proxies,timeout=(20,90))
                response.raise_for_status()
                actions={o['oid']:o.get('actions',{}).get('download') for o in response.json()['objects']}
                for f in batch:futures[pool.submit(work,f,actions.get(f['lfs']['sha256']))]=f
        else:
            futures={pool.submit(work,f):f for f in files}
        for future in concurrent.futures.as_completed(futures):
            try:future.result();done+=1
            except Exception as e:errors.append({'path':futures[future]['rfilename'],'error':str(e)})
            if done%100==0 or errors:atomic(BASE/name/'download_status.json',{'completed_files':done,'total_files':len(files),'errors':errors,'time':time.time()})
    atomic(BASE/name/'download_status.json',{'completed_files':done,'total_files':len(files),'errors':errors,'complete':not errors,'time':time.time()})
    if errors:raise RuntimeError(f'{name}: {len(errors)} files failed; see download_status.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('name',choices=NAMES+['all']);ap.add_argument('--execute',action='store_true');ap.add_argument('--workers',type=int,default=16);a=ap.parse_args()
    for name in NAMES if a.name=='all' else [a.name]:
        questions,summary=select(name);print(json.dumps(summary,ensure_ascii=False),flush=True)
        if a.execute:
            if name in ['CrossVid','MVU-Eval']:download_direct(name,questions,a.workers)
            else:
                from .native_media import download
                download(name,questions,a.workers)

if __name__=='__main__':main()
