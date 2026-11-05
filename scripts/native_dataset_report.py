"""Report annotation selection and actual local media readiness; never launch jobs."""
import collections
import csv
import functools
import json
from pathlib import Path
from exp.baseline.datasets.native_two_video import BASE, NAMES, ROOT, atomic, manifest


def report():
    rows = []
    for name in NAMES:
        folder = BASE / name
        summary = json.loads((folder / 'selection_summary.json').read_text())
        candidate = json.loads((folder / 'questions.candidates.json').read_text())
        final_path = folder / 'questions.json'
        final = json.loads(final_path.read_text()) if final_path.exists() else None
        expected = {}
        if (folder / 'assets.json').exists():
            expected = {f['rfilename']: f['size'] for f in json.loads((folder / 'assets.json').read_text())}
        @functools.lru_cache(maxsize=None)
        def checked_stream(encoded):
            v = json.loads(encoded)
            path = folder / 'media' / v['path']
            if v.get('kind') == 'frames':
                required = {p: n for p, n in expected.items() if p.startswith(v['path'] + '/') or p == v.get('bbox_path')}
                return bool(required) and all((folder/'media'/p).is_file() and (folder/'media'/p).stat().st_size == n for p,n in required.items())
            return path.is_file() and path.stat().st_size > 0 and (v['path'] not in expected or path.stat().st_size == expected[v['path']])
        def stream_ready(v):
            return checked_stream(json.dumps(v,sort_keys=True))
        row = dict(summary)
        row['paper'] = manifest()['datasets'][name]['paper']
        row['final_selected_questions'] = len(final) if final is not None else None
        row['ready_questions'] = sum(all(stream_ready(v) for v in q['streams']) for q in final) if final is not None else None
        row['download_complete'] = final is not None and row['ready_questions'] == len(final)
        if name == 'CrossVid':
            row['input_relations'] = dict(collections.Counter(q['input_relation'] for q in candidate))
        if name == 'X-Stream':
            p=folder/'audio_probes.json'; probes=json.loads(p.read_text()) if p.exists() else {}
            row['probed_source_videos']=len(probes)
            row['probed_with_audio']=sum(p['audio'] for p in probes.values())
            row['fully_probed_candidate_questions']=sum(all(v['path'] in probes for v in q['streams']) for q in candidate)
            row['confirmed_audio_questions_so_far']=sum(all(v['path'] in probes for v in q['streams']) and any(probes[v['path']]['audio'] for v in q['streams']) for q in candidate)
        rows.append(row)
    dest=ROOT/'results/native_two_video';dest.mkdir(parents=True,exist_ok=True)
    atomic(dest/'selection_report.json', {'protocol':'native_two_video_mc','datasets':rows})
    fields=['dataset','raw_questions','exact_two_input_questions','selected_mc_questions','final_selected_questions','ready_questions','download_complete','unique_source_paths','audio_filter']
    with (dest/'selection_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    print((dest/'selection_counts.csv').read_text())
    return rows

if __name__=='__main__':report()
