"""A single process/model/configuration; per-pair atomic, resumable outputs."""
import argparse,fcntl,hashlib,json,os,time,traceback,subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
from exp.baseline.runtime.paths import ROOT, model_path
from exp.baseline.methods.base import seed_for,budget_matches

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def compress_native(ns, stream, method, target, counter=None):
    if method not in ('omnizip','omniselect'):return ns['compress'](stream)
    thinker=ns['thinker'];ids=stream['inputs']['input_ids'][0]
    av=(ids==thinker.config.audio_token_id)|(ids==thinker.config.video_token_id)
    trials=[]
    def trial(offset):
        if method=='omnizip':
            # Preserve the prescribed audio-priority difference throughout calibration.
            audio_keep=min(1.,max(0.,target+.25+offset));video_keep=min(1.,max(0.,target-.05+offset))
            thinker.omnizip_config.update(rho_audio=1-audio_keep,rho_video=1-video_keep)
        else:ns['selector'].ratio=min(.999,max(.001,target+offset))
        before=counter.total if counter is not None else 0
        part=ns['compress'](stream)
        if counter is not None:part['aux']['selected_auxiliary_flops']=counter.total-before+part['aux'].get('reused_selector_auxiliary_flops',0)
        observed=float(part['final'][av].sum()/av.sum())
        trials.append({'offset':offset,'actual_av_ratio':observed})
        return part,observed
    part,observed=trial(0.);best=(abs(observed-target),part);lo,hi=-.75,.75
    if best[0]>.0002:
        for _ in range(12):
            offset=(lo+hi)/2;candidate,observed=trial(offset)
            if abs(observed-target)<best[0]:best=(abs(observed-target),candidate)
            if observed<target:lo=offset
            else:hi=offset
            if best[0]<=.0002:break
    result=best[1];result['aux'].update(calibration=trials,budget_matched=best[0]<=.005)
    return result

def prepared_inputs(questions,folder,enabled):
    """At most one future input ahead; CPU reads/hashes overlap GPU compute."""
    import torch
    def read(q):
        try:
            path=Path(folder)/(q['id']+'.pt');payload=torch.load(path,map_location='cpu',weights_only=False)
            checksum=digest(path)
            if enabled:
                for stream in payload['streams']:
                    stream['inputs']={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in stream['inputs'].items()}
            return payload,checksum,None
        except Exception as error:return None,None,error
    if not enabled:
        for q in questions:
            started=time.perf_counter();yield q,read(q),time.perf_counter()-started
        return
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='pair-prefetch') as pool:
        future=pool.submit(read,questions[0])
        for index,q in enumerate(questions):
            started=time.perf_counter();result=future.result();wait=time.perf_counter()-started
            if index+1<len(questions):future=pool.submit(read,questions[index+1])
            yield q,result,wait

def execute(job_path,resident_models=None,prefetch=True):
    a=SimpleNamespace(job=Path(job_path))
    job=json.loads(a.job.read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not physical or ',' in physical:raise ValueError('Worker requires exactly one allocated physical GPU')
    # Parent inspected availability. Recheck before model allocation.
    if not resident_models:
        occupants=subprocess.check_output(['nvidia-smi','-i',physical,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if occupants:raise RuntimeError('GPU acquired by another process; retry when free')
    import torch
    from exp.baseline.runtime.models import load
    from exp.baseline.runtime.native_prompt import assemble
    from exp.baseline.methods.video import install_encoder_stats,apply_video
    from exp.baseline.methods.random import compress as random_compress
    from exp.baseline.metrics.flops import decoder_flops
    from fro.scoring import ScoreConfig,prune
    from fro.temporal import attach_metadata
    from exp.baseline.metrics.flops import DECODE_STEPS
    DECODE_STEPS.set(job.get('flops_decode_tokens',100))
    method=job['method'];ratio=1. if method=='full_tokens' else float(job['ratio'])
    native=method if method in ('omnizip','omniselect','omnipack','fastv') else 'full_tokens'
    fingerprint=hashlib.sha256(a.job.read_bytes()).hexdigest()
    source_sha256=hashlib.sha256(b''.join(p.read_bytes() for folder in ('src','exp') for p in sorted((ROOT/folder).rglob('*.py')))).hexdigest()
    model_config_sha256=digest(model_path(job['model'])/'config.json')
    pairs_sha256=digest(job['pairs'])
    manifest=json.loads(Path(job['pairs']).read_text())
    questions=[q for b in manifest for q in b['pairs'] if q['id'] in job['pair_ids']]
    if len(questions)!=len(set(job['pair_ids'])):raise ValueError('Missing/duplicate job IDs')
    pending=[]
    for q in questions:
        target=out/(q['id']+'.json')
        if target.exists():
            row=json.loads(target.read_text())
            if row['job_sha256']!=fingerprint:raise ValueError('Output belongs to another configuration')
            if row['status']=='passed':
                if (row.get('source_sha256'),row.get('model_config_sha256'),row.get('pairs_sha256'))!=(source_sha256,model_config_sha256,pairs_sha256):
                    raise ValueError('Source/model/pair manifest changed; use a new run directory')
                continue
        pending.append(q)
    if not pending:return
    key=(native,job['model'],ratio)
    if resident_models and key not in resident_models:raise ValueError('Resident process must be restarted to switch configuration')
    reused=resident_models is not None and key in resident_models
    load_started=time.perf_counter()
    ns=resident_models[key] if reused else load(native,job['model'],ratio,max_new_tokens=job.get('max_new_tokens',32))
    load_seconds=time.perf_counter()-load_started
    if resident_models is not None:resident_models[key]=ns
    thinker=ns['thinker']
    if not reused:
        if method in ('visionzip','fastvid'):install_encoder_stats(thinker)
        from exp.baseline.metrics.encoder_aux import install
        install(thinker,method)
    failed=0
    for q,(payload,cache_checksum,input_error),input_wait in prepared_inputs(pending,job['cache'],prefetch):
        started=time.perf_counter();target=out/(q['id']+'.json')
        record={'pair_id':q['id'],'method':method,'model':job['model'],'target_ratio':ratio,'job_sha256':fingerprint,
                'question_ids':[x['id'] for x in q['questions']],'score_mask':q['score_mask'],
                'budget_mode':job.get('budget_mode','legacy_equal_retention'),'plan_sha256':job.get('plan_sha256'),'dataset':q['dataset'],'physical_gpu':physical,'pid':os.getpid(),'source_sha256':source_sha256,'model_config_sha256':model_config_sha256,'pairs_sha256':pairs_sha256,
                'model_reused':reused,'job_model_load_seconds':load_seconds,'input_wait_seconds':input_wait,'async_prefetch':prefetch}
        try:
            cache=Path(job['cache'])/(q['id']+'.pt')
            if input_error is not None:raise input_error
            if payload['pair_id']!=q['id']:raise ValueError('Cache pair ID mismatch')
            if payload['protocol']['pair_sha256']!=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest():raise ValueError('Cached pair content changed')
            data=assemble(ns['p'],payload['streams'],q)
            if data['joint']['input_ids'].numel()>32256:raise ValueError('Shared input exceeds plan budget')
            seed=seed_for(q['id']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            ids=data['joint']['input_ids'][0];av=(ids==thinker.config.audio_token_id)|(ids==thinker.config.video_token_id)
            torch.cuda.reset_peak_memory_stats();parts=[]
            if job.get('budget_mode')=='matched_flops_dagger' and method!='full_tokens':
                from exp.baseline.runtime.calibrated import run_pair
                record.update(run_pair(ns,data,method,float(job['fro_anchor_retention']),seed,job['model']))
                record.update(cache_sha256=cache_checksum,prompt_sha256=data['prompt_sha256'],preprocessing=payload['protocol'])
                if record['status']!='passed':failed+=1
                atomic(target,record)
                print(json.dumps({k:record.get(k) for k in ('pair_id','method','status','physical_gpu','elapsed_seconds','flops_relative_error')},ensure_ascii=False),flush=True)
                del data,payload
                continue
            for s in data['streams']:
                part=compress_native(ns,s,method,ratio)
                if method in ('visionzip','fastvid'):
                    part['vision_attention']=thinker.visual._vz_attn_mean.clone();part['vision_keys']=thinker.visual._vz_attn_key.clone()
                parts.append(part)
            if method=='fro':
                cfg=ScoreConfig(module='temporal_novelty',bridge_mode='same');attach_metadata(parts,data,ns['p'].tokenizer,cfg.module)
                parts=prune(parts,thinker,cfg,ratio)
            elif method=='random':parts=[random_compress(p,thinker,ratio,seed+i) for i,p in enumerate(parts)]
            elif method in ('visionzip','fastvid','vidcom2','dycoke_ttm'):
                parts=[apply_video(p,s,thinker,method,ratio,seed+i) for i,(p,s) in enumerate(zip(parts,data['streams']))]
            emb,pos,keep=ns['joint_features'](data,parts);stage1=keep.clone()
            if method in ('omnipack','fastv'):
                selected_ids=ids[keep].to('cuda');am=selected_ids==thinker.config.audio_token_id;vm=selected_ids==thinker.config.video_token_id
                if method=='omnipack':
                    thinker.model.omnipack_audio_token_mask=am;thinker.model.omnipack_video_token_mask=vm;thinker.model.omnipack_text_token_mask=~(am|vm)
                    thinker.model.omnipack_global_config={'stage2_layer':26 if job['model']=='3B' else 18,'video_stage2_keep_ratio':.5,'audio_stage2_keep_ratio':.5}
                else:
                    thinker.model.fastv_global_audio_token_mask=am;thinker.model.fastv_global_video_token_mask=vm;thinker.model.fastv_global_text_token_mask=~(am|vm)
                    thinker.model.fastv_global_config={'method_name':'fastv_omni','video_ratio':ratio,'audio_ratio':ratio,'fastv_k':5}
            torch.cuda.synchronize();selected=time.perf_counter()
            answer,tokens,first,lens,stop=ns['decode'](emb,pos)
            torch.cuda.synchronize()
            if method in ('omnipack','fastv'):
                keep=stage1.clone();keep[stage1]=ns['state']['joint_stage2'].cpu()
            if not bool(keep[~av].all()):raise ValueError('Text token was removed')
            actual=float(keep[av].sum()/av.sum())
            matched=budget_matches(int(av.sum()),int(keep[av].sum()),ratio)
            if len(lens)!=thinker.config.text_config.num_hidden_layers:raise ValueError('Missing layer instrumentation')
            config=thinker.config.text_config.to_dict()
            record.update(status='passed' if matched else 'budget_unmatched',answer=answer,generated_tokens=len(tokens),seed=seed,
                final_av_ratio=actual,original_tokens=len(ids),pre_llm_tokens=int(stage1.sum()),final_tokens=int(keep.sum()),
                layer_prefill_lengths=lens,pre_llm_seconds=selected-started,ttft_seconds=first-started,
                elapsed_seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                cache_sha256=cache_checksum,prompt_sha256=data['prompt_sha256'],preprocessing=payload['protocol'],
                flops=decoder_flops(config,lens,int(keep.sum()),kv_lengths=ns['state']['layer_kv_prefill_lengths'],auxiliary_delta=0 if method=='full_tokens' else None),
                stream_diagnostics=[p.get('score_diagnostics',p.get('aux',{})) for p in parts],
                modality_counts={name:{'before':int((ids==getattr(thinker.config,name+'_token_id')).sum()),'final':int(keep[ids==getattr(thinker.config,name+'_token_id')].sum())} for name in ('audio','video')})
            if not matched:failed+=1
            del data,parts,emb,pos,payload
            ns['state'].clear()
        except Exception:
            record.update(status='failed',error=traceback.format_exc());failed+=1
        atomic(target,record);print(json.dumps({k:record.get(k) for k in ('pair_id','status','physical_gpu','elapsed_seconds','error')},ensure_ascii=False),flush=True)
    if failed:raise SystemExit(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--no-prefetch',action='store_true');a=ap.parse_args()
    execute(a.job,prefetch=not a.no_prefetch)

if __name__=='__main__':main()
