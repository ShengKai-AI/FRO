"""Selective archive extraction for native multi-video benchmark media."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile
import requests
from .remote_archive import HTTPRangeFile
from .native_two_video import BASE, SOURCES, atomic, manifest, hf_url

def write_member(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free < len(data)+5*2**30:raise OSError('5 GiB disk reserve reached')
    temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(data);temp.replace(path)
    return {'path':str(path),'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def selective_zip(name,url,size,mapping,workers=8,index_path=None):
    """mapping maps an exact archive member to its normalized destination."""
    local=threading.local(); records=[];errors=[];name_safe=name.replace('/','_')
    def work(item):
        member,destination=item;destination=Path(destination)
        if not hasattr(local,'zip'):
            local.remote=HTTPRangeFile(url,size);local.zip=zipfile.ZipFile(local.remote)
        info=local.zip.getinfo(member)
        if destination.exists() and destination.stat().st_size==info.file_size:
            return {'path':str(destination),'size':info.file_size,'reused':True}
        return write_member(destination,local.zip.read(member))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(work,item):item for item in mapping.items()}
        for f in concurrent.futures.as_completed(futures):
            try:records.append(f.result())
            except Exception as e:errors.append({'member':futures[f][0],'error':str(e)})
            if len(records)%20==0 or errors:atomic(BASE/'download_receipts'/(name_safe+'.json'),{'completed':len(records),'total':len(mapping),'errors':errors,'time':time.time()})
    result={'completed':len(records),'total':len(mapping),'errors':errors,'files':records,'complete':not errors,'time':time.time()}
    atomic(BASE/'download_receipts'/(name_safe+'.json'),result)
    if errors:raise RuntimeError(f'{name}: {len(errors)} failed members')
    return result

def selective_tar(name,url,size,wanted):
    """Index independent TAR byte intervals, then fetch selected payloads only."""
    checkpoint=BASE/'download_receipts'/(name+'.json')
    index_path=SOURCES/(name+'.tar_index.json')
    index=json.loads(index_path.read_text()) if index_path.exists() else []
    if not set(wanted).issubset({m['name'] for m in index}):
        # A checksum-valid, 512-byte aligned ustar header anchors each interval.
        # Processing the full member that crosses a boundary retains GNU/PAX names.
        chunks=32;step=((size//chunks+511)//512)*512;root=next(iter(wanted)).split('/')[0]
        def scan(part):
            cached=SOURCES/(name+'.index_parts')/(str(part)+'.json')
            if cached.exists():return json.loads(cached.read_text())
            begin=part*step;end=min((part+1)*step,size);f=HTTPRangeFile(url,size);anchor=None
            cursor=begin
            while cursor<end and anchor is None:
                f.seek(cursor);block=f.read(min(256*1024,end-cursor))
                for offset in range(0,len(block)-511,512):
                    buf=block[offset:offset+512]
                    if buf[257:262]!=b'ustar':continue
                    try:member=tarfile.TarInfo.frombuf(buf,'utf-8','surrogateescape')
                    except tarfile.TarError:continue
                    if member.name.startswith(root) or member.type in (tarfile.GNUTYPE_LONGNAME,tarfile.XHDTYPE):
                        anchor=cursor+offset;break
                cursor+=len(block)
            records=[]
            if anchor is not None:
                f.seek(anchor)
                with tarfile.open(fileobj=f,mode='r:') as archive:
                    for member in archive:
                        if member.offset>=end:break
                        if member.isfile():records.append({'name':member.name,'size':member.size,'offset':member.offset_data})
            f.close();atomic(cached,records);return records
        parts=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for future in concurrent.futures.as_completed([pool.submit(scan,p) for p in range(chunks)]):
                parts.extend(future.result())
                atomic(checkpoint,{'indexed':len(parts),'total':len(wanted),'stage':'indexing','time':time.time()})
        by_offset={}
        for m in parts:
            if m['offset'] not in by_offset or len(m['name'])>len(by_offset[m['offset']]['name']):by_offset[m['offset']]=m
        index=sorted(by_offset.values(),key=lambda m:m['offset']);atomic(index_path,index)
    selected={m['name']:m for m in index if m['name'] in wanted}
    if set(selected)!=set(wanted):raise ValueError(f'{name}: missing TAR members {sorted(set(wanted)-set(selected))[:10]}')
    local=threading.local();records=[]
    def extract(item):
        key,m=item;path=Path(wanted[key])
        if path.exists() and path.stat().st_size==m['size']:return {'path':str(path),'size':m['size'],'reused':True}
        if not hasattr(local,'remote'):local.remote=HTTPRangeFile(url,size)
        local.remote.seek(m['offset']);return write_member(path,local.remote.read(m['size']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for future in concurrent.futures.as_completed([pool.submit(extract,item) for item in selected.items()]):
            records.append(future.result())
            if len(records)%20==0:atomic(checkpoint,{'indexed':len(index),'completed':len(records),'total':len(wanted),'stage':'extracting','time':time.time()})
    atomic(checkpoint,{'indexed':len(index),'completed':len(records),'total':len(wanted),'complete':True,'files':records,'time':time.time()})

def probe_audio(path):
    import os
    exe=os.environ.get('FFPROBE') or shutil.which('ffprobe') or '/root/seamless-media-tools/bin/ffprobe'
    r=subprocess.run([exe,'-v','error','-show_entries','stream=codec_type,codec_name','-of','json',str(path)],capture_output=True,text=True,timeout=60,check=True)
    streams=json.loads(r.stdout)['streams']
    return {'audio':any(s['codec_type']=='audio' for s in streams),'video':any(s['codec_type']=='video' for s in streams)}

def xstream(questions):
    import zstandard
    entry=manifest()['datasets']['X-Stream'];dest=BASE/'X-Stream/media';dest.mkdir(parents=True,exist_ok=True)
    wanted={v['path'] for q in questions for v in q['streams']}
    archives=json.loads((SOURCES/'X-stream/archives/archives.json').read_text())
    archives=[a for a in archives if a['split']=='eval' and a['kind']=='original']
    probes_path=BASE/'X-Stream/audio_probes.json';probes=json.loads(probes_path.read_text()) if probes_path.exists() else {}
    sizes={f['rfilename']:f['size'] for f in json.loads((SOURCES/'X-stream.repo.json').read_text())['siblings']}
    lock=threading.Lock()
    def scan(a):
        label=Path(a['archive']).name;receipt=BASE/'download_receipts'/(label+'.json')
        if receipt.exists() and json.loads(receipt.read_text()).get('complete'):return
        seen=0;kept=0
        remote=HTTPRangeFile(hf_url(entry,a['archive']),sizes[a['archive']])
        with remote, zstandard.ZstdDecompressor().stream_reader(remote,read_size=8*1024*1024) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as tar:
                for member in tar:
                    if not member.isfile():continue
                    seen+=1;key=member.name.removeprefix('./')
                    if key in wanted:
                        path=dest/key
                        if not path.exists() or path.stat().st_size!=member.size:write_member(path,tar.extractfile(member).read())
                        result=probe_audio(path);kept+=1
                        with lock:
                            probes[key]=result;atomic(probes_path,probes)
                    atomic(receipt,{'seen':seen,'selected':kept,'bytes_transferred':remote.transferred,'archive_bytes':remote.size,'complete':False,'time':time.time()})
        atomic(receipt,{'seen':seen,'selected':kept,'bytes_transferred':remote.transferred,'archive_bytes':remote.size,'complete':True,'time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for future in concurrent.futures.as_completed([pool.submit(scan,a) for a in archives]):future.result()
    missing=sorted(wanted-set(probes))
    if missing:raise ValueError(f'{len(missing)} X-Stream sources not probed')
    selected=[q for q in questions if any(probes[v['path']]['audio'] for v in q['streams'])]
    for q in selected:
        q['audio_status']='verified_audio_track'
        for v in q['streams']:v['has_audio']=probes[v['path']]['audio']
    atomic(BASE/'X-Stream/questions.json',selected)
    retained={v['path'] for q in selected for v in q['streams']}
    for path in wanted-retained:
        # Only newly downloaded non-audio candidate media; old experiments are untouched.
        (dest/path).unlink(missing_ok=True)
    summary=json.loads((BASE/'X-Stream/selection_summary.json').read_text());summary.update(audio_filter='complete',audio_selected_questions=len(selected),both_streams_audio_questions=sum(all(v['has_audio'] for v in q['streams']) for q in selected))
    atomic(BASE/'X-Stream/selection_summary.json',summary)

def download(name,questions,workers=8):
    entry=manifest()['datasets'][name]
    if name=='CVBench':
        url=hf_url(entry,'CVBench.zip');mapping={v['path'].removeprefix('CVBench/'):BASE/name/'media'/v['path'] for q in questions for v in q['streams']}
        selective_zip(name,url,40411543515,mapping,workers)
    elif name=='SYNCR':
        paths={v['path'] for q in questions for v in q['streams']};info=json.loads((SOURCES/'SYNCR.repo.json').read_text());sizes={f['rfilename']:f['size'] for f in info['siblings']}
        def tar(kind):
            wanted={p.removeprefix('videos/'):BASE/name/'media'/p for p in paths if p.startswith('videos/'+kind+'/')}
            selective_tar('SYNCR_'+kind,hf_url(entry,kind+'.tar'),sizes[kind+'.tar'],wanted)
        def clevrer():
            url='https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip';remote=HTTPRangeFile(url)
            wanted={Path(p).name:p for p in paths if p.startswith('videos/clvr/')}
            with zipfile.ZipFile(remote) as z:
                mapping={i.filename:BASE/name/'media'/wanted[Path(i.filename).name] for i in z.infolist() if Path(i.filename).name in wanted and not i.filename.startswith('__MACOSX/')}
            assert len(mapping)==len(wanted)
            selective_zip('SYNCR_CLEVRER',url,remote.size,mapping,workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(tar,'habitat_data'),pool.submit(tar,'kubric_data'),pool.submit(clevrer)]
            for f in futures:f.result()
    elif name=='MVPBench':
        for a in entry['archives']:
            mapping={v['archive_member']:BASE/name/'media'/v['path'] for q in questions for v in q['streams'] if v['archive_id']==a['id']}
            if mapping:
                url=f"https://drive.usercontent.google.com/download?id={a['id']}&export=download&confirm=t"
                selective_zip('MVPBench_'+a['id'],url,a['size'],mapping,workers)
    elif name=='X-Stream':xstream(questions)
    else:raise NotImplementedError(name)
