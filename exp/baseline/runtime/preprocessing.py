"""Shared label-blind video preprocessing and paired context-budget validation."""
import hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
import av,numpy as np,torch
from qwen_vl_utils.vision_process import smart_resize
from .prompt import SYSTEM

def tool(name):
    value=os.environ.get('FRO_'+name.upper()) or shutil.which(name)
    if not value:raise FileNotFoundError(f'Install {name} or set FRO_{name.upper()}')
    return value

def prepare_stream(processor,source,question,frame_cap,start=None,end=None,audio_path=None,sampling_scale=1.,audio_enabled=True,fixed_frames=False,max_pixels=100352,long_edge=None,native_frames=False):
    source=Path(source)
    directory=source.is_dir();files=[]
    if directory:
        from PIL import Image
        files=sorted(source.glob('*.jpg'));first=Image.open(files[0]);video={'height':first.height,'width':first.width};duration=len(files)/2.;has_audio=False;source_fps=2.
    else:
        probe=json.loads(subprocess.check_output([tool('ffprobe'),'-v','error','-show_streams','-show_format','-of','json',str(source)]))
        video=next(x for x in probe['streams'] if x['codec_type']=='video');duration=float(probe['format']['duration']);has_audio=bool(audio_path) or any(s['codec_type']=='audio' for s in probe['streams'])
        from fractions import Fraction
        source_fps=float(Fraction(video['avg_frame_rate']))
    start=max(0.,float(start or 0));end=min(duration,float(end or duration))
    if not end>start:raise ValueError('Invalid benchmark temporal bounds')
    audio_source=Path(audio_path) if audio_path else source;has_audio=has_audio and audio_enabled
    count=frame_cap if fixed_frames else max(2,min(frame_cap,round((source_fps if native_frames else 2)*(end-start))))
    count=max(2,int(count*sampling_scale));count-=count%2
    targets=np.linspace(start,max(start,end-.04),count)
    h,w=int(video['height']),int(video['width'])
    if long_edge:
        scale=min(1.,long_edge/max(h,w));h=max(28,int(h*scale));w=max(28,int(w*scale))
        max_pixels=long_edge**2
    height,width=smart_resize(h,w,factor=28,min_pixels=3136,max_pixels=max_pixels)
    frames=[];times=[];decode_started=time.perf_counter();seek_count=0;sampled_indices=[]
    if directory:
        sampled_indices=np.linspace(0,len(files)-1,count,dtype=int).tolist()
        for index in sampled_indices:
            with Image.open(files[index]) as img:frames.append(np.array(img.convert('RGB').resize((width,height))))
            times.append(index/source_fps)
    else:
        with av.open(str(source)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2
            decoder=None;previous_target=None;last_frame=None
            for stamp in targets:
                if decoder is None or stamp-previous_target>2.:
                    container.seek(max(0,int(stamp/float(stream.time_base))),stream=stream,backward=True)
                    decoder=container.decode(stream);last_frame=None;seek_count+=1
                selected=last_frame
                if selected is None or selected.time is None or selected.time+1e-6<stamp:
                    for frame in decoder:
                        selected=frame
                        if frame.time is not None and frame.time+1e-6>=stamp:break
                if selected is None:raise RuntimeError('No video frame at requested time')
                frames.append(selected.reformat(width=width,height=height).to_ndarray(format='rgb24'));times.append(float(selected.time))
                last_frame=selected;previous_target=stamp
    decode_seconds=time.perf_counter()-decode_started
    visual=np.stack(frames);audio=None;segments=[]
    if has_audio:
        audio=np.frombuffer(subprocess.check_output([tool('ffmpeg'),'-v','error','-threads','1','-ss',str(start),'-i',str(audio_source),'-t',str(end-start),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']),dtype='<f4').copy()*.5
        if not len(audio):raise ValueError('Empty decoded audio track')
        audio_budget=min(300.,end-start)*sampling_scale
        if len(audio)/16000>audio_budget+.04:
            # Segment lengths are multiples of the 40 ms audio-token period.
            n=max(1,min(60,math.ceil(audio_budget/5)))
            length=max(640,int(audio_budget*16000/n)//640*640)
            starts=np.linspace(0,max(0,len(audio)-length),n).astype(np.int64)
            starts=(starts//640)*640
            segments=[(int(s),min(len(audio),int(s)+length)) for s in starts]
            audio=np.concatenate([audio[s:e] for s,e in segments])
    text=processor.apply_chat_template([SYSTEM,{'role':'user','content':[{'type':'video','video':str(source)},{'type':'text','text':question}]}],tokenize=False,add_generation_prompt=True)
    inputs=processor(text=text,videos=[torch.from_numpy(visual).permute(0,3,1,2)],audio=[audio] if audio is not None else None,
                     return_tensors='pt',padding=True,use_audio_in_video=has_audio,fps=count/(end-start),sampling_rate=16000,do_resize=False,do_sample_frames=False,audio_kwargs={'truncation':False})
    if audio is not None and abs(int(inputs['feature_attention_mask'].sum())-math.ceil(len(audio)/160))>2:raise ValueError('Processor truncated audio')
    inputs={k:v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and k in ('pixel_values_videos','input_features') else v for k,v in inputs.items()}
    if directory:
        sha=hashlib.sha256(b''.join(files[i].read_bytes() for i in sampled_indices)).hexdigest()
    else:
        with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    result={'inputs':inputs,'visual':visual,'audio':audio,'record':{'question':question,'frame_count':count,'frame_seconds':times,'audio_present':has_audio,'source_media_sha256':sha,'start':start,'end':end,'audio_gain':.5,'audio_path':str(audio_source) if has_audio else None,'sampling_scale':sampling_scale}}
    result['record'].update(video_decode_seconds=decode_seconds,video_seek_count=seek_count,source_fps=source_fps,frame_indices=sampled_indices,original_height=int(video['height']),original_width=int(video['width']),processed_height=height,processed_width=width,audio_enabled=audio_enabled)
    if segments:
        from .timeline import AUDIO,block_layout,prompt_positions,VISION_START,VISION_END
        na=int((inputs['input_ids']==AUDIO).sum())
        boundaries=np.cumsum([0]+[e-s for s,e in segments])
        offsets=np.arange(na)*640
        indices=np.minimum(np.searchsorted(boundaries[1:],offsets,side='right'),len(segments)-1)
        seconds=(np.array([s for s,e in segments])[indices]+offsets-boundaries[indices])/16000
        result['record'].update(audio_token_seconds=seconds.tolist(),audio_segments_seconds=[[start+s/16000,start+e/16000] for s,e in segments],timeline_version=1)
        block,positions=block_layout(result,reorder=True)
        ids=inputs['input_ids'];begin=int((ids[0]==VISION_START).nonzero()[0]);finish=int((ids[0]==VISION_END).nonzero()[0])+1
        ids[0,begin:finish]=block
        inputs['position_ids']=prompt_positions(ids,[(block,positions)])
    return result
