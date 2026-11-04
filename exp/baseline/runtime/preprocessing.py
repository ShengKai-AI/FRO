"""Shared label-blind video preprocessing and paired context-budget validation."""
import hashlib,json,math,os,shutil,subprocess
from pathlib import Path
import av,numpy as np,torch
from qwen_vl_utils.vision_process import smart_resize
from .prompt import SYSTEM

def tool(name):
    value=os.environ.get('FRO_'+name.upper()) or shutil.which(name)
    if not value:raise FileNotFoundError(f'Install {name} or set FRO_{name.upper()}')
    return value

def prepare_stream(processor,source,question,frame_cap,start=None,end=None,audio_path=None):
    source=Path(source)
    probe=json.loads(subprocess.check_output([tool('ffprobe'),'-v','error','-show_streams','-show_format','-of','json',str(source)]))
    video=next(x for x in probe['streams'] if x['codec_type']=='video');duration=float(probe['format']['duration'])
    start=max(0.,float(start or 0));end=min(duration,float(end or duration))
    if not end>start:raise ValueError('Invalid benchmark temporal bounds')
    audio_source=Path(audio_path) if audio_path else source
    has_audio=bool(audio_path) or any(s['codec_type']=='audio' for s in probe['streams'])
    if has_audio and end-start>300:
        raise NotImplementedError('Long-audio segment timestamp remapping requires validation; no silent truncation or concatenation')
    count=max(2,min(frame_cap,2*max(1,round(end-start))))
    count-=count%2
    targets=np.linspace(start,max(start,end-.04),count)
    height,width=smart_resize(int(video['height']),int(video['width']),factor=28,min_pixels=3136,max_pixels=100352)
    frames=[];times=[]
    with av.open(str(source)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2
        for stamp in targets:
            container.seek(max(0,int(stamp/float(stream.time_base))),stream=stream,backward=True)
            selected=None
            for frame in container.decode(stream):
                selected=frame
                if frame.time is not None and frame.time+1e-6>=stamp:break
            if selected is None:raise RuntimeError('No video frame at requested time')
            frames.append(selected.reformat(width=width,height=height).to_ndarray(format='rgb24'));times.append(float(selected.time))
    visual=np.stack(frames);audio=None
    if has_audio:
        audio=np.frombuffer(subprocess.check_output([tool('ffmpeg'),'-v','error','-threads','1','-ss',str(start),'-i',str(audio_source),'-t',str(end-start),'-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']),dtype='<f4').copy()*.5
        if not len(audio):raise ValueError('Empty decoded audio track')
    text=processor.apply_chat_template([SYSTEM,{'role':'user','content':[{'type':'video','video':str(source)},{'type':'text','text':question}]}],tokenize=False,add_generation_prompt=True)
    inputs=processor(text=text,videos=[torch.from_numpy(visual).permute(0,3,1,2)],audio=[audio] if audio is not None else None,
                     return_tensors='pt',padding=True,use_audio_in_video=has_audio,fps=count/(end-start),sampling_rate=16000,do_resize=False,do_sample_frames=False,audio_kwargs={'truncation':False})
    if audio is not None and abs(int(inputs['feature_attention_mask'].sum())-math.ceil(len(audio)/160))>2:raise ValueError('Processor truncated audio')
    inputs={k:v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and k in ('pixel_values_videos','input_features') else v for k,v in inputs.items()}
    with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'inputs':inputs,'visual':visual,'audio':audio,'record':{'question':question,'frame_count':count,'frame_seconds':times,'audio_present':has_audio,'source_media_sha256':sha,'start':start,'end':end,'audio_gain':.5,'audio_path':str(audio_source) if has_audio else None}}
