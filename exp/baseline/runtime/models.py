"""Load pinned Omni implementations; each worker owns exactly one model."""
def load(method, model_name, ratio, max_new_tokens=32):
    """Independent AV feature compression then joint decoding; inference never reads gold.
    OmniPack adaptation: native per-stream Stage-I merges, followed by native joint
    Stage-II hidden-state selection after layer 18. No independent decoder replay.
    """
    import os,sys,json,time,gc,importlib,argparse,traceback
    from pathlib import Path
    import torch,numpy as np
    from exp.baseline.runtime.paths import repo, model_path
    from transformers import Qwen2_5OmniConfig,Qwen2_5OmniProcessor,PreTrainedModel
    m=method
    weights=model_path(model_name)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(4);torch.manual_seed(20261030);torch.cuda.manual_seed_all(20261030)
    if m=='omniselect':
     sys.path.insert(0,str(repo('OmniSelect')/'OmniSelect'))
     from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
     sys.path.insert(0,str(repo('OmniPack')))
     from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    cfg=Qwen2_5OmniConfig.from_pretrained(weights,local_files_only=True);cfg.enable_audio_output=False
    model,info=PreTrainedModel.from_pretrained.__func__(Qwen2_5OmniForConditionalGeneration,str(weights),config=cfg,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='flash_attention_2',local_files_only=True,output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys'];assert all(k.startswith(('talker.','token2wav.')) for k in info['unexpected_keys']);model.eval()
    thinker=model.thinker;plain_text_forward=type(thinker.model).forward
    selector=None
    if m in ['omnizip','omnipack']:
     from baselines.utils import apply_zip_method_patch
     apply_zip_method_patch(model,m,ratio,ratio,None)
     if m=='omnipack':
      thinker.omnipack_config.video_encoder_ratio=thinker.omnipack_config.audio_encoder_ratio=min(1.,2*ratio)
      thinker.omnipack_config.video_stage2_keep_ratio=thinker.omnipack_config.audio_stage2_keep_ratio=.5
      thinker.omnipack_config.stage2_layer=26 if model_name=='3B' else 18
    if m=='omniselect':
     from exp.baseline.methods.omniselect import Selector
     selector=Selector(ratio);selector.config=thinker.config
    p=Qwen2_5OmniProcessor.from_pretrained(weights,local_files_only=True,use_fast=False)
    state={}
    class Captured(Exception):pass
    def mask(name,x):state[name]=x.detach().flatten().clone()
    if m=='omnizip':
     mod=importlib.import_module('baselines.omnizip.modeling_qwen2_5_omni_omnizip')
     from exp.baseline.methods.omnizip_core import omnizip as orig
     def wrap(*a,**kw):
      out=orig(*a,**kw);mask('stage1',out[1]);return out
     mod.omnizip=wrap
    elif m=='omniselect':
     mod=importlib.import_module('OmniSelect')
     from exp.baseline.methods.omniselect_groups import omniselect as orig
     def wrap(*a,**kw):
      out=orig(*a,**kw);mask('stage1',out);return out
     mod.omniselect=wrap
    elif m=='omnipack':
     mod=importlib.import_module('omnipack.modeling_qwen2_5_omni_omnipack');ov=mod.video_spatiotemporal_merge;oa=mod.audio_temporal_merge;os2=mod.stage2_token_selection
     def wv(*a,**kw):
      out=ov(*a,**kw);mask('v',out[1]);return out
     def wa(*a,**kw):
      out=oa(*a,**kw);mask('a',out[1]);state['stage1']=state['v']&state['a'];return out
     def ws(*a,**kw):
      from exp.baseline.metrics.operations import MatrixCounter
      counter=MatrixCounter();torch.cuda.synchronize();t=time.perf_counter()
      with counter:out=os2(*a,**kw)
      state['inner_auxiliary_flops']=state.get('inner_auxiliary_flops',0)+counter.total
      torch.cuda.synchronize();state['stage2_seconds']=time.perf_counter()-t;mask('joint_stage2',out[-1]);return out
     mod.video_spatiotemporal_merge=wv;mod.audio_temporal_merge=wa;mod.stage2_token_selection=ws
    
    if m=='fastv':
     from types import MethodType
     from baselines.fastv.modeling_qwen2_5_omni_fastv import Qwen2_5OmniThinkerTextModel_forward_fastv
     fastmod=importlib.import_module('baselines.fastv.modeling_qwen2_5_omni_fastv')
     original_drop=fastmod.fastv_global_drop_tokens
     def capture_drop(*args,**kwargs):
      from exp.baseline.metrics.operations import MatrixCounter
      counter=MatrixCounter()
      with counter:result=original_drop(*args,**kwargs)
      state['inner_auxiliary_flops']=state.get('inner_auxiliary_flops',0)+counter.total
      mask('joint_stage2',result[-1]);return result
     fastmod.fastv_global_drop_tokens=capture_drop
     thinker.model.forward=MethodType(Qwen2_5OmniThinkerTextModel_forward_fastv,thinker.model)
    
    def first_layer(module,argv,kw):
     state['embeds']=kw.get('hidden_states',argv[0] if argv else None).detach().clone();state['positions']=kw['position_ids'].detach().clone()
     raise Captured()
    
    @torch.inference_mode()
    def compress(data):
     state.clear();inputs={k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,non_blocking=True) for k,v in data['inputs'].items()};ids=inputs['input_ids'][0]
     state['stage1']=torch.ones_like(ids,dtype=torch.bool);extra={};aux={}
     if selector is not None:
      # Calibration changes only retention, so question-specific AudioCLIP scores
      # are immutable across its trials. Attribute their full logical cost to
      # every selected invocation even when reuse avoids recalculating them.
      from exp.baseline.metrics.operations import MatrixCounter
      if os.environ.get('FRO_DISABLE_SELECTOR_CACHE')=='1':data.pop('_selector_prepared',None)
      if '_selector_prepared' not in data:
       semantic_counter=MatrixCounter()
       with semantic_counter:need,semantic_aux=selector.prepare(data,inputs)
       data['_selector_prepared']=(need,dict(semantic_aux),semantic_counter.total)
       reused_semantic=0
      else:reused_semantic=data['_selector_prepared'][2]
      need,semantic_aux,semantic_cost=data['_selector_prepared']
      need=dict(need);need['args']=argparse.Namespace(prune=True,prune_ratio_a=1-selector.ratio,prune_ratio_v=1-selector.ratio)
      aux=dict(semantic_aux,reused_selector_auxiliary_flops=reused_semantic,semantic_auxiliary_flops=semantic_cost)
      extra['prune_need']=need
     if data.get('audio') is None and m in ('omniselect','omnizip','omnipack'):
      from .native_visual import compress_visual
      return compress_visual(locals_ns,data,inputs,extra.get('prune_need'),aux)
     thinker.nframes=data['record']['frame_count'];handle=thinker.model.layers[0].register_forward_pre_hook(first_layer,with_kwargs=True)
     try:
      try:thinker(**inputs,use_audio_in_video=data['record'].get('audio_present',True),use_cache=False,return_dict=True,**extra)
      except Captured:pass
      else:raise RuntimeError('Feature capture did not stop decoder')
     finally:handle.remove()
     stage1=state['stage1'];assert state['embeds'].shape[1]==stage1.sum()
     local=state.get('stage2',torch.ones(int(stage1.sum()),dtype=torch.bool,device='cuda'))
     final=torch.zeros_like(stage1);final[stage1]=local
     assert torch.all(final[~((ids==thinker.config.audio_token_id)|(ids==thinker.config.video_token_id))])
     result={'embeds':state['embeds'][:,local,:], 'positions':state['positions'][...,local], 'ids':ids[final], 'stage1':stage1.cpu(), 'final':final.cpu(), 'original_ids':ids.cpu(),'aux':aux}
     del inputs;state.clear();return result
    
    @torch.inference_mode()
    def decode(embeds,pos,fixed=False):
     lens=[]
     def hook(module,argv,kw):
      h=kw.get('hidden_states',argv[0] if argv else None)
      lens.append(int(h.shape[1]))
     handles=[layer.register_forward_pre_hook(hook,with_kwargs=True) for layer in thinker.model.layers]
     try:
      out=(thinker.model.forward if m in ('omnipack','fastv') else lambda **kw: plain_text_forward(thinker.model,**kw))(inputs_embeds=embeds,position_ids=pos,attention_mask=torch.ones(embeds.shape[:2],device='cuda',dtype=torch.long),use_cache=True,return_dict=True)
     finally:
      for h in handles:h.remove()
     token=thinker.lm_head(out.last_hidden_state[:,-1:]).argmax(-1);cache=out.past_key_values;tokens=[];start=int(pos.max())+1
     state['layer_kv_prefill_lengths']=[int(cache.get_seq_length(i)) for i in range(len(thinker.model.layers))]
     torch.cuda.synchronize();first=time.perf_counter()
     eos=thinker.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);stop=False;budget=max_new_tokens
     for step in range(budget):
      value=int(token.item())
      if not stop:tokens.append(value)
      if value in eos:
       stop=True
       if not fixed:break
      if step+1==budget:break
      n=cache.get_seq_length();out=(thinker.model.forward if m in ('omnipack','fastv') else lambda **kw: plain_text_forward(thinker.model,**kw))(input_ids=token,position_ids=torch.full((3,1,1),start+step,device='cuda',dtype=torch.long),attention_mask=torch.ones((1,n+1),device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True,return_dict=True)
      token=thinker.lm_head(out.last_hidden_state[:,-1:]).argmax(-1);cache=out.past_key_values
     return p.decode(tokens,skip_special_tokens=True),tokens,first,lens,stop
    
    @torch.inference_mode()
    def joint_features(data,parts):
     j={k:v.to('cuda') for k,v in data['joint'].items()};ids=j['input_ids'][0];emb=thinker.get_input_embeddings()(ids[None]);keep=torch.ones_like(ids,dtype=torch.bool)
     if 'position_ids' in j:pos=j['position_ids']
     else:pos,delta=thinker.get_rope_index(j['input_ids'],None,j['video_grid_thw'],j['attention_mask'],True,j['feature_attention_mask'].sum(-1),j['video_second_per_grid'])
     for kind in ['audio','video']:
      tokenid=getattr(thinker.config,kind+'_token_id');slots=(ids==tokenid).nonzero().flatten();offset=0
      for part in parts:
       original_kind=part['original_ids']==tokenid;count=int(original_kind.sum());these=slots[offset:offset+count];offset+=count
       selected=part['final'][original_kind].to('cuda');keep[these]=selected
       values=part['embeds'][0,part['ids']==tokenid,:]
       assert len(values)==int(selected.sum());emb[0,these[selected],:]=values
      assert offset==len(slots)
     return emb[:,keep,:],pos[...,keep],keep.cpu()
    
    
    
    
    locals_ns=locals()
    return locals_ns
