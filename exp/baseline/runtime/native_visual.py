"""Apply published visual branches when no audio tokens are present."""
import torch

@torch.inference_mode()
def compress_visual(ns, data, inputs, need=None, aux=None):
    thinker=ns['thinker'];method=ns['m'];ids=inputs['input_ids'][0]
    embeds=thinker.get_input_embeddings()(inputs['input_ids'])
    visual=thinker.get_video_features(inputs['pixel_values_videos'],inputs['video_grid_thw']).to(embeds)
    video=ids==thinker.config.video_token_id
    embeds[0,video]=visual
    pos=inputs.get('position_ids')
    if pos is None:
        pos,_=thinker.get_rope_index(inputs['input_ids'],None,inputs['video_grid_thw'],inputs['attention_mask'],False,None,inputs['video_second_per_grid'])
    if method=='omniselect':
        from exp.baseline.methods.omniselect_groups import omniselect
        keep=omniselect(inputs['video_grid_thw'],embeds,None,visual,torch.empty(0,device=embeds.device),torch.nn.functional.normalize(visual,dim=-1),ids,need)
    elif method=='omnizip':
        from exp.baseline.methods.omnizip_core import omnizip
        c=thinker.omnizip_config
        embeds,keep=omnizip(embeds,torch.empty(0,device=embeds.device),ids,thinker.config.audio_token_id,thinker.config.video_token_id,inputs['video_grid_thw'],merging_ratio_audio=c['rho_audio'],merging_ratio_v=c['rho_video'],contextual_ratio=c['contextual_ratio'],g=c['g'])
    elif method=='omnipack':
        from omnipack.pre_llm_units import video_spatiotemporal_merge
        c=thinker.omnipack_config
        embeds,keep=video_spatiotemporal_merge(inputs_embeds=embeds,input_ids=inputs['input_ids'],video_attn_mean=thinker.visual._vz_attn_mean,video_token_id=thinker.config.video_token_id,video_ratio=c.video_encoder_ratio,spatial_merge_unit=thinker.config.vision_config.spatial_merge_size**2,video_grid_thw=inputs['video_grid_thw'],grid_in_window=c.grid_in_window)
    else:raise ValueError(method)
    keep=keep.flatten().bool()
    assert keep[~video].all()
    return {'embeds':embeds[:,keep,:],'positions':pos[...,keep],'ids':ids[keep],'stage1':keep.cpu(),'final':keep.cpu(),'original_ids':ids.cpu(),'aux':dict(aux or {},modality_adapter='published_visual_branch_no_audio_tokens')}
