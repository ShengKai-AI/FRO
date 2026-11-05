"""Additional encoder attention statistics, excluding ordinary encoder inference."""
import inspect

def install(thinker,method):
    if method=='omnizip':
        def audio_hook(module,args,kwargs):
            h=kwargs.get('hidden_states',args[0] if args else None)
            n,d=h.shape;thinker.audio_tower._fro_encoder_aux=4*n*d*d+2*n*n*d
        thinker.audio_tower.layers[-1].self_attn.register_forward_pre_hook(audio_hook,with_kwargs=True)
    if method in ('omnipack','visionzip','fastvid'):
        def reset(module,*_):module._fro_encoder_aux=0
        for owner,blocks,attribute in [(thinker.visual,thinker.visual.blocks,'attn')]+([(thinker.audio_tower,thinker.audio_tower.layers,'self_attn')] if method=='omnipack' else []):
            owner.register_forward_pre_hook(reset)
            for block in blocks:
                module=getattr(block,attribute);signature=inspect.signature(module.forward)
                def hook(module,args,kwargs,owner=owner,signature=signature):
                    bound=signature.bind(*args,**kwargs);bound.apply_defaults();a=bound.arguments
                    if not a.get('return_logits'):return
                    window=a.get('window_tokens',a.get('audio_window_tokens'))
                    if not window:return
                    n,d=a['hidden_states'].shape;window=int(window)
                    owner._fro_encoder_aux+=2*d*((n//window)*window**2+(n%window)**2)
                module.register_forward_pre_hook(hook,with_kwargs=True)
