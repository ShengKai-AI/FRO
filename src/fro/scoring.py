"""Inference-only cross-stream scoring; original embeddings are replayed after selection.

score_layer is ONE-BASED: 2 executes block 1 and uses block 2 Q/K;
5 executes blocks 1..4 and uses block 5 Q/K. This is a separate scorer
prefix, NOT pruning inside the joint decoder. No prefix cache is reused.
"""
from dataclasses import dataclass, asdict
import sys
import time
import math
import torch
import torch.nn.functional as F

from .selection import native_question_attention, rank01, select_with_coverage
from .rotary import apply_multimodal_rotary_pos_emb, repeat_kv
from .temporal import (head_attention,neutral_rank,contrast_scores,head_selected_scores,value_scores,
                        option_dispersion,temporal_novelty,select_with_novelty,temporal_smooth,diversity_penalty,quality_aggregate)


@dataclass(frozen=True)
class ScoreConfig:
    score_layer: int = 2
    bridge_weight: float = .3
    support_topk: int = 3
    anchors_per_modality: int = 16
    coverage_fraction: float = .2
    query_gate_floor: float = .5

    module: str = 'none'
    selected_heads: int = 8
    diagnostic_mix: float = .5
    novelty_fraction: float = .2

    query_mode: str = 'all'
    head_aggregation: str = 'mean'
    head_temperature: float = 1.
    head_token_boost: float = .5
    value_power: float = 1.
    value_cap_quantile: float = 1.
    novelty_source: str = 'encoder'
    bridge_mode: str = 'all'
    reciprocal_fraction: float = .3
    smooth_mix: float = 0.
    diversity_strength: float = 0.

    def validate(self, layer_count):
        if self.head_aggregation not in ('mean','paq','concentration','specialization'):
            raise ValueError('Unknown aggregation')
        if not .01<=self.head_temperature<=10 or not 0<=self.head_token_boost<=2:
            raise ValueError('Invalid head weight parameter')
        if self.head_aggregation!='mean' and self.module not in ('none','value_weight','temporal_novelty','value_novelty'):
            raise ValueError('Ambiguous multiple query aggregation recipes')
        if self.query_mode not in ('all','own','both_content','own_stem','both_stems','own_options'):
            raise ValueError('Unknown question scope')
        if not math.isfinite(self.value_power) or not 0 < self.value_power <= 4:
            raise ValueError('Invalid value exponent')
        if not .5 <= self.value_cap_quantile <= 1 or not 0 <= self.smooth_mix <= 1:
            raise ValueError('Invalid cap or smoothing')
        if not 0 <= self.diversity_strength <= 1 or not 0 < self.reciprocal_fraction <= 1:
            raise ValueError('Invalid diversity or reciprocal fraction')
        if self.novelty_source not in ('encoder','prefix') or self.bridge_mode not in ('all','same','cross','balanced','reciprocal'):
            raise ValueError('Unknown feature/support variant')
        if self.module not in ('none','query_contrast','head_selection','value_weight','temporal_novelty','option_evidence','value_novelty'):
            raise ValueError('Unknown explicit recipe')
        if self.score_layer != 2: raise ValueError('Round2 fixes score_layer=2')
        if self.selected_heads<1 or not 0<=self.diagnostic_mix<=1 or not 0<=self.novelty_fraction<=1:
            raise ValueError('Invalid diagnostic settings')
        if not 1 <= self.score_layer <= layer_count:
            raise ValueError('score_layer must be a one-based decoder layer')
        if not 0 <= self.bridge_weight <= 1 or not 0 <= self.query_gate_floor <= 1:
            raise ValueError('weights must be in [0, 1]')
        if self.support_topk < 1 or self.anchors_per_modality < 1:
            raise ValueError('topk and anchor count must be positive')
        if not 0 <= self.coverage_fraction <= 1:
            raise ValueError('coverage fraction must be in [0, 1]')


def gated_support(similarity, anchor_relevance, own_relevance, config):
    """Absolute support: all zero/negative affinities give exactly zero support.

    Unlike ranking a maximum, weak affinities are not inflated to rank 1.
    The top-k average reduces sensitivity to a single accidental match.
    """
    if similarity.shape[-1] == 0:
        return torch.zeros_like(own_relevance)
    weighted = similarity.float().clamp(0, 1) * (.5 + .5 * anchor_relevance)[None]
    support = weighted.topk(min(config.support_topk, weighted.shape[-1]), dim=-1).values.mean(-1)
    gate = config.query_gate_floor + (1 - config.query_gate_floor) * own_relevance
    return support * gate


@torch.inference_mode()
def prefix_hidden(part, thinker, score_layer):
    h, pos = part['embeds'], part['positions']
    text = thinker.model
    if getattr(text.config, '_attn_implementation', None) != 'flash_attention_2':
        raise ValueError('This no-padding prefix implementation requires flash_attention_2')
    pe = text.rotary_emb(h, pos)
    cache_pos = torch.arange(h.shape[1], device=h.device)
    for layer in text.layers[:score_layer - 1]:
        h = layer(h, attention_mask=None, position_ids=pos, past_key_value=None,
                  output_attentions=False, use_cache=False,
                  cache_position=cache_pos, position_embeddings=pe)[0]
    return h, pe


@torch.inference_mode()
def prune(parts, thinker, config=ScoreConfig(), ratio=.3):
    config.validate(len(thinker.model.layers))
    if len(parts) != 2 or not 0 < ratio <= 1:
        raise ValueError('Requires exactly two unpruned streams and 0 < ratio <= 1')
    groups, scores, features, diagnostics = [], [], [], []
    for part in parts:
        ids = part['original_ids'].to(part['embeds'].device)
        if part['embeds'].shape[1] != ids.numel():
            raise ValueError('Scorer requires original, unpruned embeddings')
        g = torch.full_like(ids, -1)
        g[ids == thinker.config.audio_token_id] = 0
        g[ids == thinker.config.video_token_id] = 1
        if not bool((g == 1).any()):
            raise ValueError('Each stream must contain video')
        qpos = ((torch.arange(len(ids), device=ids.device) > int((g >= 0).nonzero()[-1]))
                & (ids < 151643)).nonzero().flatten()
        if config.query_mode!='all': qpos=part['selected_queries']
        if not len(qpos):
            raise ValueError('Missing question tokens after AV tokens')
        torch.cuda.synchronize(ids.device)
        start = time.perf_counter()
        h, pe = prefix_hidden(part, thinker, config.score_layer)
        torch.cuda.synchronize(ids.device)
        prefix_seconds = time.perf_counter() - start
        layer=thinker.model.layers[1]
        detail={}
        if config.module in ('head_selection','value_weight','value_novelty') or config.head_aggregation!='mean':
            head_result=head_attention(layer,h,pe,qpos,need_value=config.module in ('value_weight','value_novelty'),quality_groups=g if config.head_aggregation=='paq' else None)
            heads,used,value_norm=head_result[:3]
            quality=head_result[3] if len(head_result)==4 else None
            if config.head_aggregation!='mean':
                raw,weights=quality_aggregate(heads,g,config.head_aggregation,quality,config.head_temperature,config.head_token_boost,value_norm)
                detail['head_aggregation']=config.head_aggregation
                detail['head_weights_by_modality']=weights
                if quality is not None:detail['head_paq']=quality.cpu().tolist()
            elif config.module=='head_selection':
                raw,chosen=head_selected_scores(heads,g,config.selected_heads)
                detail['selected_heads_by_modality']=chosen
            else:
                if config.value_cap_quantile < 1:
                    value_norm=value_norm.clone()
                    for mod in (0,1):
                        mask=g==mod
                        cap=torch.quantile(value_norm[:,mask],config.value_cap_quantile,dim=1,keepdim=True)
                        value_norm[:,mask]=torch.minimum(value_norm[:,mask],cap)
                raw=value_scores(heads,value_norm.pow(config.value_power))
                detail['mean_value_l1_norm']=float(value_norm.mean())
        else:
            raw,used=native_question_attention(layer,h,pe,qpos,apply_multimodal_rotary_pos_emb,repeat_kv)
        rel=torch.zeros_like(raw)
        for mod in (0,1): rel[g==mod]=rank01(raw[g==mod])
        if config.module=='query_contrast':
            content,cused=native_question_attention(layer,h,pe,part['content_queries'],apply_multimodal_rotary_pos_emb,repeat_kv)
            generic,gused=native_question_attention(layer,h,pe,part['generic_queries'],apply_multimodal_rotary_pos_emb,repeat_kv)
            diagnostic=contrast_scores(content,generic,g)
            detail.update(content_query_positions=cused.cpu().tolist(),generic_query_positions=gused.cpu().tolist())
        elif config.module=='option_evidence':
            maps=[];positions=[]
            for query in part['option_queries']:
                if len(query)>16:query=query[torch.linspace(0,len(query)-1,16,device=query.device).round().long()]
                a,used_option=native_question_attention(layer,h,pe,query,apply_multimodal_rotary_pos_emb,repeat_kv)
                maps.append(a);positions.append(used_option.cpu().tolist())
            diagnostic=option_dispersion(torch.stack(maps),g)
            detail.update(option_question_slot=part['source_slot'],option_query_positions=positions)
        if config.module in ('query_contrast','option_evidence'):
            for mod in (0,1):
                values=diagnostic[g==mod]
                if float(values.max()-values.min())>1e-12:
                    rel[g==mod]=(1-config.diagnostic_mix)*rel[g==mod]+config.diagnostic_mix*neutral_rank(values)
            raw=rel.clone()
        if config.module in ('temporal_novelty','value_novelty'):
            part['novelty']=temporal_novelty(part['embeds'][0] if config.novelty_source=='encoder' else h[0],g,part['video_shape'])
            detail['mean_encoder_feature_change']=float(part['novelty'][g>=0].mean())
        groups.append(g)
        scores.append((raw, rel))
        features.append(h[0])
        diagnostics.append({'score_layer_one_based': config.score_layer,
                            'prefix_blocks': config.score_layer - 1,
                            'prefix_token_layer_exposure': (config.score_layer - 1) * len(ids),
                            'prefix_seconds': prefix_seconds,
                            'question_positions_used': used.cpu().tolist(), **detail})
    # Same centering as the frozen baseline; feature depth is configurable.
    z = [h.float() - h.float().mean(-1, keepdim=True) for h in features]
    for mod in (0, 1):
        available = torch.cat([h[g == mod] for h, g in zip(z, groups)])
        if not len(available): continue
        mean = available.mean(0, keepdim=True)
        for h, g in zip(z, groups):
            h[g == mod] -= mean
    z = [F.normalize(h, dim=-1).to(features[0].dtype) for h in z]
    anchors = []
    for g, (raw, rel) in zip(groups, scores):
        aa = []
        for mod in (0, 1):
            ii = (g == mod).nonzero().flatten()
            aa.append(ii[torch.topk(raw[ii], min(config.anchors_per_modality, len(ii))).indices])
        anchors.append(torch.cat(aa))
    for i, (part, g, (_, rel), diag) in enumerate(zip(parts, groups, scores, diagnostics)):
        other = 1 - i
        bridge = torch.zeros_like(rel)
        aa = anchors[other]
        for mod in (0, 1):
            ii = (g == mod).nonzero().flatten()
            sim = z[i][ii] @ z[other][aa].T
            if config.bridge_mode in ('same','cross'):
                valid=groups[other][aa]==mod
                if config.bridge_mode=='cross': valid=~valid
                bridge[ii]=gated_support(sim[:,valid],scores[other][1][aa[valid]],rel[ii],config)
            elif config.bridge_mode=='balanced':
                vals=[]
                for target_mod in (0,1):
                    valid=groups[other][aa]==target_mod
                    vals.append(gated_support(sim[:,valid],scores[other][1][aa[valid]],rel[ii],config))
                bridge[ii]=torch.stack(vals).mean(0)
            else:
                if config.bridge_mode=='reciprocal':
                    k=max(1,math.ceil(len(ii)*config.reciprocal_fraction))
                    threshold=sim.topk(k,dim=0).values[-1]
                    sim=sim.masked_fill(sim<threshold,0)
                bridge[ii]=gated_support(sim,scores[other][1][aa],rel[ii],config)
        final = (1 - config.bridge_weight) * rel + config.bridge_weight * bridge
        if config.smooth_mix:
            final=temporal_smooth(final,g,part['video_shape'],config.smooth_mix)
        if config.diversity_strength:
            for mod in (0,1):
                ii=(g==mod).nonzero().flatten()
                final[ii]=diversity_penalty(final[ii],z[i][ii],config.diversity_strength)
        assert bool(torch.isfinite(final).all())
        keep = g < 0
        for mod in (0, 1):
            ii = (g == mod).nonzero().flatten()
            if not len(ii): continue
            k = max(1, round(len(ii) * ratio))
            if config.module in ('temporal_novelty','value_novelty'):
                keep[ii],swaps=select_with_novelty(final[ii],rel[ii],part['novelty'][ii],k,config.novelty_fraction,config.coverage_fraction)
                diag.setdefault('novelty_exchanges',{})[str(mod)]=swaps
            else:
                keep[ii] = select_with_coverage(final[ii], k, config.coverage_fraction)
            assert int(keep[ii].sum()) == k
        # Retain ORIGINAL embeddings, never context-enriched scorer states.
        part['embeds'] = part['embeds'][:, keep]
        part['positions'] = part['positions'][..., keep]
        part['ids'] = part['original_ids'].to(keep.device)[keep]
        part['final'] = keep.cpu()
        part['stage1'] = keep.cpu()
        part['selection_score'] = final.cpu()
        part['query_relevance'] = rel.cpu()
        part['bridge_score'] = bridge.cpu()
        diag.update({'config': asdict(config), 'bridge_mean': float(bridge[g >= 0].mean()),
                     'bridge_max': float(bridge.max()), 'anchor_indices_other': aa.cpu().tolist()})
        part['score_diagnostics'] = diag
    return parts
