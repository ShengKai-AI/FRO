# Extracted from pinned FastVID official Qwen2.5-VL implementation. See manifests/methods.json.
import torch
from types import SimpleNamespace

@torch.inference_mode()
def compress(features, grid, attention, ratio):
    self=SimpleNamespace(fastvid_DySeg_c=8,fastvid_DySeg_tau=.84,fastvid_DySeg_ignore=1.0,fastvid_STPrune_d=.4,fastvid_DTM_p=4,fastvid_DTM_beta=.6,fastvid_retention_ratio=ratio)
    inputs_embeds=features[None]; seq_length=features.shape[0]; prompt_start_len=0; video_len=seq_length
    video_grid_thw=grid[None]; frame_attn_weights=attention.reshape(int(grid[0]),-1)
    frame_global_features=features.reshape(int(grid[0]),-1,features.shape[-1]).mean(1)
    frame_num, frame_w, frame_h = video_grid_thw[0,0].item(), video_grid_thw[0,1].item() // 2, video_grid_thw[0,2].item() // 2
    device_type = inputs_embeds.device
    start_tokens = inputs_embeds[:,:prompt_start_len,:]
    video_hidden_states = inputs_embeds[:,prompt_start_len:prompt_start_len + video_len,:].reshape(frame_num, frame_w * frame_h, -1)
    end_tokens = inputs_embeds[:,prompt_start_len + video_len:,:]
    keep_indexs = ()
    keep_indexs += (torch.arange(prompt_start_len, device=device_type),)
    keep_indexs += (torch.arange(prompt_start_len + video_len,seq_length, device=device_type),)

    frame_global_features = frame_global_features / frame_global_features.norm(dim=1, keepdim=True) 
    similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)

    cut_indices = torch.topk(similarity_matrix, min(self.fastvid_DySeg_c-1, max(frame_num-1, 0)), largest=False).indices
    cos_indices = torch.nonzero(similarity_matrix < self.fastvid_DySeg_tau).squeeze(1)
    cut_indices = torch.unique(torch.cat([cut_indices, cos_indices]))
    cut_indices = cut_indices.sort().values

    segment_sizes = [cut_indices[0].item() + 1] if cut_indices.numel() else [frame_num]  
    for i in range(1, cut_indices.shape[0]):
        segment_sizes.append(cut_indices[i].item() - cut_indices[i-1].item())  

    if cut_indices.numel(): segment_sizes.append(frame_num - cut_indices[-1].item() - 1)

    segments_hidden_states = torch.split(video_hidden_states, segment_sizes)
    segments_attn_weights = torch.split(frame_attn_weights, segment_sizes)
    segment_num = len(segment_sizes)

    segments_global_features = torch.split(frame_global_features, segment_sizes)

    vision_tokens = []

    frame_retain_num = int(frame_w * frame_h * self.fastvid_retention_ratio)

    seg_index_start = prompt_start_len
    for seg_i in range(segment_num):
        cur_hidden_states = segments_hidden_states[seg_i]
        cur_attn_weights = segments_attn_weights[seg_i]
        cur_global_features = segments_global_features[seg_i]
        cur_seg_len, frm_len = cur_attn_weights.shape

        seg_retain_num = frame_retain_num * cur_seg_len
        seg_context_num = max(int(seg_retain_num * self.fastvid_STPrune_d), 1)
        seg_salient_num = seg_retain_num - seg_context_num

        if cur_seg_len == 1:
            frm_salient_num = [seg_salient_num]
            frm_context_num = [seg_context_num]
        else:
            cur_global_similarity_matrix = (cur_global_features[:-1] * cur_global_features[1:]).sum(dim=1)
            cur_cut_indices = torch.nonzero(cur_global_similarity_matrix < self.fastvid_DySeg_ignore).squeeze(1)
            cur_cut_indices = cur_cut_indices.sort().values
            if cur_cut_indices.shape[0] > 0:
                cur_segment_sizes = [cur_cut_indices[0].item() + 1]  
                for i in range(1, cur_cut_indices.shape[0]):
                    cur_segment_sizes.append(cur_cut_indices[i].item() - cur_cut_indices[i-1].item())  
                cur_segment_sizes.append(cur_seg_len - cur_cut_indices[-1].item() - 1)
            else:
                cur_segment_sizes = [cur_seg_len]

            valid_seg_len = len(cur_segment_sizes)

            chunk_size = seg_salient_num // valid_seg_len
            remainder = seg_salient_num % valid_seg_len
            frm_salient_num = [chunk_size + (1 if i < remainder else 0) for i in range(valid_seg_len)]

            temp_num = (valid_seg_len + self.fastvid_DTM_p - 1) //  self.fastvid_DTM_p

            chunk_size = seg_context_num // temp_num
            remainder = seg_context_num % temp_num
            temp_context_num = [chunk_size + (1 if i < remainder else 0) for i in range(temp_num)]

            tmp_ctx_idx = 0
            frm_context_num = []
            for tmp_i in range(valid_seg_len):
                if tmp_i % self.fastvid_DTM_p == 0:
                    frm_context_num.append(min(temp_context_num[tmp_ctx_idx], frm_len//2))
                    if temp_context_num[tmp_ctx_idx] + frm_context_num[-1] > frm_len:
                        temp_context_num[tmp_ctx_idx] = frm_len - frm_context_num[-1]
                    tmp_ctx_idx += 1
                else:
                    frm_context_num.append(0)
            frm_context_num.reverse()

            valid_frm_salient_num = []
            valid_frm_context_num = []
            for tmp_tmp_tmp_i in range(valid_seg_len):
                valid_frm_salient_num.extend([0] * (cur_segment_sizes[tmp_tmp_tmp_i] - 1))
                valid_frm_context_num.extend([0] * (cur_segment_sizes[tmp_tmp_tmp_i] - 1))

                if frm_context_num[tmp_tmp_tmp_i] == 0:
                    valid_frm_salient_num.append(min(frm_salient_num[tmp_tmp_tmp_i], frm_len))
                    valid_frm_context_num.append(0)
                elif frm_context_num[tmp_tmp_tmp_i] > 0:
                    valid_frm_context_num.append(min(frm_context_num[tmp_tmp_tmp_i], frm_len//2))
                    valid_frm_salient_num.append(min(frm_salient_num[tmp_tmp_tmp_i], frm_len - valid_frm_context_num[-1]))
                else:
                    valid_frm_salient_num.append(frm_len-int(frm_len * self.fastvid_STPrune_d))
                    valid_frm_context_num.append(int(frm_len * self.fastvid_STPrune_d))

            frm_salient_num = valid_frm_salient_num
            frm_context_num = valid_frm_context_num

        cur_salient_indexs = []
        cur_context_indexs = []

        for frame_i in range(cur_seg_len):
            ### Get Salient Indexs
            if frm_salient_num[frame_i] > 0:
                frm_top_indexes = torch.topk(cur_attn_weights[frame_i], frm_salient_num[frame_i]).indices
            else:
                frm_top_indexes = None

            ### Get Context Indexs
            if frm_context_num[frame_i] > 0:
                frm_all_indexs = torch.arange(frm_len, device=device_type)
                if frm_top_indexes is not None:
                    tmp_mask = torch.isin(frm_all_indexs, frm_top_indexes)
                    remaining_indexs = frm_all_indexs[~tmp_mask]  
                else:
                    remaining_indexs = frm_all_indexs

                tmp_frm_hidden_states = cur_hidden_states[frame_i]
                tmp_frm_hidden_states = tmp_frm_hidden_states[remaining_indexs].unsqueeze(0)
                _, tmp_N, tmp_D = tmp_frm_hidden_states.shape

                dist_matrix = torch.cdist(tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()) / (tmp_D ** 0.5)

                # get local density
                dist_nearest, index_nearest = torch.topk(dist_matrix, k=4, dim=-1, largest=False)
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                density = density + torch.rand(
                    density.shape, device=device_type, dtype=density.dtype) * 1e-6

                # get distance indicator
                mask = density[:, None, :] > density[:, :, None]
                mask = mask.type(tmp_frm_hidden_states.dtype)
                dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
                dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

                # select clustering center according to score
                score = dist * density
                _, sampled_indexs = torch.topk(score, k=frm_context_num[frame_i], dim=-1)


                sampled_indexs = sampled_indexs[0].sort().values

                frm_context_indexes = remaining_indexs[sampled_indexs]
            else:
                frm_context_indexes = None

            ### Index Process
            if frm_top_indexes is not None:
                cur_salient_indexs.append(frm_top_indexes + frame_i * frm_len)
            if frm_context_indexes is not None:
                cur_context_indexs.append(frm_context_indexes + frame_i * frm_len)

        cur_tome_hidden_states = cur_hidden_states.reshape(cur_seg_len * frm_len, -1)

        cur_salient_indexs = torch.cat(cur_salient_indexs)
        cur_salient_hidden_states = cur_tome_hidden_states[cur_salient_indexs]

        ### Context Merge
        cur_all_indexs = torch.arange(cur_seg_len * frm_len, device=device_type)
        cur_tome_tokens = cur_tome_hidden_states / cur_tome_hidden_states.norm(dim=-1, keepdim=True) 
        cur_context_hidden_states = []

        for tmp_context_indexs in cur_context_indexs:
            tmp_retain_index = torch.cat([cur_salient_indexs, tmp_context_indexs])
            tmp_mask = torch.isin(cur_all_indexs, tmp_retain_index)
            tmp_merge_indexs = cur_all_indexs[~tmp_mask] 

            tmp_target_tokens = cur_tome_tokens[tmp_context_indexs]
            tmp_to_merge_tokens = cur_tome_tokens[tmp_merge_indexs]

            similarity = torch.mm(tmp_to_merge_tokens, tmp_target_tokens.T)
            assign_one_hot = torch.zeros(tmp_to_merge_tokens.shape[0], tmp_context_indexs.shape[0], dtype=cur_hidden_states.dtype, device=device_type)
            assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)

            avg_weights = (1 / (assign_one_hot.sum(dim=0).unsqueeze(-1) + 1)).clamp(min=self.fastvid_DTM_beta)
            counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = cur_tome_hidden_states[tmp_merge_indexs]
            aggregated_hidden = torch.mm(assign_one_hot.T, hidden_to_merge) / counts
            target_hidden = cur_tome_hidden_states[tmp_context_indexs] 
            target_hidden = avg_weights * target_hidden + (1 - avg_weights) * aggregated_hidden

            cur_context_hidden_states.append(target_hidden)

        cur_context_indexs = torch.cat(cur_context_indexs)
        cur_context_hidden_states = torch.cat(cur_context_hidden_states, dim=0)

        ### segment final hidden_states & indexs
        cur_index_combined = torch.cat([cur_salient_indexs, cur_context_indexs])
        cur_hidden_combined = torch.cat([cur_salient_hidden_states, cur_context_hidden_states], dim=0)

        sorted_indexs = torch.argsort(cur_index_combined)
        cur_index_combined = cur_index_combined[sorted_indexs]
        cur_hidden_combined = cur_hidden_combined[sorted_indexs]

        vision_tokens.append(cur_hidden_combined)
        keep_indexs += (cur_index_combined + seg_index_start,)
        seg_index_start = seg_index_start + cur_seg_len * frm_len

    vision_tokens = torch.cat(vision_tokens, dim=0).unsqueeze(0)

    inputs_embeds = torch.cat([start_tokens, vision_tokens, end_tokens], dim=1)
    keep_indexs = torch.cat(keep_indexs)
    keep_indexs = keep_indexs.sort().values

    return inputs_embeds[0], keep_indexs
