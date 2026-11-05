"""Explicit mRoPE coordinates for uniformly sampled, noncontiguous audio.

Audio encoder inputs are packed in chronological order. Their decoder coordinates
refer to the original clip timeline, as do video coordinates and chunk ordering.
"""
import numpy as np
import torch

VISION_START, VISION_END, VIDEO, AUDIO = 151652, 151653, 151656, 151646


def block_layout(stream, reorder=False):
    ids = stream['inputs']['input_ids'][0]
    begin = int((ids == VISION_START).nonzero()[0])
    end = int((ids == VISION_END).nonzero()[0]) + 1
    block = ids[begin:end].clone()
    grid = stream['inputs']['video_grid_thw'][0].tolist()
    t, h, w = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
    record = stream['record']
    seconds = float(torch.as_tensor(stream['inputs']['video_second_per_grid']).flatten()[0])
    if record.get('timeline_version'):
        stamps=torch.tensor(record['frame_seconds'][::2],dtype=torch.float64)-record['start']
        vt=stamps.clamp_min(0).mul(25).long()
    else:
        vt = (torch.arange(t) * torch.tensor(seconds,dtype=torch.float32) * 25).long()
    vp = torch.stack([vt.repeat_interleave(h*w), torch.arange(h).repeat_interleave(w).repeat(t), torch.arange(w).repeat(t*h)])
    na = int((block == AUDIO).sum())
    if 'audio_token_seconds' in record:
        at = torch.tensor(record['audio_token_seconds'], dtype=torch.float64).mul(25).round().long()
        if len(at) != na:
            raise ValueError('Audio timeline/token count mismatch')
    else:
        at = torch.arange(na)
    ap = at.unsqueeze(0).expand(3, -1)
    nv = int((block == VIDEO).sum())
    if nv != vp.shape[1]:
        raise ValueError('Video grid/token count mismatch')
    body = (block == VIDEO) | (block == AUDIO)
    slots = body.nonzero().flatten()
    first, last = int(slots[0]), int(slots[-1]) + 1
    if int(body.sum()) != last-first:
        raise ValueError('Unexpected tokens inside AV content')
    if reorder:
        tokens, coords = [], []
        # Qwen orders video before audio within each two-second source-time chunk.
        chunks = sorted(set((vt//50).tolist()) | set((at//50).tolist()))
        for chunk in chunks:
            for token, pos in ((VIDEO, vp), (AUDIO, ap)):
                selected = pos[:, pos[0]//50 == chunk]
                if selected.shape[1]:
                    tokens.append(torch.full((selected.shape[1],), token, dtype=block.dtype))
                    coords.append(selected)
        block[first:last] = torch.cat(tokens)
        content_pos = torch.cat(coords, dim=1)
    else:
        content_pos = torch.empty((3, last-first), dtype=torch.long)
        content_pos[:, block[first:last] == VIDEO] = vp
        content_pos[:, block[first:last] == AUDIO] = ap
    positions = torch.zeros((3, len(block)), dtype=torch.long)
    positions[:, first:last] = content_pos + 1
    positions[:, last:] = positions[:, :last].max() + 1
    return block, positions


def prompt_positions(ids, layouts):
    """Place each independent clip after preceding text, without joining timelines."""
    row = ids[0]
    starts = (row == VISION_START).nonzero().flatten().tolist()
    ends = ((row == VISION_END).nonzero().flatten()+1).tolist()
    if len(starts) != len(layouts) or len(ends) != len(layouts):
        raise ValueError('Prompt/clip layout mismatch')
    result = torch.empty((3, 1, len(row)), dtype=torch.long)
    cursor, offset = 0, 0
    for begin, end, (block, positions) in zip(starts, ends, layouts):
        n = begin-cursor
        result[:, 0, cursor:begin] = torch.arange(n)+offset
        offset += n
        if not torch.equal(row[begin:end], block):
            raise ValueError('AV block changed while rebuilding prompt')
        result[:, 0, begin:end] = positions+offset
        offset = int(result[:, 0, :end].max())+1
        cursor = end
    result[:, 0, cursor:] = torch.arange(len(row)-cursor)+offset
    return result
