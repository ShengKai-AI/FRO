"""Label-blind pairing with exhaustive, unique scored coverage."""
import collections,hashlib,heapq

def make_pairs(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate source IDs')
    dataset=rows[0]['dataset'];videos=collections.defaultdict(list)
    if any(r['dataset']!=dataset for r in rows):raise ValueError('Do not pair across datasets')
    for r in sorted(rows,key=lambda r:r['id']):videos[r['video_id']].append(r)
    def order(v):return hashlib.sha256(('20261031:'+v).encode()).hexdigest()
    heap=[(-len(q),order(v),v) for v,q in videos.items()];heapq.heapify(heap);pairs=[]
    while len(heap)>1:
        a,b=heapq.heappop(heap),heapq.heappop(heap);pairs.append(([videos[a[2]].pop(0),videos[b[2]].pop(0)],[True,True]))
        for n,h,v in (a,b):
            if n+1<0:heapq.heappush(heap,(n+1,h,v))
    if heap:
        _,_,v=heap[0];companion=next((r for r in rows if r['video_id']!=v),None)
        if companion is None:raise ValueError('Two distinct videos required')
        pairs.extend(([q,companion],[True,False]) for q in videos[v])
    bundles=collections.defaultdict(list);covered=[]
    for questions,mask in pairs:
        ident=dataset+'-'+hashlib.sha256('|'.join(q['id'] for q in questions).encode()).hexdigest()[:20]
        pair={'id':ident,'dataset':dataset,'question':'\n'.join('Q'+str(i+1)+': '+q['question'] for i,q in enumerate(questions)),
              'questions':[{'id':q['id'],'question':q['question'],'options':[f'{chr(65+i)}. {o}' for i,o in enumerate(q['options'])]} for q in questions],
              'streams':[{k:q[k] for k in ('video_id','start','end') if k in q} for q in questions],'score_mask':mask}
        bundles[tuple(q['video_id'] for q in questions)].append(pair)
        covered.extend(q['id'] for q,score in zip(questions,mask) if score)
    if collections.Counter(covered)!=collections.Counter(r['id'] for r in rows):raise ValueError('Pair coverage mismatch')
    return [{'bundle_id':hashlib.sha256('|'.join(v).encode()).hexdigest()[:16],'video_ids':v,'pairs':qs} for v,qs in sorted(bundles.items())]
