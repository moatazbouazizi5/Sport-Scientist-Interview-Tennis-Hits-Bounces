import json
from collections import Counter
p='data/predictions/output.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
num_points=len(data)
total_frames=sum(len(frames) for frames in data.values())
labels=Counter()
missing=0
for frames in data.values():
    for fd in frames.values():
        pa=fd.get('pred_action', None)
        if pa is None:
            missing+=1
        else:
            labels[pa]+=1
print('points:',num_points)
print('frames:',total_frames)
print('labels:',dict(labels))
print('missing_pred_action:',missing)
