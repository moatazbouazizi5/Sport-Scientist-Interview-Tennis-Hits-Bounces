import json
from collections import Counter

path = 'data/predictions/output.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

true_labels = []
pred_labels = []
label_map = {'air': 0, 'bounce': 1, 'hit': 2}
inv_map = {v: k for k, v in label_map.items()}

for frames in data.values():
    for fd in frames.values():
        true = fd.get('action', 'air')
        pred = fd.get('pred_action', 'air')
        true_labels.append(label_map.get(true, 0))
        pred_labels.append(label_map.get(pred, 0))

print('Total frames:', len(true_labels))

try:
    from sklearn.metrics import classification_report, confusion_matrix
    print('\n--- Classification Report ---')
    print(classification_report(true_labels, pred_labels, target_names=['air','bounce','hit']))
    print('\n--- Confusion Matrix (rows=true, cols=pred) ---')
    print(confusion_matrix(true_labels, pred_labels))
except Exception:
    # Fallback simple counts and per-class precision/recall/f1
    print('\nscikit-learn non disponible — fallback metrics')
    counts = Counter()
    tp = Counter()
    pred_count = Counter()
    true_count = Counter()
    for t, p in zip(true_labels, pred_labels):
        counts[(t, p)] += 1
        if t == p:
            tp[t] += 1
        pred_count[p] += 1
        true_count[t] += 1

    for cls in [0,1,2]:
        prec = tp[cls] / pred_count[cls] if pred_count[cls] else 0.0
        rec = tp[cls] / true_count[cls] if true_count[cls] else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        print(f"Class {cls} ({inv_map[cls]}): precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
