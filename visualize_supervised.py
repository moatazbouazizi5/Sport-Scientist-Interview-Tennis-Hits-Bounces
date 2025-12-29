#!/usr/bin/env python
"""visualize_supervised.py

Visualise les prédictions supervisées d'un point.

Options:
 - charger le modèle sauvegardé et prédire un point
 - ou charger le fichier de prédictions sauvegardées

Affiche la trajectoire (`y_smooth`) et un sous-graphique avec labels vrais vs prédits.
"""
import argparse
import json
import yaml
import matplotlib.pyplot as plt

from src.utils import load_raw_data
from src.preprocessing import load_point_dataframe, clean_trajectory
from src.features import create_features
from src.models import TennisClassifier


INV_LABEL = {0: 'air', 1: 'bounce', 2: 'hit'}


def plot_labels(df, true_labels, pred_labels, title=None, out_path=None):
    frames = df.index.to_list()
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]})

    # Trajectory
    axs[0].plot(frames, df['y_smooth'], label='y_smooth')
    axs[0].set_ylabel('y (px)')
    axs[0].legend()

    # Labels plot (as step)
    # Map labels to ints for plotting
    label_to_int = {'air': 0, 'bounce': 1, 'hit': 2}
    true_ints = [label_to_int.get(l, 0) for l in true_labels]
    pred_ints = [label_to_int.get(l, 0) for l in pred_labels]

    axs[1].step(frames, true_ints, where='mid', label='true', linewidth=2)
    axs[1].step(frames, pred_ints, where='mid', label='pred', linewidth=1, linestyle='--')
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(['air', 'bounce', 'hit'])
    axs[1].set_xlabel('Frame')
    axs[1].legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(out_path)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize supervised predictions for a point')
    parser.add_argument('--point', '-p', help='Point ID to visualise', default=None)
    parser.add_argument('--file', '-f', help='Path to a single JSON file for a point', default=None)
    parser.add_argument('--model', '-m', action='store_true', help='Load model and predict (preferred)')
    parser.add_argument('--preds', help='Path to predictions JSON (overrides model)', default=None)
    parser.add_argument('--out', '-o', help='Output image path to save', default=None)
    args = parser.parse_args()

    with open('config/config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    # Load raw data or single file
    if args.file:
        from pathlib import Path
        p = Path(args.file)
        with open(p, 'r') as f:
            raw = json.load(f)
        pid = p.stem.split('_')[-1]
        raw_data = {pid: raw}
    else:
        raw_data = load_raw_data(config['paths']['raw_data'])

    if not raw_data:
        print('No data found.')
        return

    point_id = args.point or next(iter(raw_data.keys()))
    if point_id not in raw_data:
        print(f'Point {point_id} not found')
        return

    point_json = raw_data[point_id]
    df = load_point_dataframe(point_json)
    df = clean_trajectory(df,
                          window_length=config['smoothing']['window_length'],
                          poly_order=config['smoothing']['poly_order'])

    # Determine source of predictions
    pred_labels = None

    if args.preds:
        # Load provided preds file
        with open(args.preds, 'r') as f:
            preds_json = json.load(f)
        point_preds = preds_json.get(point_id, {})
        # Extract per-frame pred_action in same frame order
        frames = df.index.to_list()
        pred_labels = [point_preds.get(str(fr), {}).get('pred_action', 'air') for fr in frames]

    elif args.model or (not args.preds and config['paths'].get('model_save')):
        # Try loading model and predict
        model_path = config['paths'].get('model_save')
        clf = TennisClassifier(config.get('model_params', {}))
        try:
            clf.load(model_path)
            # Build feature matrix identical to training
            df_feat = create_features(df)
            drop_cols = ['visible', 'action', 'pred_action', 'frame']
            feature_cols = [c for c in df_feat.columns if c not in drop_cols]
            preds = clf.predict(df_feat[feature_cols])
            pred_labels = [INV_LABEL.get(int(p), 'air') for p in preds]
        except Exception as e:
            print(f'Failed to load/predict with model: {e}')

    # If still no preds, try supervised_output file
    if pred_labels is None:
        sup_path = config['paths'].get('supervised_output')
        try:
            with open(sup_path, 'r') as f:
                sup = json.load(f)
            point_preds = sup.get(point_id, {})
            frames = df.index.to_list()
            pred_labels = [point_preds.get(str(fr), {}).get('pred_action', 'air') for fr in frames]
        except Exception:
            # Fallback: all air
            pred_labels = ['air'] * len(df)

    # True labels from dataset (if available)
    true_labels = df['action'].fillna('air').tolist()

    title = f'Point {point_id} — Supervised predictions'
    plot_labels(df, true_labels, pred_labels, title=title, out_path=args.out)


if __name__ == '__main__':
    main()
