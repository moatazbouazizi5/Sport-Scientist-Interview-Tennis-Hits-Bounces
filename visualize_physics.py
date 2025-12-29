#!/usr/bin/env python
"""visualize_physics.py

Script CLI pour visualiser la trajectoire, vitesses et accélérations
d'un point (ball_data). Utilise les fonctions de preprocessing et physics
du projet.
"""
import argparse
import yaml
import matplotlib.pyplot as plt

from src.utils import load_raw_data
from src.preprocessing import load_point_dataframe, clean_trajectory
from src.features import create_features
from src.physics import detect_physics_events


def plot_point(df, events_df=None, title=None, out_path=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    frames = df.index.to_list()

    # Trajectory
    axs[0].plot(frames, df['x_smooth'], label='x_smooth')
    axs[0].plot(frames, df['y_smooth'], label='y_smooth')
    axs[0].set_ylabel('Position')
    axs[0].legend()

    # Velocity
    axs[1].plot(frames, df['vx'], label='vx')
    axs[1].plot(frames, df['vy'], label='vy')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()

    # Acceleration
    axs[2].plot(frames, df['acc_mag'], label='acc_mag')
    axs[2].plot(frames, df['ax'], label='ax', alpha=0.6)
    axs[2].plot(frames, df['ay'], label='ay', alpha=0.6)
    axs[2].set_ylabel('Acceleration')
    axs[2].set_xlabel('Frame')
    axs[2].legend()

    if events_df is not None:
        # Mark events (bounce / hit)
        for i, row in events_df.iterrows():
            if row.get('pred_action') == 'bounce':
                for ax in axs:
                    ax.axvline(x=i, color='C3', linestyle='--', alpha=0.7)
            elif row.get('pred_action') == 'hit':
                for ax in axs:
                    ax.axvline(x=i, color='C4', linestyle='-.', alpha=0.7)

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        fig.savefig(out_path)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize physics for a single point')
    parser.add_argument('--point', '-p', help='Point ID (e.g. 12). If omitted, the first point is used.', default=None)
    parser.add_argument('--file', '-f', help='Path to a single JSON file for a point', default=None)
    parser.add_argument('--preds', help='Path to predictions JSON (overrides recompute)', default=None)
    parser.add_argument('--out', '-o', help='Output image path to save', default=None)
    args = parser.parse_args()

    # Load config for data path and thresholds
    with open('config/config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    thresholds = config.get('physics_thresholds', {})

    # Load data
    raw_data = {}
    if args.file:
        # If a single file is given, load it as a dict via load_raw_data expects folder,
        # so fallback to reading manually
        import json
        from pathlib import Path
        p = Path(args.file)
        with open(p, 'r') as f:
            raw = json.load(f)
        pid = p.stem.split('_')[-1]
        raw_data[pid] = raw
    else:
        raw_data = load_raw_data(config['paths']['raw_data'])

    # Load saved predictions if requested
    preds_json = None
    if args.preds:
        import json
        try:
            with open(args.preds, 'r') as fh:
                preds_json = json.load(fh)
        except Exception as e:
            print(f"Failed to load preds file {args.preds}: {e}")
            preds_json = None
    else:
        # fallback to config-defined physics_output if exists
        phys_path = config.get('paths', {}).get('physics_output')
        if phys_path and os.path.exists(phys_path):
            import json
            try:
                with open(phys_path, 'r') as fh:
                    preds_json = json.load(fh)
            except Exception:
                preds_json = None

    if not raw_data:
        print('No data found.')
        return

    # Select point
    point_id = args.point or next(iter(raw_data.keys()))
    if point_id not in raw_data:
        print(f'Point {point_id} not found in data folder.')
        return

    point_json = raw_data[point_id]

    # Preprocess
    df = load_point_dataframe(point_json)
    df = clean_trajectory(df,
                          window_length=config['smoothing']['window_length'],
                          poly_order=config['smoothing']['poly_order'])

    # If saved preds are available, build events_df from them, else recompute
    events_df = None
    if preds_json and point_id in preds_json:
        point_preds = preds_json.get(point_id, {})
        frames = df.index.to_list()
        # create DataFrame aligned on frames
        import pandas as _pd
        pred_list = [point_preds.get(str(fr), {}).get('pred_action', 'air') for fr in frames]
        events_df = _pd.DataFrame({'pred_action': pred_list}, index=frames)
    else:
        events_df = detect_physics_events(df.copy(), thresholds)[['pred_action']]

    # Plot
    title = f'Point {point_id} — Physics visualization'
    plot_point(df, events_df=events_df, title=title, out_path=args.out)


if __name__ == '__main__':
    main()
