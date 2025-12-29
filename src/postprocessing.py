import pandas as pd

def apply_post_processing(df: pd.DataFrame, min_gap: int = 8) -> pd.DataFrame:
    """Simple post-processing: non-max suppression on predicted events.

    Keeps the first non-'air' event and suppresses subsequent events within `min_gap` frames.
    This is a lightweight fallback to ensure the pipeline runs when no complex postprocessing
    module is provided.
    """
    if 'pred_action' not in df.columns:
        return df

    last_kept = -9999
    keep_mask = []
    for idx, row in df.iterrows():
        action = row.get('pred_action', 'air')
        if action == 'air':
            keep_mask.append(True)
            continue

        if (idx - last_kept) >= min_gap:
            # keep this event
            keep_mask.append(True)
            last_kept = idx
        else:
            # suppress (set to air)
            keep_mask.append(False)

    # apply suppression
    df = df.copy()
    for keep, (idx, row) in zip(keep_mask, df.iterrows()):
        if not keep and df.at[idx, 'pred_action'] != 'air':
            df.at[idx, 'pred_action'] = 'air'

    return df
