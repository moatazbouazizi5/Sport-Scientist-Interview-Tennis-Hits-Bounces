import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, window_size=5) -> pd.DataFrame:
    """
    Génère les features pour le Machine Learning.
    Input: DataFrame avec x, y, vx, vy, ax, ay, acc_mag
    Output: DataFrame enrichi avec Lags, Leads et Rolling stats
    """
    df_feat = df.copy()
    
    # 1. Features de base (déjà calculées dans preprocessing, mais on s'assure qu'elles sont là)
    # On normalise parfois ici, mais pour XGBoost ce n'est pas strictement nécessaire.

    # 2. Lag & Lead (Regarder le passé et le futur)
    # Pour savoir si c'est un pic, il faut comparer avec t-2, t-1, t+1, t+2
    shifts = [-2, -1, 1, 2]
    cols_to_shift = ['y', 'vy', 'acc_mag']
    
    for shift in shifts:
        suffix = f"_prev_{-shift}" if shift < 0 else f"_next_{shift}"
        for col in cols_to_shift:
            df_feat[f'{col}{suffix}'] = df_feat[col].shift(-shift)

    # 3. Rolling Statistics (Moyennes glissantes)
    # Donne une idée de "l'énergie" locale de la balle
    df_feat[f'acc_mean_{window_size}'] = df_feat['acc_mag'].rolling(window_size, center=True).mean()
    df_feat[f'vy_std_{window_size}'] = df_feat['vy'].rolling(window_size, center=True).std()

    # 4. Remplir les NaN créés par shift/rolling
    df_feat = df_feat.fillna(0)

    return df_feat


def prepare_dataset(raw_data_dict, clean_func):
    """
    Transforme le dictionnaire de JSONs en un grand X (features) et y (labels)
    """
    X_list = []
    y_list = []
    
    # Mapping des labels texte -> chiffres
    label_map = {'air': 0, 'bounce': 1, 'hit': 2}
    
    print("Prepauration du Dataset ML...")
    for point_id, data_json in raw_data_dict.items():
        # 1. Preprocessing (Lissage)
        df = clean_func(data_json)
        
        # 2. Feature Engineering
        df_features = create_features(df)
        
        # 3. Separation Features / Target
        drop_cols = ['visible', 'action', 'pred_action', 'frame']
        features_cols = [c for c in df_features.columns if c not in drop_cols]
        
        X_list.append(df_features[features_cols])
        
        # Gestion des labels (Target)
        targets = df['action'].map(label_map).fillna(0)
        y_list.append(targets)

    # Concatenation
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    
    return X, y