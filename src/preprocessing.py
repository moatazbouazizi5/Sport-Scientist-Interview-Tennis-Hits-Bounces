import numpy as np
import pandas as pd
try:
    from scipy.signal import savgol_filter
except Exception:
    # Fallback simple smoothing if scipy is not available or fails to import.
    def savgol_filter(x, window_length=5, polyorder=2):
        # simple moving average fallback (window_length must be >=1)
        if window_length <= 1:
            return np.array(x)
        w = np.ones(window_length) / window_length
        # pad edges to preserve length
        xp = np.pad(np.asarray(x), (window_length//2, window_length//2), mode='edge')
        y = np.convolve(xp, w, mode='valid')
        return y

def load_point_dataframe(point_data_json: dict) -> pd.DataFrame:
    """
    Convertit le JSON d'un point en DataFrame Pandas trié par frame.
    """
    # 1. Convertir les clés (numéros de frame) en entiers pour trier correctement
    # Sinon "10" arrive avant "2" (tri alphabétique)
    frames = sorted([int(k) for k in point_data_json.keys()])
    
    data_list = []
    for f in frames:
        entry = point_data_json[str(f)]
        data_list.append({
            'frame': f,
            'x': entry.get('x', np.nan), # Met NaN si pas de valeur
            'y': entry.get('y', np.nan),
            'visible': entry.get('visible', False),
            'action': entry.get('action', 'air') # On garde le label pour le test supervisé plus tard
        })
    
    df = pd.DataFrame(data_list).set_index('frame')
    return df

def clean_trajectory(df: pd.DataFrame, window_length=7, poly_order=2) -> pd.DataFrame:
    """
    Pipeline complet : Interpolation -> Lissage -> Calcul Vitesse/Accélération
    """
    # 1. Interpolation (Remplir les trous)
    # Si la balle disparaît 2 frames, on trace une ligne droite entre avant et après
    df['x'] = df['x'].interpolate(method='linear').bfill().ffill()
    df['y'] = df['y'].interpolate(method='linear').bfill().ffill()

    # 2. Lissage (Savitzky-Golay)
    # C'est magique pour la physique : ça enlève le bruit sans écraser les pics (rebonds)
    # window_length doit être impair (ex: 5, 7, 9)
    if window_length % 2 == 0:
        window_length += 1
        
    df['x_smooth'] = savgol_filter(df['x'], window_length=window_length, polyorder=poly_order)
    df['y_smooth'] = savgol_filter(df['y'], window_length=window_length, polyorder=poly_order)

    # 3. Calcul des Dérivées (Cinématique)
    # Vitesse (Dérivée 1ère)
    df['vx'] = np.gradient(df['x_smooth'])
    df['vy'] = np.gradient(df['y_smooth'])
    
    # Accélération (Dérivée 2ème)
    df['ax'] = np.gradient(df['vx'])
    df['ay'] = np.gradient(df['vy'])
    
    # Magnitude de l'accélération totale (Racine carrée de ax² + ay²)
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)

    return df