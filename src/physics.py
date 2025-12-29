import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_physics_events(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Détecte les Hits et Bounces en utilisant les règles physiques.
    Ajoute une colonne 'pred_action' au DataFrame.
    - Mappe les positions retournées par `find_peaks` aux index (numéros de frame)
    - Utilise `thresholds.get(...)` avec valeurs par défaut pour robustesse
    """
    # Defaults (au cas où la config manque des clefs)
    bounce_thresh = thresholds.get('bounce_accel', -0.5)
    hit_thresh = thresholds.get('hit_accel', 1.5)
    hit_min_height_ratio = thresholds.get('hit_min_height_ratio', 0.95)

    # 1. Initialisation : Tout est "air" par défaut
    df['pred_action'] = 'air'
    
    # --- A. DÉTECTION DES REBONDS (BOUNCES) ---
    # Un rebond est un maximum local de Y (le bas de l'image a un grand Y)
    # prominence=10 : Le pic doit ressortir d'au moins 10 pixels par rapport au voisinage
    peaks_y, _ = find_peaks(df['y_smooth'], prominence=10, width=[1, 15])
    
    bounce_indices = []
    
    for p in peaks_y:
        # Vérification physique : Accélération verticale brutale vers le haut (négative)
        # On utilise le seuil défini dans config.yaml
        if df.iloc[p]['ay'] < thresholds['bounce_accel']:
            df.iloc[p, df.columns.get_loc('pred_action')] = 'bounce'
            bounce_indices.append(p)

    # --- B. DÉTECTION DES FRAPPES (HITS) ---
    # Une frappe est un pic d'accélération totale (choc raquette)
    # height=2.0 : L'accélération doit être forte
    peaks_acc, _ = find_peaks(df['acc_mag'], height=thresholds['hit_accel'], distance=5)
    
    for p in peaks_acc:
        # Règle d'exclusion : Une frappe ne peut pas être un rebond
        if p in bounce_indices:
            continue
            
        # Règle géométrique : Une frappe ne se fait pas au ras du sol
        # On vérifie que la balle n'est pas à son point le plus bas (max Y)
        y_max = df['y_smooth'].max()
        current_y = df.iloc[p]['y_smooth']
        
        # Si la hauteur est < 95% du max (donc un peu en l'air), c'est un Hit
        if current_y < (y_max * thresholds['hit_min_height_ratio']):
            df.iloc[p, df.columns.get_loc('pred_action')] = 'hit'
            
    return df