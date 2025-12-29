import yaml
import pandas as pd
import os
import sys
import argparse

# Ensure project root is on sys.path so `src` package is importable when running as script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from src.utils import load_raw_data, save_predictions
from src.preprocessing import load_point_dataframe, clean_trajectory
from src.features import prepare_dataset, create_features
from src.models import TennisClassifier
from src.physics import detect_physics_events
from src.postprocessing import apply_post_processing

# -------------------------------------------
# HELPERS DE CONFIG ET PREPROCESSING
# -------------------------------------------
def run_preprocessing(data_json, config):
    """Applique le nettoyage et le calcul des features physiques."""
    df = load_point_dataframe(data_json)
    df_clean = clean_trajectory(df, 
                                window_length=config['smoothing']['window_length'],
                                poly_order=config['smoothing']['poly_order'])
    return df_clean

# -------------------------------------------
# FONCTION PRINCIPALE POUR LE PIPELINE
# -------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run pipeline methods')
    parser.add_argument('--method', choices=['physics', 'ml', 'both'], default='both',
                        help='Which method to run: physics, ml, or both')
    args = parser.parse_args()
    method = args.method

    # 1. Chargement de la configuration
    print("Chargement de la configuration...")
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 2. Chargement des données brutes
    raw_data = load_raw_data(config['paths']['raw_data'])
    if not raw_data: 
        print("ERREUR : Aucun fichier de données trouvé. Vérifiez 'config/config.yaml'.")
        return

    print(f"Chargement terminé : {len(raw_data)} points trouvés.")

    # ==========================================
    # MÉTHODE 1 : PHYSIQUE (NON-SUPERVISÉ)
    # ==========================================
    if method in ("physics", "both"):
        print("\n--- Exécution Méthode 1 : Physique ---")
        results_physics = {}
        # Parcours de tous les points pour appliquer la logique physique
        for point_id, data_json in raw_data.items():
            df = run_preprocessing(data_json, config)
            df_pred = detect_physics_events(df, config['physics_thresholds'])

            # Formatage pour la sauvegarde JSON
            point_res = {}
            for idx, row in df_pred.iterrows():
                orig = data_json.get(str(idx), {})
                orig['pred_action'] = row['pred_action'] # Ajout de la prédiction
                point_res[str(idx)] = orig
            results_physics[point_id] = point_res

        # Sauvegarde des résultats de la méthode 1
        phys_out = config.get('paths', {}).get('physics_output', 'data/predictions/physics.json')
        # ensure parent dir exists (save_predictions also does this, but be defensive)
        os.makedirs(os.path.dirname(phys_out), exist_ok=True)
        save_predictions(results_physics, phys_out) # On utilise un chemin dédié


    # ==========================================
    # MÉTHODE 2 : MACHINE LEARNING (SUPERVISÉ)
    # ==========================================
    if method in ("ml", "both"):
        print("\n--- Exécution Méthode 2 : Machine Learning ---")
    else:
        print("\n--- Skipping Machine Learning method (not requested) ---")
        return
    # Séparation Train/Test par ID de point (crucial pour la validation)
    point_ids = list(raw_data.keys())
    train_ids, test_ids = train_test_split(point_ids, test_size=0.2, random_state=42)
    
    print(f"Split: {len(train_ids)} points pour l'Entraînement, {len(test_ids)} points pour le Test.")

    # Préparation des datasets avec features enrichies
    train_data_dict = {k: raw_data[k] for k in train_ids}
    test_data_dict = {k: raw_data[k] for k in test_ids}
    
    # Utilisation de la fonction de nettoyage définie plus haut
    clean_func = lambda d: run_preprocessing(d, config)

    X_train, y_train = prepare_dataset(train_data_dict, clean_func)
    X_test, y_test = prepare_dataset(test_data_dict, clean_func)

    if X_train is None or X_test is None:
        print("Erreur lors de la préparation des datasets. Arrêt.")
        return

    # --- ENTRAINEMENT AVEC WEIGHTS OPTIMISÉS ---
    print(f"\nClasses avant gestion du déséquilibre (Train) : {y_train.value_counts().to_dict()}")
    # Utilisation de compute_sample_weight pour gérer le déséquilibre
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    clf = TennisClassifier(config['model_params'])
    # Entraînement avec les poids calculés
    clf.train(X_train, y_train, sample_weight=weights)

    # Évaluation finale (utilise l'API existante)
    print("\n--- Évaluation du Modèle ML sur Test Set ---")
    clf.evaluate(X_test, y_test)

    # Sauvegarde du modèle
    model_save_path = config['paths']['model_save']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    clf.save(model_save_path)

    # --- GÉNÉRATION DU FICHIER FINAL AVEC PRÉDICTIONS ML + POST-PROCESSING ---
    results_ml = {}
    inv_map = {0: 'air', 1: 'bounce', 2: 'hit'}
    
    print("\nGénération des prédictions finales (ML + Post-processing)...")
    for point_id, data_json in raw_data.items():
        # 1. Nettoyage + Features
        df = run_preprocessing(data_json, config)
        df_feat = create_features(df)
        
        # 2. Prédiction (utilise l'API existante `predict` et `predict_proba` si disponible)
        feature_cols = X_train.columns
        preds_encoded = clf.predict(df_feat[feature_cols])
        probs = None
        try:
            probs = clf.model.predict_proba(df_feat[feature_cols])
        except Exception:
            probs = None

        # 3. Ajout des probabilités pour le post-processing (fallbacks si absent)
        df['pred_action'] = [inv_map[int(p)] for p in preds_encoded]
        if probs is not None:
            df['prob_bounce'] = probs[:, 1]  # Proba de la classe 1 (bounce)
            df['prob_hit'] = probs[:, 2]     # Proba de la classe 2 (hit)
        else:
            df['prob_bounce'] = 0.0
            df['prob_hit'] = 0.0
        
        # 4. Post-processing (NMS + Règles physiques)
        gap_param = config.get('post_processing', {}).get('min_gap', 8)
        
        df_final = apply_post_processing(df, min_gap=gap_param)

        
        # 5. Formatage JSON pour la sauvegarde
        point_res = {}
        for idx, row in df_final.iterrows():
            orig = data_json.get(str(idx), {})
            orig['pred_action'] = row['pred_action'] # Remplacer l'ancienne prédiction
            point_res[str(idx)] = orig
        results_ml[point_id] = point_res

    # Sauvegarde des résultats de la méthode 2
    ml_out = config['paths'].get('supervised_output', 'data/predictions/supervised.json')
    os.makedirs(os.path.dirname(ml_out), exist_ok=True)
    save_predictions(results_ml, ml_out) # Chemin dédié

    print(f"\nTerminé ! Résultats sauvegardés dans physics.json et supervised.json")

if __name__ == "__main__":
    main()