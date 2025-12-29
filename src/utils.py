import os
import json
import glob
from pathlib import Path

def load_raw_data(data_folder: str):
    """
    Charge tous les fichiers JSON du dossier spécifié.
    Retourne un dictionnaire { '1': {...}, '2': {...} } où les clés sont les IDs des points.
    """
    data = {}
    # On cherche tous les fichiers .json dans le dossier
    files = glob.glob(os.path.join(data_folder, "*.json"))
    
    if not files:
        print(f" ATTENTION : Aucun fichier JSON trouvé dans {data_folder} !")
        return {}

    print(f"Chargement de {len(files)} fichiers depuis {data_folder}...")

    for file_path in files:
        # Extraction de l'ID du point depuis le nom de fichier (ex: ball_data_12.json -> 12)
        filename = Path(file_path).stem # ball_data_12
        try:
            point_id = filename.split('_')[-1] # 12
        except:
            point_id = filename # Fallback si le nom est bizarre
            
        with open(file_path, 'r') as f:
            data[point_id] = json.load(f)
            
    return data

def save_predictions(results: dict, output_path: str):
    """Sauvegarde le dictionnaire de résultats en JSON."""
    # Créer le dossier parent si inexistant
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f" Prédictions sauvegardées dans {output_path}")