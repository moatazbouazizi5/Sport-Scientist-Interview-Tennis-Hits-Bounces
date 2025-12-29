import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class TennisClassifier:
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            # Poids pour gérer le déséquilibre (Hit/Bounce sont rares)
            # On donne plus d'importance aux classes 1 et 2
            scale_pos_weight=params.get('scale_pos_weight', 1), 
            objective='multi:softmax',
            num_class=3, # 0: Air, 1: Bounce, 2: Hit
            n_jobs=-1
        )

    def train(self, X, y, sample_weight=None):
        print(f"Entraînement sur {len(X)} frames...")
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        print("Entraînement terminé.")

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        print("\n--- Rapport de Classification ---")
        # Noms des classes : 0=Air, 1=Bounce, 2=Hit
        print(classification_report(y, preds, target_names=['Air', 'Bounce', 'Hit']))
        return preds

    def optimize_thresholds(self, X_val, y_val):
        """Optimize simple per-class thresholds for classes 1 (bounce) and 2 (hit).

        Strategy:
        - Compute `probs = predict_proba(X_val)`.
        - For each target class c in (1,2), scan thresholds 0.0..1.0 and pick the
          threshold that maximizes the class-specific F1 (treating it as a binary problem).
        - Store thresholds in `self.thresholds` for later use by `predict_smart`.
        """
        try:
            probs = self.model.predict_proba(X_val)
        except Exception:
            print("predict_proba not available; skipping threshold optimization.")
            self.thresholds = {1: 0.5, 2: 0.5}
            return self.thresholds

        y = np.asarray(y_val)
        thresholds = {}
        for c in (1, 2):
            best_thr = 0.5
            best_f1 = -1.0
            base_preds = probs.argmax(axis=1)
            for thr in np.linspace(0.0, 1.0, 101):
                preds = base_preds.copy()
                preds[probs[:, c] >= thr] = c
                score = f1_score((y == c).astype(int), (preds == c).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thr = thr
            thresholds[c] = float(best_thr)

        self.thresholds = thresholds
        print(f"Optimized thresholds: {self.thresholds}")
        return self.thresholds

    def predict_smart(self, X):
        """Predict using the model and apply optimized thresholds if available.

        Returns (preds_encoded, probs)
        """
        probs = None
        try:
            probs = self.model.predict_proba(X)
        except Exception:
            probs = None

        if probs is None:
            preds = self.model.predict(X)
            return np.asarray(preds), None

        preds = probs.argmax(axis=1)

        thr1 = self.thresholds.get(1, 0.5) if hasattr(self, 'thresholds') else 0.5
        thr2 = self.thresholds.get(2, 0.5) if hasattr(self, 'thresholds') else 0.5

        mask1 = probs[:, 1] >= thr1
        mask2 = probs[:, 2] >= thr2

        for i in range(len(preds)):
            if mask1[i] and mask2[i]:
                preds[i] = 1 if probs[i, 1] >= probs[i, 2] else 2
            elif mask1[i]:
                preds[i] = 1
            elif mask2[i]:
                preds[i] = 2

        return np.asarray(preds), probs

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Modèle sauvegardé : {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Modèle chargé : {path}")