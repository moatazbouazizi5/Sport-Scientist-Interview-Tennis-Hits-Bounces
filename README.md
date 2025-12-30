# 🎾 Tennis Event Detection - Roland-Garros 2025

> **A production-ready pipeline for detecting tennis ball Hits and Bounces using Computer Vision data, combining Physics-based heuristics and Advanced Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-green?style=for-the-badge)]()

## 📖 Project Overview

This project processes 2D ball-tracking data from the Roland-Garros 2025 final to detect two specific events:
1.  **Bounce:** When the ball hits the court.
2.  **Hit:** When the ball is struck by a racket.

The main challenge is the **extreme class imbalance** (events represent <1% of frames) and the noise inherent in computer vision tracking. This solution implements a robust architecture to maximize **Recall** (capturing every event) while maintaining physical consistency.

---

## 📂 Project Structure

The project follows a modular "Clean Architecture" pattern for scalability and maintainability.

```text
ROLANDGARROS_TRACKING/
├── config/                  # ⚙️ Centralized Configuration
│   ├── config.yaml          # Parameters for Physics thresholds & ML hyperparameters
│   └── logging.yaml         # Logger settings
│
├── data/                    # 💾 Data Management (GitIgnored)
│   ├── raw/                 # Input JSON files
│   └── predictions/         # Output results (physics.json, supervised.json)
│
├── models/                  # 🤖 Saved Model Artifacts
│   └── xgboost_model.json   # Trained XGBoost model
│
├── scripts/                 # 🛠️ Utility Scripts
│   ├── check_preds.py       # Sanity checks
│   └── evaluate_preds.py    # Standalone evaluation
│
├── src/                     # 🧠 Core Source Code
│   ├── features.py          # Advanced Feature Engineering (Z-scores, Jerk, Lags)
│   ├── physics.py           # Unsupervised physics engine
│   ├── models.py            # XGBoost wrapper & training logic
│   ├── preprocessing.py     # Data cleaning (Savitzky-Golay smoothing)
│   ├── postprocessing.py    # Non-Maximum Suppression (NMS)
│   └── utils.py             # I/O helpers
│
├── main.py                  # 🚀 Main entry point (CLI)
├── visualize_physics.py     # Visualization tool for Method 1
├── visualize_supervised.py  # Visualization tool for Method 2
└── requirements.txt         # Dependencies
```

## 📊 Performance & Results

We prioritize **Recall** to ensure no game event is missed. The model handles the 1:100 class imbalance using weighted sampling (`sample_weight='balanced'`) and dynamic threshold optimization.

**Test Set Results (63 unseen points):**

| Class | Recall (Capture Rate) | F1-Score | Support |
|-------|-----------------------|----------|---------|
| **Air** | 96% | 0.98 | 33,721 |
| **Hit** | **93%** 🚀 | 0.43 | 307 |
| **Bounce** | **95%** 🚀 | 0.43 | 272 |

> *Note: Precision is traded for Recall. It is better to detect a false positive (which can be filtered later) than to miss a match-winning point.*

---

## 🧠 Methodologies

This repository implements two distinct approaches:

### 1. Unsupervised Method (Physics)
A baseline approach using kinematic rules without training data.
*   **Bounce:** Detects local maxima in the Y-axis (lowest point in image) combined with vertical acceleration inversion.
*   **Hit:** Detects spikes in total acceleration magnitude ($a_{mag}$), excluding zones near the ground.

### 2. Supervised Method (Machine Learning) - *Recommended*
A Gradient Boosting approach (XGBoost) trained on ~420,000 frames.

*   **Preprocessing:** Linear interpolation for missing data + Savitzky-Golay filtering (window=5, poly=2) to smooth noise.
*   **Feature Engineering:**
    *   **Context:** Lag/Lead features (t-5 to t+5) to see the trajectory shape.
    *   **Dynamics:** Calculation of **Jerk** (derivative of acceleration) and **Local Z-Scores** to normalize impacts regardless of ball speed.
*   **Post-Processing:**
    *   **NMS (Non-Maximum Suppression):** Clusters nearby detections and keeps only the highest probability candidate.
    *   **Physical Filters:** Rejects bounces occurring in the sky (Y-axis threshold).

---

## 🚀 Installation & Usage

### 1. Setup Environment
It is recommended to use a virtual environment to keep dependencies clean.

```bash
# Clone the repository
git clone https://github.com/moatazbouazizi5/Sport-Scientist-Interview-Tennis-Hits-Bounces.git
cd Sport-Scientist-Interview-Tennis-Hits-Bounces

# Create virtual env
# Windows:
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt



