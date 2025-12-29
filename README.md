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
