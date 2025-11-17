# src/models/infer.py
import pandas as pd
import logging
import numpy as np
import os
from datetime import datetime, timedelta

# --- Importing the core model loading/prediction functions from utils ---
from src.utils import load_csv_data, load_model, predict_demand, _resolve_project_path

logger = logging.getLogger(__name__)

# --- MODEL CONFIGURATION FOR HYBRID ENSEMBLE ---
# Define paths for both models used in the ensemble
LGBM_MODEL_PATH = 'src/models/lgb_bookings_1step.pkl'
GNN_MODEL_PATH = 'src/models/gwn_L12.pt' 

FEATURES_PATH = 'datasets/engineered_features.csv'
MODEL_PREDICTIONS_PATH = 'datasets/forecast_15min_predictions.csv'


def infer_predictions(features_path=FEATURES_PATH, output_path=MODEL_PREDICTIONS_PATH):
    """
    Loads the Hybrid Model (LGBM + GNN), infers future demand by averaging their 
    predictions, and saves the ensemble forecast.
    """
    logger.info(f"Starting HYBRID ENSEMBLE inference (LGBM + GNN) using features from {features_path}...")
    
    # 1. Load Features (Latest Snapshot)
    # Use _resolve_project_path to ensure the engineered features file is found
    feature_df = load_csv_data(_resolve_project_path(features_path), parse_dates=['timestamp'])
    if feature_df.empty:
        logger.error("Feature data is empty. Cannot run inference.")
        # Save empty output file using the resolved path to prevent later stage failures
        pd.DataFrame({'h3_index': [], 'next_time': [], 'pred_bookings_15min': []}).to_csv(_resolve_project_path(output_path), index=False)
        return pd.DataFrame()

    # --- Feature Preparation ---
    feature_df.columns = [c.strip().lower() for c in feature_df.columns]
    rename_map = {'h3': 'h3_index', 'hex_id': 'h3_index', 'zone_id': 'h3_index', 'bookings': 'bookings'}
    feature_df.rename(columns=rename_map, inplace=True)

    if 'h3_index' not in feature_df.columns:
        logger.error(f"Missing 'h3_index' column. Columns found: {list(feature_df.columns)}")
        return pd.DataFrame()
        
    if 'timestamp' not in feature_df.columns:
        logger.error("Missing 'timestamp' column in features. Cannot determine latest features.")
        return pd.DataFrame()
        
    latest_time = feature_df['timestamp'].max()
    target_features = feature_df[feature_df['timestamp'] == latest_time].copy()
    
    if target_features.empty:
        logger.error("Latest feature data slice is empty. Cannot run inference.")
        return pd.DataFrame()

    # ----------------------------------------------------
    # --- HYBRID MODEL PREDICTION ---
    # ----------------------------------------------------
    
    # A. Load and Predict with LGBM Model (The time-series component)
    try:
        lgbm_model = load_model(LGBM_MODEL_PATH)
        lgbm_preds = predict_demand(lgbm_model, target_features)
    except Exception as e:
        logger.error(f"LGBM Prediction failed ({LGBM_MODEL_PATH}). Using zero array: {e}")
        lgbm_preds = np.zeros(len(target_features))

    # B. Load and Predict with GNN Model (The spatial component)
    try:
        gnn_model = load_model(GNN_MODEL_PATH)
        gnn_preds = predict_demand(gnn_model, target_features)
    except Exception as e:
        logger.error(f"GNN Prediction failed ({GNN_MODEL_PATH}). Using zero array: {e}")
        gnn_preds = np.zeros(len(target_features))
        
    # Ensure prediction arrays are consistent length
    if len(lgbm_preds) != len(gnn_preds):
        min_len = min(len(lgbm_preds), len(gnn_preds))
        # Truncate to the minimum length to allow ensemble averaging
        lgbm_preds = lgbm_preds[:min_len]
        gnn_preds = gnn_preds[:min_len]
        target_features = target_features.iloc[:min_len]
        logger.warning(f"Prediction length mismatch. Truncating predictions to {min_len} zones.")


    # C. Ensemble Blending (Simple Averaging)
    final_predictions_array = (lgbm_preds + gnn_preds) / 2.0
    
    # D. Calculate the predicted time slot
    next_time = (latest_time + pd.Timedelta(minutes=15)).replace(microsecond=0)

    # E. Construct the Predictions DataFrame
    real_predictions = pd.DataFrame({
        'h3_index': target_features['h3_index'].values,
        'next_time': next_time,
        'pred_bookings_15min': final_predictions_array.flatten(), 
    })
    
    # 2. Save Output
    real_predictions = real_predictions[['h3_index', 'next_time', 'pred_bookings_15min']].fillna(0)
    # Save using the resolved path
    real_predictions.to_csv(_resolve_project_path(output_path), index=False)
    logger.info(f"Hybrid inference complete. Predictions saved for {len(real_predictions)} zones to {output_path}")
    
    return real_predictions

if __name__=="__main__":
    infer_predictions()