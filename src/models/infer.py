import pandas as pd
import logging
import numpy as np
import os
from datetime import datetime, timedelta

# --- Importing the core model loading/prediction functions from utils ---
from src.utils import load_csv_data, load_model, predict_demand, _resolve_project_path

logger = logging.getLogger(__name__)

# --- MODEL CONFIGURATION ---
LGBM_MODEL_PATH = 'src/models/lgb_bookings_1step.pkl'
GNN_MODEL_PATH = 'src/models/gwn_L12.pt' 
FEATURES_PATH = 'datasets/engineered_features.csv'
MODEL_PREDICTIONS_PATH = 'datasets/forecast_15min_predictions.csv'

# --- EXACT FEATURE SETS ---
LGBM_FEATURES = [
    'bookings_lag_1', 'bookings_lag_2', 'bookings_lag_3', 'bookings_lag_4',
    'traffic_volume', 'average_speed', 'congestion_level', 'temperature', 
    'drivers_earnings', 'distance_travelled_km', 
    'hour', 'day_of_week', 'is_weekend', 'zone_lbl', 'event_lbl'
]

GNN_FEATURES = [
    'completed_trips', 'traffic_volume', 'average_speed', 
    'congestion_level', 'temperature', 'drivers_earnings', 
    'distance_travelled_km', 'event_importance', 
    'event_type', 'pred_bookings'
]

def infer_predictions(features_path=FEATURES_PATH, output_path=MODEL_PREDICTIONS_PATH):
    """
    Runs Hybrid Inference using the latest STABLE data snapshot.
    """
    logger.info(f"Starting HYBRID ENSEMBLE inference using features from {features_path}...")
    
    # 1. Load Features
    feature_df = load_csv_data(_resolve_project_path(features_path), parse_dates=['timestamp'])
    if feature_df.empty:
        logger.error("Feature data is empty. Cannot run inference.")
        return pd.DataFrame()

    # 2. Preprocessing & renaming
    feature_df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace("'", "") for c in feature_df.columns]
    rename_map = {'h3': 'h3_index', 'hex_id': 'h3_index', 'zone_id': 'h3_index', 'zone': 'h3_index', 'bookings': 'bookings', 'weekday': 'day_of_week'}
    feature_df.rename(columns=rename_map, inplace=True)

    # 3. Get Latest *STABLE* Snapshot
    # FIX: Don't just take max(). Find the latest timestamp that has data for many zones.
    if 'timestamp' not in feature_df.columns:
        logger.error("Missing 'timestamp' column.")
        return pd.DataFrame()

    # Get unique timestamps sorted
    unique_times = np.sort(feature_df['timestamp'].unique())
    
    target_features = pd.DataFrame()
    latest_data_time = None

    # Look backwards from the end to find a snapshot with > 10 zones
    # This avoids "tail-end" issues where the last second of data is incomplete
    for ts in unique_times[::-1]:
        slice_df = feature_df[feature_df['timestamp'] == ts]
        if len(slice_df) > 10: # Threshold: Assume a valid snapshot has at least 10 zones
            target_features = slice_df.copy()
            latest_data_time = ts
            break
    
    # Fallback if nothing found
    if target_features.empty:
        logger.warning("No complete snapshot found. Using absolute latest (might be partial).")
        latest_data_time = feature_df['timestamp'].max()
        target_features = feature_df[feature_df['timestamp'] == latest_data_time].copy()

    if target_features.empty:
        logger.error("Target features empty.")
        return pd.DataFrame()
        
    logger.info(f"Using input features from data timestamp: {latest_data_time} (Zones: {len(target_features)})")

    # --- Prepare Missing Columns ---
    if 'zone_lbl' not in target_features.columns: target_features['zone_lbl'] = pd.factorize(target_features['h3_index'])[0]
    if 'event_lbl' not in target_features.columns: target_features['event_lbl'] = 0
    if 'is_weekend' not in target_features.columns:
        if 'day_of_week' in target_features.columns: target_features['is_weekend'] = target_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        else: target_features['is_weekend'] = 0
    
    target_features['event_type'] = pd.to_numeric(target_features.get('event_type', 0), errors='coerce').fillna(0)

    # ----------------------------------------------------
    # STEP A: LightGBM Prediction
    # ----------------------------------------------------
    lgbm_preds = np.zeros(len(target_features))
    try:
        lgbm_model = load_model(LGBM_MODEL_PATH)
        missing_lgb = [c for c in LGBM_FEATURES if c not in target_features.columns]
        if missing_lgb:
            for c in missing_lgb: target_features[c] = 0
            
        lgbm_input = target_features[LGBM_FEATURES].copy()
        lgbm_preds = predict_demand(lgbm_model, lgbm_input)
        target_features['pred_bookings'] = lgbm_preds 
        
    except Exception as e:
        logger.error(f"LGBM Prediction failed: {e}")
        target_features['pred_bookings'] = 0

    # ----------------------------------------------------
    # STEP B: GNN Prediction
    # ----------------------------------------------------
    gnn_preds = np.zeros(len(target_features))
    try:
        gnn_model = load_model(GNN_MODEL_PATH)
        missing_gnn = [c for c in GNN_FEATURES if c not in target_features.columns]
        if missing_gnn:
             for c in missing_gnn: target_features[c] = 0

        gnn_input_df = target_features[['h3_index', 'timestamp'] + GNN_FEATURES].copy()
        gnn_preds = predict_demand(gnn_model, gnn_input_df)
        
    except Exception as e:
        logger.error(f"GNN Prediction failed: {e}")

    # ----------------------------------------------------
    # STEP C: Final Ensemble & Saving
    # ----------------------------------------------------
    min_len = min(len(lgbm_preds), len(gnn_preds))
    lgbm_preds = lgbm_preds[:min_len]
    gnn_preds = gnn_preds[:min_len]
    target_features = target_features.iloc[:min_len]

    final_preds = (lgbm_preds + gnn_preds) / 2.0
    
    # Use current real-world time for "Live" appearance
    next_time = (datetime.now() + timedelta(minutes=15)).replace(microsecond=0)
    
    results = pd.DataFrame({
        'h3_index': target_features['h3_index'].values,
        'pred_bookings_15min': final_preds,
        'next_time': next_time
    })
    
    results['pred_bookings_15min'] = results['pred_bookings_15min'].clip(lower=0).round(1)
    
    save_path = _resolve_project_path(output_path)
    results = results[['h3_index', 'next_time', 'pred_bookings_15min']]
    results.to_csv(save_path, index=False)
    
    logger.info(f"Hybrid forecast generated for {next_time} with {len(results)} zones.")
    
    return results

if __name__=="__main__":
    infer_predictions()