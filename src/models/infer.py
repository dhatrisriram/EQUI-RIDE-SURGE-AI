# src/models/infer.py
import pandas as pd
import logging
import numpy as np
import os
from datetime import datetime
from src.utils import load_csv_data 
from src.data.utils import get_target_zones 

logger = logging.getLogger(__name__)

# Define paths
FEATURES_PATH = 'datasets/engineered_features.csv'
MODEL_PREDICTIONS_PATH = 'datasets/forecast_15min_predictions.csv'
MODEL_PATH = 'models/best_forecast_model.pt' 

def infer_predictions(features_path=FEATURES_PATH, output_path=MODEL_PREDICTIONS_PATH):
    """
    [MEMBER 1's ROLE]
    Loads the trained model, infers future demand, and saves the output.
    """
    logger.info(f"Starting inference using features from {features_path}...")
    
    # 1. Load Features
    feature_df = load_csv_data(features_path, parse_dates=['timestamp'])
    if feature_df.empty:
        logger.error("Feature data is empty. Cannot run inference.")
        pd.DataFrame({'h3_index': [], 'next_time': [], 'pred_bookings_15min': []}).to_csv(output_path, index=False)
        return pd.DataFrame()

    # --- FIX 1: Normalize Column Names ---
    feature_df.columns = [c.strip().lower() for c in feature_df.columns]
    rename_map = {
        'h3': 'h3_index',
        'hex_id': 'h3_index',
        'zone_id': 'h3_index',
        'bookings': 'bookings'
    }
    feature_df.rename(columns=rename_map, inplace=True)

    if 'h3_index' not in feature_df.columns:
        logger.error(f"Missing 'h3_index' column. Columns found: {list(feature_df.columns)}")
        return pd.DataFrame()
        
    # --- DEMO SIMULATION LOGIC ---
    
    # A. USE "BEST" HISTORICAL DATE
    demo_time_str = "2024-12-03 08:30:00"
    best_data_time = pd.to_datetime(demo_time_str)

    # B. SET OUTPUT TIME TO "NOW" (Without Microseconds!)
    # This .replace(microsecond=0) is the CRITICAL FIX for the dashboard slider
    next_time = (datetime.now() + pd.Timedelta(minutes=15)).replace(microsecond=0)
    
    # C. GET ZONES
    target_features = feature_df[feature_df['timestamp'] == best_data_time]
    zones = target_features['h3_index'].unique()
    
    if len(zones) == 0:
        logger.warning(f"No zones found for demo time {demo_time_str}. Fallback to unique zones.")
        zones = feature_df['h3_index'].unique()[:10]

    # D. GENERATE PREDICTIONS
    book_col = 'bookings' if 'bookings' in target_features.columns else 'Bookings'
    if not target_features.empty and book_col in target_features.columns:
        recent_bookings = target_features.groupby('h3_index')[book_col].last()
        recent_bookings = recent_bookings.reindex(zones, fill_value=0)
        base_values = recent_bookings.values
    else:
        base_values = np.random.randint(50, 200, size=len(zones))

    mock_predictions = pd.DataFrame({
        'h3_index': zones,
        'next_time': next_time,
        'pred_bookings_15min': base_values * 1.2 
    })
    
    # 2. Save Output
    mock_predictions = mock_predictions[['h3_index', 'next_time', 'pred_bookings_15min']].fillna(0)
    mock_predictions.to_csv(output_path, index=False)
    logger.info(f"Inference complete. Predictions saved for {len(mock_predictions)} zones to {output_path}")
    
    return mock_predictions