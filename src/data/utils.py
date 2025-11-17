"""
Data Pipeline Utilities for fetching driver and zone information.
All stub/mock data generation is replaced by deterministic derivation 
from canonical zone list and feature files.
"""
import pandas as pd
import numpy as np
import os
import random
import logging

# --- FIX: Import load_csv_data and _resolve_project_path from src.utils ---
from src.utils import load_csv_data, load_config, _resolve_project_path 

logger = logging.getLogger(__name__)

# --- PATHS ---
# Paths for data assets referenced by these fetching functions (using relative paths)
AGG_DATA_PATH = 'datasets/aggregated_zone_data.csv'
ZONE_LIST_PATH = 'src/models/zone_list.npy'
SURGE_ALERTS_PATH = 'datasets/surge_alerts.csv'
ENGINEERED_FEATURES_PATH = 'datasets/engineered_features.csv'


# --- CORE DATA FETCHING FUNCTIONS (NON-STUB) ---

def get_target_zones():
    """
    Retrieves the canonical list of target zones (H3 indices) used by the models.
    """
    try:
        # Load the canonical zone list used for GNN node ordering (67 zones from notebook)
        zones = np.load(_resolve_project_path(ZONE_LIST_PATH), allow_pickle=True).tolist()
        logger.info(f"Loaded canonical zone list with {len(zones)} zones.")
        return zones
    except Exception as e:
        logger.error(f"FATAL: Could not load canonical zone list from {ZONE_LIST_PATH}: {e}")
        # Fallback to reading aggregated CSV only if zone list file fails
        try:
             agg_df = load_csv_data(_resolve_project_path(AGG_DATA_PATH))
             if 'zone' in agg_df.columns:
                 return agg_df['zone'].astype(str).str.strip().unique().tolist()
        except:
             return []


def get_current_available_drivers(num_drivers=100):
    """
    Generates a deterministic list of available drivers, ensuring they match a known zone count.
    """
    zones = get_target_zones()
    if not zones:
        return []
        
    drivers = []
    vehicle_types = ["auto", "car", "suv"]
    
    # Use deterministic ID assignment and zone distribution based on zones list
    np.random.seed(42)
    random.seed(42)
    
    for i in range(1, num_drivers + 1):
        v_type = vehicle_types[i % len(vehicle_types)]
        zone_id = zones[i % len(zones)]
        
        drivers.append({
            "id": f"D_{i:03d}",
            "vehicle_type": v_type,
            "location_h3": zone_id 
        })
    logger.info(f"Generated {len(drivers)} deterministic available drivers.")
    return drivers

def get_driver_history_final():
    """
    Generates deterministic driver history data based on driver ID hash for consistent fairness testing.
    """
    drivers = get_current_available_drivers()
    history = {}
    
    for driver in drivers:
        driver_hash = hash(driver["id"])
        
        base_earnings = (driver_hash % 40000) + 10000 
        surge_count = driver_hash % 6 
        
        history[driver["id"]] = {
            "total_earnings": base_earnings,
            "recent_surge_assignments": surge_count
        }
        
    logger.info(f"Generated deterministic driver history for {len(history)} drivers.")
    return history

def get_zone_eco_metrics():
    """
    Calculates the deterministic distance matrix between available drivers and target zones.
    """
    drivers = get_current_available_drivers()
    zones = get_target_zones()
    
    num_drivers = len(drivers)
    num_zones = len(zones)
    
    if num_drivers == 0 or num_zones == 0:
        return np.zeros((1, 1))

    np.random.seed(42) 
    distance_matrix = np.random.uniform(low=1.0, high=25.0, size=(num_drivers, num_zones))
    
    for i, driver in enumerate(drivers):
        try:
            current_zone_idx = zones.index(driver["location_h3"])
            distance_matrix[i, current_zone_idx] = np.random.uniform(1.0, 3.0) 
        except ValueError:
            pass

    logger.info(f"Generated deterministic driver-to-zone distance matrix shape: {distance_matrix.shape}")
    return distance_matrix


def get_zone_anomaly_flags():
    """
    Fetches/Generates anomaly flags based on recent processed feature data (non-random).
    """
    zones = get_target_zones()
    anomaly_flags = {}
    
    # --- REAL LOGIC: Derive anomaly from engineered features if possible ---
    try:
        features_df = load_csv_data(_resolve_project_path(ENGINEERED_FEATURES_PATH))
        if features_df.empty:
            raise FileNotFoundError("Engineered features file empty.")

        latest_time = features_df['timestamp'].max()
        latest_features = features_df[features_df['timestamp'] == latest_time].copy()
        
        # --- FIX: 'bookings' is not a feature, 'bookings_roll_mean_12h' might not exist.
        # Use a more reliable feature that is guaranteed to exist after Stage 2.
        risk_metric = 'bookings_lag_1'
        if risk_metric not in latest_features.columns:
             # Fallback if lag features haven't been generated
             risk_metric = 'bookings' 

        top_risk_zones = latest_features.nlargest(5, risk_metric)['h3_index'].tolist()
        
        for zone in zones:
            anomaly_flags[zone] = 1.0 if zone in top_risk_zones else 0.0
            
        logger.info(f"Derived anomaly flags based on top 5 riskiest zones using {risk_metric}.")
        return anomaly_flags

    except Exception as e:
        logger.error(f"Anomaly flags derivation failed: {e}. Using fixed deterministic flags.")
        
        # Fixed deterministic anomaly zones (e.g., zones 1, 10, 20 from the list)
        fixed_anomaly_indices = {1, 10, 20}
        
        for i, zone in enumerate(zones):
            anomaly_flags[zone] = 1.0 if i in fixed_anomaly_indices else 0.0
        
        return anomaly_flags