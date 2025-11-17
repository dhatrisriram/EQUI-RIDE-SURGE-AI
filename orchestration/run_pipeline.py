#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Runs the complete EquiRide project pipeline end-to-end (Data, Features, ML, Optimization, Alerts)
"""
import yaml
import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path to ensure module visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports for setup and existing stages
from config.logging_config import setup_logging
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.feature_store import FeatureStore
from src.alert_system import AlertSystem
# Import path resolver from utils
from src.utils import _resolve_project_path 

# Imports for prediction and optimization stages
from src.models.infer import infer_predictions
from src.optimization.assignments import get_repositioning_plan 


# Define output paths explicitly (Relative to project root, used as inputs/outputs for functions)
MODEL_PREDICTIONS_PATH = 'datasets/forecast_15min_predictions.csv'
REPOSITIONING_PLAN_PATH = 'datasets/repositioning_plan.csv'
ENGINEERED_FEATURES_PATH = 'datasets/engineered_features.csv'

# Define Project Root for consistent file saving
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_complete_pipeline(config_path='config/config.yaml'):
    """
    Execute the complete data pipeline.
    """
    
    # Load configuration
    full_config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(
        log_file=config['monitoring']['log_file'],
        log_level=config['monitoring']['log_level']
    )
    
    logger.info("")
    logger.info("="*80)
    logger.info("          EQUI-RIDE SURGE AI - FULL PRODUCTION RUN          ")
    logger.info("="*80)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    # Variables initialized for scope reference in summary
    features_df = pd.DataFrame()
    repositioning_plan_df = pd.DataFrame()
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    records_inserted = 0
    sent_count = 0
    store = None # Initialize store outside try block for final close

    try:
        # Determine full paths for saving intermediate files
        processed_file_rel = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
        processed_file_abs = _resolve_project_path(processed_file_rel)
        
        target_features_file_abs = _resolve_project_path(ENGINEERED_FEATURES_PATH)
        
        # ========== STAGE 1: DATA CLEANING (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 1: DATA CLEANING & ALIGNMENT")
        
        cleaner = DataCleaner(config, logger)
        input_file = os.path.join(PROJECT_ROOT, config['data']['input_csv'])
        
        cleaned_df = cleaner.run_cleaning_pipeline(input_file, processed_file_abs)
        logger.info(f"SUCCESS Stage 1 Complete: {len(cleaned_df)} records cleaned")
        
        
        # ========== STAGE 2: FEATURE ENGINEERING (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        
        engineer = FeatureEngineer(config, logger)
        features_df = engineer.run_feature_engineering_pipeline(processed_file_abs, target_features_file_abs)
        logger.info(f"SUCCESS Stage 2 Complete: {len(features_df.columns)} features generated")
        
        
        # ========== STAGE 3: FEATURE STORE (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 3: FEATURE STORE MANAGEMENT")
        
        store = FeatureStore(config, logger)
        store.connect()
        store.create_tables()
        
        # Insert features (using the DataFrame output from Stage 2)
        records_inserted = store.insert_features(features_df, batch_id=batch_id)
        logger.info(f"SUCCESS Stage 3 Complete: {records_inserted} records inserted into feature store")
        
        
        # ========== STAGE 4: MODEL INFERENCE (MEMBER 1 - HYBRID) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 4: MODEL INFERENCE (HYBRID ENSEMBLE)")
        
        # infer_predictions must take the relative path to ENGINEERED_FEATURES_PATH
        infer_predictions(features_path=ENGINEERED_FEATURES_PATH, output_path=MODEL_PREDICTIONS_PATH) 
        logger.info(f"SUCCESS Stage 4 Complete: Hybrid Forecast predictions ready at {MODEL_PREDICTIONS_PATH}")

        
        # ========== STAGE 5: REPOSITIONING OPTIMIZATION (MEMBER 2) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 5: DRIVER REPOSITIONING OPTIMIZATION (PROFIT, FAIRNESS, ECO)")
        
        repositioning_plan_df = get_repositioning_plan() 
        
        if not repositioning_plan_df.empty:
            final_plan_path = os.path.join(PROJECT_ROOT, REPOSITIONING_PLAN_PATH)
            repositioning_plan_df.to_csv(final_plan_path, index=False)
            logger.info(f"SUCCESS Stage 5 Complete: Repositioning Plan saved to {REPOSITIONING_PLAN_PATH}")
        else:
            logger.warning("SUCCESS Stage 5 Complete: No repositioning plan was generated.")

            
        # ========== STAGE 6: SURGE ALERT DETECTION (MEMBER 4) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 6: SURGE ALERT DETECTION & NOTIFICATION")
        
        alert_system = AlertSystem(config, logger)
        
        # Pass the relative path for the alert system to load the predictions
        sent_count = alert_system.process_forecast_alerts(MODEL_PREDICTIONS_PATH)

        logger.info(f"SUCCESS Stage 6 Complete: {sent_count} forecast-based alerts processed") 
        
        
        # ========== PIPELINE SUCCESS ==========
        logger.info("\n" + "="*80)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if store:
            store.close()
        
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nSUMMARY:")
        logger.info(f"   Records processed: {len(features_df)}")
        logger.info(f"   Features generated: {len(features_df.columns)}")
        logger.info(f"   Batch ID: {batch_id}")
        logger.info(f"   Alerts processed: {sent_count}")
        logger.info(f"   Repositioning Plan created: {len(repositioning_plan_df)} assignments")
        logger.info("\nOUTPUT FILES:")
        logger.info(f"   Processed data: {processed_file_abs}")
        logger.info(f"   Engineered features: {target_features_file_abs}")
        logger.info(f"   Forecast predictions: {_resolve_project_path(MODEL_PREDICTIONS_PATH)}")
        logger.info(f"   Repositioning Plan: {_resolve_project_path(REPOSITIONING_PLAN_PATH)}")
        logger.info("="*80)

def main():
    """Main entry point"""
    run_complete_pipeline()

if __name__ == "__main__":
    main()