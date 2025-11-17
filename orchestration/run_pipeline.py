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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
# Imports for existing stages
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.feature_store import FeatureStore
from src.alert_system import AlertSystem

# Imports for new stages (Member 1 & 2)
from src.models.infer import infer_predictions
from src.optimization.assignments import get_repositioning_plan


# Define output paths explicitly
MODEL_PREDICTIONS_PATH = 'datasets/forecast_15min_predictions.csv'
REPOSITIONING_PLAN_PATH = 'datasets/repositioning_plan.csv'
ENGINEERED_FEATURES_PATH = 'datasets/engineered_features.csv'


def run_complete_pipeline(config_path='config/config.yaml'):
    """
    Execute the complete data pipeline:
    1. Data Cleaning & Alignment
    2. Feature Engineering
    3. Feature Store Management
    4. Model Inference (GNN/LSTM/TFT)
    5. Driver Repositioning Optimization
    6. Surge Alert Detection & Notification
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(
        log_file=config['monitoring']['log_file'],
        log_level=config['monitoring']['log_level']
    )
    
    logger.info("")
    logger.info("="*80)
    logger.info("         EQUI-RIDE SURGE AI - FULL PRODUCTION RUN         ")
    logger.info("="*80)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    try:
        # Determine paths based on config for consistency
        processed_file = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
        features_file = os.path.join(config['data']['features_path'], 'engineered_features.csv')


        # ========== STAGE 1: DATA CLEANING (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 1: DATA CLEANING & ALIGNMENT")
        logger.info("*" * 80)
        
        cleaner = DataCleaner(config, logger)
        input_file = config['data']['input_csv'] # Assumed to be Data_set.csv
        
        cleaned_df = cleaner.run_cleaning_pipeline(input_file, processed_file)
        logger.info(f"SUCCESS Stage 1 Complete: {len(cleaned_df)} records cleaned")
        
        
        # ========== STAGE 2: FEATURE ENGINEERING (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("*" * 80)
        
        engineer = FeatureEngineer(config, logger)
        
        # NOTE: features_df must contain the input needed for the ML model
        features_df = engineer.run_feature_engineering_pipeline(processed_file, features_file)
        logger.info(f"SUCCESS Stage 2 Complete: {len(features_df.columns)} features generated")
        
        
        # ========== STAGE 3: FEATURE STORE (MEMBER 3) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 3: FEATURE STORE MANAGEMENT")
        logger.info("*" * 80)
        
        store = FeatureStore(config, logger)
        store.connect()
        store.create_tables()
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        records_inserted = store.insert_features(features_df, batch_id=batch_id)
        logger.info(f"SUCCESS Stage 3 Complete: {records_inserted} records inserted into feature store")
        
        
        # ========== STAGE 4: MODEL INFERENCE (MEMBER 1'S WORK) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 4: MODEL INFERENCE (GNN/LSTM/TFT)")
        logger.info("*" * 80)
        
        # This function generates the 'datasets/forecast_15min_predictions.csv' file
        infer_predictions(features_path=features_file, output_path=MODEL_PREDICTIONS_PATH) 
        logger.info(f"SUCCESS Stage 4 Complete: Forecast predictions ready at {MODEL_PREDICTIONS_PATH}")

        
        # ========== STAGE 5: REPOSITIONING OPTIMIZATION (YOUR WORK - MEMBER 2) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 5: DRIVER REPOSITIONING OPTIMIZATION (PROFIT, FAIRNESS, ECO)")
        logger.info("*" * 80)
        
        repositioning_plan_df = get_repositioning_plan()
        
        if not repositioning_plan_df.empty:
            repositioning_plan_df.to_csv(REPOSITIONING_PLAN_PATH, index=False)
            logger.info(f"SUCCESS Stage 5 Complete: Repositioning Plan saved to {REPOSITIONING_PLAN_PATH}")
        else:
            logger.warning("SUCCESS Stage 5 Complete: No repositioning plan was generated.")

            
        # ========== STAGE 6: SURGE ALERT DETECTION (MEMBER 4) ==========
        logger.info("\n" + "*" * 80)
        logger.info("STAGE 6: SURGE ALERT DETECTION & NOTIFICATION")
        logger.info("*" * 80)
        
        alert_system = AlertSystem(config, logger)
        
        # Check for surge conditions based on the new features_df
        forecast_csv_path = MODEL_PREDICTIONS_PATH

        sent_count = alert_system.process_forecast_alerts(forecast_csv_path)

        logger.info(f"SUCCESS Stage 6 Complete: {sent_count} forecast-based alerts processed")  
        store.close()
        
        
        # ========== PIPELINE SUCCESS ==========
        logger.info("\n" + "="*80)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nSUMMARY:")
        logger.info(f"   Records processed: {len(features_df)}")
        logger.info(f"   Features generated: {len(features_df.columns)}")
        logger.info(f"   Batch ID: {batch_id}")
        logger.info(f"   Alerts detected: {sent_count}")
        logger.info(f"   Repositioning Plan created: {len(repositioning_plan_df)} assignments")
        logger.info("\nOUTPUT FILES:")
        logger.info(f"   Processed data: {processed_file}")
        logger.info(f"   Engineered features: {features_file}")
        logger.info(f"   Forecast predictions: {MODEL_PREDICTIONS_PATH}")
        logger.info(f"   Repositioning Plan: {REPOSITIONING_PLAN_PATH}")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    success = run_complete_pipeline()
    # sys.exit(0 if success else 1) # Commented out exit for interactive environments

if __name__ == "__main__":
    main()