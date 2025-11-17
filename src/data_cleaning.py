import pandas as pd
import numpy as np
import sys
import os
import yaml

# Ensure the project root path is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage
# Import utils methods
from src.utils import parse_indian_number, clean_percentage, ensure_dir, load_csv_data

class DataCleaner:
    def __init__(self, config, logger=None):
        """Initialize Data Cleaner"""
        self.config = config
        self.logger = logger if logger is not None else setup_logging()
        self.module_name = "DataCleaner"
        
    def load_data(self, file_path):
        """Load raw CSV data using simple read, before cleaning non-numeric entries."""
        try:
            log_stage(self.logger, 'LOAD_RAW_DATA', 'START', file=file_path)
            # Use raw read to keep non-standard formats as strings initially
            df = pd.read_csv(file_path, low_memory=False) 
            log_stage(self.logger, 'LOAD_RAW_DATA', 'SUCCESS', rows=len(df), cols=len(df.columns))
            return df
        except Exception as e:
            log_stage(self.logger, 'LOAD_RAW_DATA', 'FAILURE', error=str(e))
            raise
    
    def clean_numeric_columns(self, df):
        """Clean columns with Indian number formatting and currency/percentage symbols."""
        try:
            log_stage(self.logger, 'CLEAN_NUMERIC', 'START')
            
            # --- Renaming the datetime column early to prevent NameError in parse_datetime ---
            if 'Datetime' in df.columns:
                 df = df.rename(columns={'Datetime': 'timestamp_raw'})
            elif 'Date' in df.columns:
                 df = df.rename(columns={'Date': 'timestamp_raw'})
            
            # Columns expected to contain Indian/currency formatting
            indian_cols = [
                'Searches', 'Searches which got estimate', 'Searches for Quotes',
                'Searches which got Quotes', 'Bookings', 'Completed Trips',
                'Cancelled Bookings', "Drivers' Earnings", 'Distance Travelled (km)',
                'Average Fare per Trip', 'Traffic Volume', 'Average Speed', 
                'Congestion Level', 'Temperature', 'Incident Reports', 
                'Road Capacity Utilization', 'event importance'
            ]
            
            for col in indian_cols:
                if col in df.columns:
                    df[col] = df[col].apply(parse_indian_number)
            
            # Percentage columns
            pct_cols = [
                'Search-to-estimate Rate', 'Estimate-to-search for quotes Rate',
                'Quote Acceptance Rate', 'Quote-to-booking Rate',
                'Booking Cancellation Rate', 'Conversion Rate'
            ]
            
            for col in pct_cols:
                if col in df.columns:
                    df[col] = df[col].apply(clean_percentage)
            
            log_stage(self.logger, 'CLEAN_NUMERIC', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'CLEAN_NUMERIC', 'FAILURE', error=str(e))
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies (FFILL/BFILL for time-series, Median/Mode for others)."""
        try:
            log_stage(self.logger, 'HANDLE_MISSING', 'START')
            
            initial_missing = df.isnull().sum().sum()
            
            # --- FIX: Use snake_case column names, as this runs AFTER align_spatial_temporal ---
            # (e.g., 'traffic_volume', 'average_speed', 'h3_index')
            ts_cols = ['traffic_volume', 'average_speed', 'congestion_level', 'temperature']
            for col in ts_cols:
                if col in df.columns and df[col].dtype in (np.float64, np.int64):
                    # Group by h3_index to fill missing values within each zone
                    df[col] = df.groupby('h3_index')[col].ffill().bfill()
            
            # Median for remaining numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    # Calculate median per-zone and fill, then fill remaining NaNs with global median
                    global_median = df[col].median()
                    df[col] = df.groupby('h3_index')[col].transform(lambda x: x.fillna(x.median()))
                    df[col] = df[col].fillna(global_median)
            
            # Mode for remaining categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

            final_missing = df.isnull().sum().sum()
            log_stage(self.logger, 'HANDLE_MISSING', 'SUCCESS', 
                      filled=initial_missing - final_missing)
            return df
        except Exception as e:
            log_stage(self.logger, 'HANDLE_MISSING', 'FAILURE', error=str(e))
            raise
    
    def parse_datetime(self, df, date_col='timestamp_raw'):
        """Parse the main Datetime column, extract temporal features, and rename to 'timestamp'."""
        try:
            log_stage(self.logger, 'PARSE_DATETIME', 'START')
            
            if date_col not in df.columns:
                 raise ValueError(f"Date column '{date_col}' not found. Available: {df.columns.to_list()}")

            df['timestamp'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')

            invalid_dates = df['timestamp'].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"[PARSE_DATETIME] {invalid_dates} invalid dates found; dropping them.")
                df = df.dropna(subset=['timestamp'])

            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Create time bucket using config setting
            bucket_min = self.config['features']['time_buckets_minutes']
            df['time_bucket'] = df['timestamp'].dt.floor(f'{bucket_min}T')
            
            # Drop the raw column
            df = df.drop(columns=[date_col])

            log_stage(self.logger, 'PARSE_DATETIME', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'PARSE_DATETIME', 'FAILURE', error=str(e))
            raise
    
    def remove_duplicates(self, df):
        """Remove duplicate rows based on all columns."""
        try:
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'START')
            initial = len(df)
            df = df.drop_duplicates()
            removed = initial - len(df)
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'SUCCESS', removed=removed)
            return df
        except Exception as e:
            log_stage(self.logger, 'REMOVE_DUPLICATES', 'FAILURE', error=str(e))
            raise
    
    def align_spatial_temporal(self, df):
        """Ensure spatial (H3) and temporal consistency by sorting and snake_case renaming."""
        try:
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'START')
            
            # Convert all column names to snake_case after numeric/percentage cleanup
            df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')','').replace("'", "").lower() for c in df.columns]

            # Rename the common zone column to the standard 'h3_index'
            if 'zone' in df.columns:
                 df = df.rename(columns={'zone': 'h3_index'})
            elif 'hex_id' in df.columns:
                 df = df.rename(columns={'hex_id': 'h3_index'})
            
            if 'h3_index' not in df.columns:
                 self.logger.warning("[ALIGN_SPATIAL_TEMPORAL] Missing 'h3_index' column, skipping spatial operations.")
                 return df

            df['h3_index'] = df['h3_index'].astype(str)
            
            # --- FIX: Check if timestamp column exists before sorting ---
            if 'timestamp' not in df.columns:
                raise KeyError("'timestamp' column not found. Ensure parse_datetime ran first.")
                
            # Sort by h3_index and time is CRUCIAL for windowing/interpolation
            df = df.sort_values(['h3_index', 'timestamp']).reset_index(drop=True)
            
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'SUCCESS')
            return df
        except Exception as e:
            log_stage(self.logger, 'ALIGN_SPATIAL_TEMPORAL', 'FAILURE', error=str(e))
            raise
    
    # --- CONSOLIDATED RUNNER ---
    def run_cleaning_pipeline(self, input_file, output_file):
        """
        Executes the full data cleaning pipeline sequentially.
        """
        try:
            self.logger.info("\n" + "="*40)
            self.logger.info(f"[{self.module_name}] Starting full cleaning pipeline.")
            self.logger.info("="*40)
            
            # --- FIX: Re-ordered the pipeline steps ---
            
            # 1. Load data
            df = self.load_data(input_file)
            
            # 2. Clean numeric/percentage formats (also renames 'Datetime' -> 'timestamp_raw')
            df = self.clean_numeric_columns(df)
            
            # 3. Remove duplicates
            df = self.remove_duplicates(df)
            
            # 4. Parse datetime (consumes 'timestamp_raw', creates 'timestamp')
            df = self.parse_datetime(df)
            
            # 5. Standardize column naming (snake_case) and sort (now finds 'timestamp')
            df = self.align_spatial_temporal(df)
            
            # 6. Final Missing Value Imputation (now runs on snake_case columns)
            df = self.handle_missing_values(df) 
            
            # Save processed file
            ensure_dir(os.path.dirname(output_file))
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"[{self.module_name}] SUCCESS: Cleaned data saved to {output_file} with {len(df)} records.")
            return df
            
        except Exception as e:
            self.logger.error(f"[{self.module_name}] FAILED Cleaning pipeline failed: {str(e)}")
            raise # Re-raise the exception to terminate the orchestrator