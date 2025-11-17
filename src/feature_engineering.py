import pandas as pd
import numpy as np
import os
import sys

# Ensure project paths are resolvable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_csv_data, ensure_dir, safe_divide, detect_anomalies_zscore
from config.logging_config import log_stage

class FeatureEngineer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.module_name = "FeatureEngineer"

    # --- Feature Generators ---

    def generate_demand_features(self, df):
        """Generate demand-related ratios and unfulfilled demand metrics."""
        try:
            log_stage(self.logger, 'DEMAND_FEATURES', 'START')
            
            # --- CORRECTED COLUMN REFERENCES (snake_case) ---
            df['demand_supply_ratio'] = safe_divide(
                df['searches'], df['completed_trips'], default=1.0
            )
            df['booking_success_rate'] = safe_divide(
                df['completed_trips'], df['bookings'], default=0.0
            )
            df['search_conversion_rate'] = safe_divide(
                df['bookings'], df['searches'], default=0.0
            )
            df['cancellation_rate_numeric'] = safe_divide(
                df['cancelled_bookings'], df['bookings'], default=0.0
            )
            
            # [FIX] Changed "drivers'_earnings" to "drivers_earnings"
            df['earnings_per_trip'] = safe_divide(
                df['drivers_earnings'], df['completed_trips'], default=0.0
            )
            
            df['unfulfilled_demand'] = df['searches'] - df['completed_trips']
            df['unfulfilled_demand'] = df['unfulfilled_demand'].clip(lower=0)
            
            log_stage(self.logger, 'DEMAND_FEATURES', 'SUCCESS', features=6)
            return df
        except Exception as e:
            log_stage(self.logger, 'DEMAND_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_traffic_features(self, df):
        """Generate traffic-related features (changes, variance, utilization)."""
        try:
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'START')
            
            # Congestion change rate (lag-1 difference)
            df['congestion_change'] = df.groupby('h3_index')['congestion_level'].diff().fillna(0)
            
            # Traffic volume change percentage
            df['traffic_volume_pct_change'] = df.groupby('h3_index')['traffic_volume'].pct_change().fillna(0)
            
            # Speed variance (rolling 3-period standard deviation)
            df['speed_variance'] = df.groupby('h3_index')['average_speed'].transform(
                lambda x: x.rolling(3, min_periods=1).std()
            ).fillna(0)
            
            # Capacity utilization category
            if 'road_capacity_utilization' in df.columns:
                 df['capacity_category'] = pd.cut(
                    df['road_capacity_utilization'],
                    bins=[0, 50, 75, 90, 100],
                    labels=['low', 'medium', 'high', 'critical']
                ).astype(str)
            
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'SUCCESS', features=4)
            return df
        except Exception as e:
            log_stage(self.logger, 'TRAFFIC_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_rolling_features(self, df):
        """Generate rolling window mean and max features for key time-series."""
        try:
            log_stage(self.logger, 'ROLLING_FEATURES', 'START')
            
            windows = self.config['features']['rolling_window_sizes']
            # --- CORRECTED COLUMN REFERENCES (snake_case) ---
            target_cols = ['traffic_volume', 'searches', 'completed_trips', 'congestion_level', 'bookings']
            features_count = 0
            
            for col in target_cols:
                if col not in df.columns: continue
                    
                for window in windows:
                    # Rolling mean (shifted by 1 to prevent target leakage)
                    df[f'{col}_roll_mean_{window}h'] = df.groupby('h3_index')[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                    
                    # Rolling max (shifted by 1)
                    df[f'{col}_roll_max_{window}h'] = df.groupby('h3_index')[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).max()
                    )
                    features_count += 2
            
            log_stage(self.logger, 'ROLLING_FEATURES', 'SUCCESS', features=features_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'ROLLING_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_lag_features(self, df):
        """Generate point-in-time lag features."""
        try:
            log_stage(self.logger, 'LAG_FEATURES', 'START')
            
            lags = self.config['features']['lag_periods']
            # --- CORRECTED COLUMN REFERENCES (snake_case) ---
            target_cols = ['searches', 'completed_trips', 'congestion_level', 'bookings']
            features_count = 0
            
            for col in target_cols:
                if col not in df.columns: continue
                    
                for lag in lags:
                    # Shift feature by lag period
                    df[f'{col}_lag_{lag}'] = df.groupby('h3_index')[col].shift(lag)
                    features_count += 1
            
            log_stage(self.logger, 'LAG_FEATURES', 'SUCCESS', features=features_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'LAG_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_temporal_features(self, df):
        """Generate categorical and cyclical time-based features."""
        try:
            log_stage(self.logger, 'STATIC_TEMPORAL', 'START')
            
            df['hour_category'] = pd.cut(
                df['hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening']
            ).astype(str)
            
            df['is_rush_hour'] = df['hour'].isin([8, 9, 18, 19, 20]).astype(int)
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            log_stage(self.logger, 'STATIC_TEMPORAL', 'SUCCESS', features=4)
            return df
        except Exception as e:
            log_stage(self.logger, 'STATIC_TEMPORAL', 'FAILURE', error=str(e))
            raise
    
    def generate_anomaly_features(self, df):
        """Detect anomalies in key metrics using the configured Z-score threshold."""
        try:
            log_stage(self.logger, 'ANOMALY_FEATURES', 'START')
            
            threshold = self.config['features']['anomaly_threshold']
            
            # Anomaly detection for key metrics
            df['demand_anomaly'] = df.groupby('h3_index')['searches'].transform(
                lambda x: detect_anomalies_zscore(x, threshold)
            ).astype(int)
            
            df['traffic_anomaly'] = df.groupby('h3_index')['traffic_volume'].transform(
                lambda x: detect_anomalies_zscore(x, threshold)
            ).astype(int)
            
            df['congestion_anomaly'] = df.groupby('h3_index')['congestion_level'].transform(
                lambda x: detect_anomalies_zscore(x, threshold)
            ).astype(int)
            
            # Combined anomaly flag
            df['any_anomaly'] = ((df['demand_anomaly'] == 1) | 
                                 (df['traffic_anomaly'] == 1) | 
                                 (df['congestion_anomaly'] == 1)).astype(int)
            
            anomaly_count = df['any_anomaly'].sum()
            log_stage(self.logger, 'ANOMALY_FEATURES', 'SUCCESS', 
                      features=4, anomalies_detected=anomaly_count)
            return df
        except Exception as e:
            log_stage(self.logger, 'ANOMALY_FEATURES', 'FAILURE', error=str(e))
            raise
    
    def generate_weather_event_features(self, df):
        """Generate weather and event features (encoding categorical data)."""
        try:
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'START')
            
            weather_map = {
                'Clear': 0, 'Overcast': 1, 'Cloudy': 1,
                'Rain': 2, 'Fog': 3, 'Windy': 1, 'Unknown': 0
            }
            # --- CORRECTED COLUMN REFERENCES (snake_case) ---
            if 'weather_conditions' in df.columns:
                df['weather_encoded'] = df['weather_conditions'].map(weather_map).fillna(0).astype(int)
                df['weather_severity'] = (df['weather_encoded'] >= 2).astype(int)
            
            if 'roadwork_and_construction_activity' in df.columns:
                df['has_roadwork'] = (df['roadwork_and_construction_activity'].astype(str) == 'Yes').astype(int)
            
            if 'incident_reports' in df.columns:
                df['has_incidents'] = (df['incident_reports'] > 0).astype(int)
            
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'SUCCESS', features=4)
            return df
        except Exception as e:
            log_stage(self.logger, 'WEATHER_EVENT_FEATURES', 'FAILURE', error=str(e))
            raise
    
    # --- CONSOLIDATED RUNNER ---
    def run_feature_engineering_pipeline(self, input_file, output_file):
        """Execute complete feature engineering pipeline"""
        try:
            self.logger.info("="*70)
            self.logger.info("FEATURE ENGINEERING PIPELINE - STARTED")
            self.logger.info("="*70)
            
            # Load processed data (which should be timestamp-sorted and cleaned)
            df = load_csv_data(input_file, parse_dates=['timestamp'])
            
            initial_cols = len(df.columns)
            
            # Generate all feature categories sequentially
            df = self.generate_demand_features(df)
            df = self.generate_traffic_features(df)
            df = self.generate_rolling_features(df)
            df = self.generate_lag_features(df)
            df = self.generate_temporal_features(df)
            df = self.generate_anomaly_features(df)
            df = self.generate_weather_event_features(df)
            
            # Final Cleanup: Fill any remaining NaNs from lag/rolling operations
            df = df.ffill().bfill().fillna(0)
            
            final_cols = len(df.columns)
            new_features = final_cols - initial_cols
            
            # Save engineered features
            ensure_dir(os.path.dirname(output_file))
            df.to_csv(output_file, index=False)
            
            self.logger.info("="*70)
            self.logger.info(f"SUCCESS Engineered features saved: {output_file}")
            self.logger.info(f"SUCCESS Shape: {df.shape}")
            self.logger.info(f"SUCCESS New features created: {new_features}")
            self.logger.info("FEATURE ENGINEERING PIPELINE - COMPLETED")
            self.logger.info("="*70)
            
            return df
            
        except Exception as e:
            self.logger.error(f"FAILED Feature engineering failed: {str(e)}")
            return pd.DataFrame()