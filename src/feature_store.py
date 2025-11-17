import pandas as pd
import sqlite3
import os
import sys

# Ensure project paths are resolvable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging, log_stage
from src.utils import ensure_dir, _resolve_project_path 

class FeatureStore:
    def __init__(self, config, logger):
        """Initialize Feature Store"""
        self.config = config
        self.logger = logger
        
        # Resolve DB path using the centralized utility function
        self.db_path = _resolve_project_path(config['data']['feature_store_path'])
        
        # Ensure directory exists before connecting
        ensure_dir(os.path.dirname(self.db_path))
        self.conn = None
        self.module_name = "FeatureStore"
        self.table_name = "engineered_features"

    def connect(self):
        """Connects to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info(f"[{self.module_name}] SUCCESS Connected to feature store: {self.db_path}")
        except Exception as e:
            self.logger.error(f"[{self.module_name}] FAILED Failed to connect: {str(e)}")
            raise
    
    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info(f"[{self.module_name}] SUCCESS Feature store connection closed")
            self.conn = None # Set connection to None after closing

    def create_tables(self):
        """
        Creates the 'engineered_features' table and 'ingestion_log' table.
        The schema is simplified to include core features produced by Stage 2.
        """
        try:
            log_stage(self.logger, 'CREATE_TABLES', 'START')
            cursor = self.conn.cursor()
            
            # Simplified schema mapping key outputs from feature_engineering.py
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    timestamp TEXT NOT NULL,
                    h3_index TEXT NOT NULL,
                    bookings REAL,
                    hour INTEGER,
                    day_of_week INTEGER,
                    -- Sample engineered features --
                    bookings_roll_mean_24h REAL,
                    bookings_lag_1 REAL,
                    any_anomaly INTEGER,
                    congestion_level REAL,
                    PRIMARY KEY (timestamp, h3_index)
                ) WITHOUT ROWID;
            """)

            # Ingestion log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    records_inserted INTEGER,
                    ingestion_time TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            log_stage(self.logger, 'CREATE_TABLES', 'SUCCESS')
            
        except Exception as e:
            log_stage(self.logger, 'CREATE_TABLES', 'FAILURE', error=str(e))
            raise
    
    def insert_features(self, features_df: pd.DataFrame, batch_id: str):
        """
        Inserts a new batch of features into the store, handling column selection
        and utilizing INSERT OR REPLACE to avoid unique constraint violations.
        """
        if features_df.empty or self.conn is None:
            self.logger.warning(f"[{self.module_name}] No features to insert or DB connection is closed.")
            return 0
        
        try:
            log_stage(self.logger, 'INSERT_FEATURES', 'START', batch=batch_id)
            
            # --- Column Selection for Schema Consistency ---
            # Columns expected in the SQL table
            sql_cols = [
                'timestamp', 'h3_index', 'bookings', 'hour', 'day_of_week',
                'bookings_roll_mean_24h', 'bookings_lag_1', 'any_anomaly', 'congestion_level'
            ]
            
            # Map FeatureEngineer outputs (snake_case) to match potential SQL schema variations
            insert_df = features_df.rename(columns={
                'bookings_roll_mean_24': 'bookings_roll_mean_24h', 
                'bookings_lag_1': 'bookings_lag_1'
            }, errors='ignore').copy()
            
            # Filter to ensure we only select columns that exist in the DataFrame
            available_cols = [c for c in sql_cols if c in insert_df.columns]
            final_insert_df = insert_df[available_cols].copy()
            
            # Ensure timestamp is string format for SQLite
            final_insert_df['timestamp'] = final_insert_df['timestamp'].astype(str)
            
            # --- FIX: Use INSERT OR REPLACE logic ---
            # We cannot use to_sql(if_exists='append') because it fails on duplicates.
            # We define the query manually to use the upsert capability.
            
            placeholders = ', '.join(['?'] * len(available_cols))
            columns_formatted = ', '.join(available_cols)
            
            query = f"""
                INSERT OR REPLACE INTO {self.table_name} 
                ({columns_formatted}) 
                VALUES ({placeholders})
            """
            
            # Convert DataFrame to list of tuples for executemany
            # We use itertuples(index=False) to get standard Python types where possible
            data_to_insert = list(final_insert_df.itertuples(index=False, name=None))
            
            cursor = self.conn.cursor()
            cursor.executemany(query, data_to_insert)
            self.conn.commit()

            records_inserted = len(data_to_insert)

            # Log successful ingestion
            cursor.execute('''
                INSERT INTO ingestion_log (batch_id, records_inserted)
                VALUES (?, ?)
            ''', (batch_id, records_inserted))
            self.conn.commit()
            
            log_stage(self.logger, 'INSERT_FEATURES', 'SUCCESS',
                      inserted=records_inserted)
            
            return records_inserted
            
        except Exception as e:
            log_stage(self.logger, 'INSERT_FEATURES', 'FAILURE', error=str(e))
            self.logger.error(f"[{self.module_name}] SQL Error: {e}")
            raise