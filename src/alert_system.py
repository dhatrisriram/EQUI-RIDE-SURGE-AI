"""
EquiRide Surge AI - Forecast-Based Alert System (Standalone + Compatible)
Reads ONLY datasets/forecast_15min_predictions.csv for surge alerts.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure project paths are resolvable (Fixed typo in __file__)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.logging_config import setup_logging, log_stage
from src.utils import _resolve_project_path

# Load environment variables
load_dotenv()

class AlertSystem:
    # --- FIX 1: Corrected constructor name from _init_ to __init__ ---
    def __init__(self, config, logger=None):
        """Initialize Alert System"""
        self.config = config
        self.logger = logger or setup_logging()
        self.module_name = "AlertSystem"
        
        self.twilio_enabled = config.get('twilio', {}).get('enabled', False)
        self.alert_cooldown = config.get('alerts', {}).get('alert_cooldown_minutes', 30)
        self.last_alert_time = {}
        self.twilio_client = None

        if self.twilio_enabled:
            self._initialize_twilio()
        else:
            self.logger.info("Twilio alerts disabled in config.")

    def _initialize_twilio(self):
        """Initialize Twilio client"""
        try:
            from twilio.rest import Client
            
            # Use os.getenv as intended
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
            self.alert_phone = os.getenv('ALERT_PHONE_NUMBER') # Target phone number for the operator

            if not all([account_sid, auth_token, self.twilio_phone, self.alert_phone]):
                self.logger.warning("Twilio credentials missing or incomplete in .env. SMS disabled.")
                self.twilio_enabled = False
                return

            self.twilio_client = Client(account_sid, auth_token)
            self.logger.info("SUCCESS Twilio client initialized successfully.")

        except ImportError:
            self.logger.error("FAILED Twilio client initialization: 'twilio' library not found. SMS disabled.")
            self.twilio_enabled = False
        except Exception as e:
            self.logger.error(f"FAILED Twilio initialization failed: {str(e)}. SMS disabled.")
            self.twilio_enabled = False

    # ===================================================================
    # COOLDOWN CHECK
    # ===================================================================
    def check_alert_cooldown(self, area_name):
        """Checks if a specific area is still within the alert cooldown period."""
        if area_name not in self.last_alert_time:
            return True
        
        time_since_last = datetime.now() - self.last_alert_time[area_name]
        # Cooldown in seconds
        cooldown_duration = self.alert_cooldown * 60
        return time_since_last.total_seconds() > cooldown_duration

    # ===================================================================
    # SEND SMS
    # ===================================================================
    def send_sms_alert(self, message, area_name):
        """Send SMS via Twilio"""
        if not self.twilio_enabled:
            # Use logger.warning for high-visibility output when SMS is disabled
            self.logger.warning(f"[ALERT - SMS DISABLED] {area_name}: {message.splitlines()[1].strip()}")
            return False

        if not self.check_alert_cooldown(area_name):
            self.logger.info(f"Cooldown active for {area_name}. Skipping SMS.")
            return False

        try:
            msg = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=self.alert_phone
            )
            self.last_alert_time[area_name] = datetime.now()
            self.logger.info(f"SUCCESS SMS sent to {self.alert_phone}. SID: {msg.sid}")
            return True

        except Exception as e:
            self.logger.error(f"FAILED SMS send error for {area_name}: {str(e)}")
            return False

    # ===================================================================
    # FORECAST CSV LOADING
    # ===================================================================
    def load_forecast_data(self, csv_path="datasets/forecast_15min_predictions.csv"):
        """Loads the forecast CSV, ensuring the correct columns are present."""
        try:
            # Use the global resolver utility for path consistency
            full_path = _resolve_project_path(csv_path) 
            df = pd.read_csv(full_path)

            required = ["h3_index", "next_time", "pred_bookings_15min"]
            if not all(col in df.columns for col in required):
                raise ValueError("CSV must contain: h3_index, next_time, pred_bookings_15min")

            return df

        except Exception as e:
            self.logger.error(f"FAILED to load forecast CSV: {str(e)}")
            return pd.DataFrame()

    # ===================================================================
    # FORECAST-BASED SURGE DETECTION
    # ===================================================================
    def calculate_forecast_surges(self, df):
        """Calculates which zones are experiencing a predicted surge (2-sigma threshold)."""
        try:
            log_stage(self.logger, "CHECK_FORECAST_SURGE", "START")
            
            # --- Detection Logic: 2-Sigma Threshold ---
            # Use mean and standard deviation of predicted bookings across all zones
            mean = df["pred_bookings_15min"].mean()
            std = df["pred_bookings_15min"].std()

            # Set threshold to 2 standard deviations above the mean (A common anomaly/surge definition)
            threshold = mean + 2 * std 

            # Filter rows that exceed the surge prediction threshold
            surge_rows = df[df["pred_bookings_15min"] > threshold]

            alerts = []
            for _, row in surge_rows.iterrows():
                alerts.append({
                    "h3_index": row["h3_index"],
                    "area_name": row["h3_index"], # Use h3_index as area name for simplicity
                    "severity": self._calculate_severity(row["pred_bookings_15min"], mean, std),
                    "pred_bookings": row["pred_bookings_15min"],
                    "next_time": row["next_time"],
                    "timestamp": datetime.now().isoformat()
                })

            log_stage(self.logger, "CHECK_FORECAST_SURGE", "SUCCESS", alerts_found=len(alerts))
            return alerts

        except Exception as e:
            log_stage(self.logger, "CHECK_FORECAST_SURGE", "FAILURE", error=str(e))
            return []

    def _calculate_severity(self, val, mean, std):
        """Helper to assign severity based on distance from the mean."""
        if val > mean + 3 * std:
            return "CRITICAL"
        elif val > mean + 2.5 * std:
            return "HIGH"
        else:
            return "MEDIUM"

    # ===================================================================
    # MESSAGE FORMATTING
    # ===================================================================
    def format_forecast_alert_message(self, alert):
        """Formats a clean, concise message for SMS/console output."""
        msg = f"""
PREDICTED SURGE ALERT

H3 Zone: {alert['h3_index']}
Severity: {alert['severity']}
Predicted Bookings (next 15 min): {alert['pred_bookings']:.1f}
Forecast Time: {alert['next_time']}

Action: Deploy drivers from optimized repositioning plan.
"""
        return msg.strip()

    # ===================================================================
    # PROCESS FORECAST ALERTS (MAIN ENTRY POINT)
    # ===================================================================
    def process_forecast_alerts(self, csv_path="datasets/forecast_15min_predictions.csv"):
        """Main function to trigger alert calculation and dispatch."""
        try:
            df = self.load_forecast_data(csv_path)

            if df.empty:
                self.logger.info("No forecast data to process alerts.")
                return 0

            alerts = self.calculate_forecast_surges(df)

            if len(alerts) == 0:
                self.logger.info("No forecast surge detected above threshold.")
                return 0

            sent = 0
            for alert in alerts:
                msg = self.format_forecast_alert_message(alert)
                # Use send_sms_alert (which includes the cooldown check)
                if self.send_sms_alert(msg, alert["h3_index"]): 
                    sent += 1

            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "SUCCESS", alerts_sent=sent)
            return sent

        except Exception as e:
            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "FAILURE", error=str(e))
            return 0