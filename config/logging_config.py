import logging
import sys
import os

# Define the log stage utility used extensively in data modules
def log_stage(logger, stage, status, **kwargs):
    """Logs the status of a specific processing stage with key metrics."""
    details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{stage:<20}] {status:<8} | {details}")

def setup_logging(log_file="logs/pipeline.log", log_level="INFO"):
    """
    Sets up the project-wide logging configuration based on config.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Resolve log file path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_log_file = os.path.join(project_root, log_file)
    os.makedirs(os.path.dirname(full_log_file), exist_ok=True)
    
    # 1. Clear any existing handlers to prevent duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # 2. Configure the logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(full_log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("EquiRidePipeline")
    logger.setLevel(level)
    return logger