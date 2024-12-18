import os
import sys
import logging


logging_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir="logs"
log_filepath=os.path.join(log_dir,"logging.log")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.INFO, # Sets the minimum severity level to INFO
    format=logging_str, # Specifies the log message format

    # handlers defines the destinations for log messages:
    handlers=[ 
        logging.FileHandler(log_filepath), # Saves logs to the file specified by log_filepath
        logging.StreamHandler(sys.stdout) # displays log in terminal
    ]
)

logger=logging.getLogger("wine_quality_ds_logger") # Creates or retrieves a logger with the name
# The custom logger "wine_quality_ds_logger" can be used throughout the codebase to generate logs using logger.info, logger.error, etc.
