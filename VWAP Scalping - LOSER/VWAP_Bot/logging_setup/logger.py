import logging
import sys
from logging_setup.refresh_console_handler import RefreshingConsoleHandler
from config.config import Config

def setup_logger():
    config = Config()
    log_config = config.get('logging')
    
    logger = logging.getLogger('bot')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO').upper()))
    
    # File handler
    file_handler = logging.FileHandler(log_config.get('file', 'bot.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Refreshed logger for dynamic console updates
    refresh_logger = logging.getLogger('bot.refresh')
    refresh_logger.setLevel(logging.INFO)
    
    refresh_console_handler = RefreshingConsoleHandler(sys.stdout)
    refresh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    refresh_console_handler.setFormatter(refresh_formatter)
    refresh_logger.addHandler(refresh_console_handler)
    refresh_logger.propagate = False
    
    return logger, refresh_logger

logger, refresh_logger = setup_logger()
