import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.data_handler import RealTimeDataHandler
from datetime import datetime, timedelta, timezone
import time
import pandas as pd



class RealtimeDealer:
    def __init__(self, data_handler):
        self.data_handler = data_handler  # DataHandler is responsible for fetching and managing data
        self.strategy = None
        self.is_running = False

    def start(self):
        self.is_running = True
        closure = False
        next_fetch_time,last_fetch_time = self.data_handler.pre_run_data()

        while self.is_running:
            self.data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
            
            now = datetime.now(timezone.utc)
            next_fetch_time = self.calculate_next_grid(now)
            self.monitor_system_health()  # High-level system checks
            sleep_duration = (next_fetch_time - now).total_seconds()
            self.data_logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
            time.sleep(sleep_duration)

    def monitor_system_health(self):
        # Check if the data handler is functioning correctly
        if not self.data_handler.is_healthy():
            print("Data Handler issue detected. Taking action...")
            self.restart_system()
    def check_trading_signals(self):
        # Check for trading signals
        pass
    def restart_system(self):
        print("Restarting system...")
        self.stop()
        self.start()

    def stop(self):
        self.is_running = False
        # self.data_handler.stop_fetching()  # Stop fetching data
