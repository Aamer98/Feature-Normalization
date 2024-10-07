import datetime
import time
import os

import pandas as pd

__all__ = ['SaveLog']

class SaveLog:
    """
    A class for saving training logs to a CSV file incrementally.

    Attributes:
        INCREMENTAL_UPDATE_TIME (int): Time interval in seconds for incremental updates.
        file_path (str): Path to the CSV file where logs are saved.
        data (dict): Dictionary to store log data.
        last_update_time (float): Timestamp of the last update.
    """

    INCREMENTAL_UPDATE_TIME = 0  # Time interval in seconds for incremental updates

    def __init__(self, directory, name):
        """
        Initializes the SaveLog instance.

        Args:
            directory (str): Directory where the log file will be saved.
            name (str): Base name for the log file.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{name}_{timestamp}.csv"
        self.file_path = os.path.join(directory, filename)
        self.data = {}
        self.last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record(self, step, value_dict):
        """
        Records a new entry in the log.

        Args:
            step (int): The current step or epoch.
            value_dict (dict): A dictionary of values to log.
        """
        self.data[step] = value_dict
        current_time = time.time()
        if current_time - self.last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self.last_update_time = current_time
            self.save()

    def save(self):
        """
        Saves the current log data to the CSV file.
        """
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.to_csv(self.file_path)
