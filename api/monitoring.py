import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from api.utils import process
import csv
import os



class DataMonitor:
    def __init__(self, training_data: str, output_csv: str):
        self.buffer = []
        self.buffer_size = 3
        self.request_count = 0
        self.training_data = training_data
        self.output_csv = output_csv
        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['narrative', 'product'])
    

    def collect_data(self, text, prediction):
        self.request_count += 1
        self.buffer.append([text, prediction])
        
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()
            self.analyze_data_drift()


    def flush_buffer(self):
        file_exists = os.path.isfile(self.output_csv)  # Check if file exists
        with open(self.output_csv, mode='a', newline='') as file:  # Use 'a' for append mode
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['narrative', 'product'])  # Write header if file didn't exist
            writer.writerows(self.buffer)
        self.buffer = []


    def analyze_data_drift(self):
        df_current = process(self.output_csv)
        df_reference = process(self.training_data)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_reference, current_data=df_current)

        report_filename = "data_drift_report.html"
        report.save_html(report_filename)
        mlflow.log_artifact(report_filename)
