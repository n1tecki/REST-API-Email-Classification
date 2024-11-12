import os
import csv
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from sklearn.feature_extraction.text import CountVectorizer
from api.utils import load_in, vectorise, load_config


class DataMonitor:
    def __init__(self):
        self.config = load_config()
        self.buffer = []
        self.buffer_size = 3
        self.request_count = 0
        self.training_data_path = self.config["data"]["data_path"]
        self.output_csv_path = self.config["data"]["log_path"]
        self.vectorizer = CountVectorizer(max_features=100, min_df=1, max_df=1.0)
        self.df_reference = self._initialize_reference_data()
        self._initialize_csv_logging()



    def _initialize_reference_data(self):
        df_reference = load_in(self.training_data_path)
        self.vectorizer.fit(df_reference['narrative'].tolist())
        return vectorise(df_reference, self.vectorizer)



    def _initialize_csv_logging(self):
        if not os.path.isfile(self.output_csv_path):
            with open(self.output_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['narrative', 'product'])



    def collect_data(self, text: str, prediction: str):
        self.request_count += 1
        self.buffer.append([text, prediction])
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
            self._analyze_data_drift()



    def _flush_buffer(self):
        with open(self.output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.buffer)
        self.buffer = []



    def _analyze_data_drift(self):
        df_current = load_in(self.output_csv_path)
        df_current = vectorise(df_current, self.vectorizer)

        column_mapping = ColumnMapping()
        column_mapping.numerical_features = self.config["monitoring"]["data_drift"]["numerical_features"]
        column_mapping.categorical_features = self.config["monitoring"]["data_drift"]["categorical_features"]
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.df_reference, current_data=df_current, column_mapping=column_mapping)

        report_filename = "data_drift_report.html"
        report.save_html(report_filename)
        with mlflow.start_run(run_name="Data Drift Analysis") as run:
            mlflow.log_artifact(report_filename, artifact_path="data_drift_reports")
