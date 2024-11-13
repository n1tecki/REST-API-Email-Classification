import pandas as pd
import re


class DataPreprocessor:
    def __init__(self):
        pass


    def nan_removal(self, data, columns):
        return data.dropna(subset=columns)


    def remove_duplicates(self, data):
        return data.drop_duplicates()


    def normalize_text(self, data, column):
        data[column] = data[column].str.lower()
        return data


    def remove_special_characters(self, data, column):
        data[column] = data[column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        return data


    def preprocess(self, data):
        data = self.nan_removal(data, ["narrative", "product"])
        data = self.remove_duplicates(data)
        data = self.normalize_text(data, "narrative")
        data = self.remove_special_characters(data, "narrative")
        return data
