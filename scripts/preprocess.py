import pandas as pd
import re

class DataPreprocessor:
    def __init__(self):
        pass

    def nan_removal(self, data, columns):
        """Removes rows with NaN values in specified columns."""
        return data.dropna(subset=columns)

    def remove_duplicates(self, data):
        """Removes duplicate rows."""
        return data.drop_duplicates()

    def normalize_text(self, data, column):
        """Converts text to lowercase."""
        data[column] = data[column].str.lower().str
        return data

    def remove_special_characters(self, data, column):
        """Removes special characters from a text column."""
        data[column] = data[column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        return data

    def preprocess(self, data):
        """Applies a sequence of preprocessing steps."""
        data = self.nan_removal(data, ["narrative", "product"])
        data = self.remove_duplicates(data)
        data = self.normalize_text(data, "narrative")
        data = self.remove_special_characters(data, "narrative")
        return data
