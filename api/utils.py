import yaml
import hashlib
from ml_pipeline.preprocess import DataPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def load_config(config_file='api\config.yaml'):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_hashed_key(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode()).hexdigest()


def verify_api_key(provided_key: str, hashed_key: str) -> bool:
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return provided_hash == hashed_key


def process(path: str):
    preprocessor = DataPreprocessor()
    vectorizer = TfidfVectorizer()

    df = pd.read_csv(path)
    df = preprocessor.preprocess(df)
    df['input_length'] = df['narrative'].str.len()
    df['tfidf_features'] = list(vectorizer.fit_transform(df['narrative']).toarray())
    return df
