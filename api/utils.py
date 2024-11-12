import yaml
import hashlib
import pandas as pd
from ml_pipeline.preprocess import DataPreprocessor



def load_config(config_file='api/config.yaml'):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)



def get_hashed_key(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode()).hexdigest()



def verify_api_key(provided_key: str, hashed_key: str) -> bool:
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return provided_hash == hashed_key



def load_in(path: str):
    preprocessor = DataPreprocessor()
    df = pd.read_csv(path)
    df = preprocessor.preprocess(df)
    df['input_length'] = df['narrative'].str.len()
    df['word_count'] = df['narrative'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['narrative'].apply(lambda x: len(str(x)))
    return df



def vectorise(df, vectorizer):
    vectors = vectorizer.transform(df['narrative']).toarray()
    vector_df = pd.DataFrame(vectors, columns=vectorizer.get_feature_names_out())
    vector_combined_df = pd.concat([vector_df, df[['input_length', 'product']].reset_index(drop=True)], axis=1)
    return vector_combined_df
