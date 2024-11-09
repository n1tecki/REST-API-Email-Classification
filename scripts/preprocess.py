import pandas as pd

def nan_removal(data):
    #Removing empty values
    data = data.dropna(subset=["narrative", "product"])

    return data

# Some further preprocessing
