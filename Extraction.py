import pandas as pd 

#extract data
def extract_csv(filepath):
    return pd.read_csv(filepath)
