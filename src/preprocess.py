import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path="data/google_review_ratings.csv"):
    """Carrega, limpa (imputação com 0) e aplica a Normalização Min-Max."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None

    df = df.dropna(axis=1, how='all')
    
    if 'User' in df.columns:
        df = df.set_index('User')
    else:
        return None

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    num_cols = df.columns 
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df