import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def carregar_dados(caminho):
    df = pd.read_csv(caminho)

    if 'User' in df.columns:
        df = df.set_index('User')

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df
