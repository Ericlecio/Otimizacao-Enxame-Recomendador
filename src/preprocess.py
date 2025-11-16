import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path="data/google_review_ratings.csv"):
    """
    Carrega, limpa e normaliza o dataset de avaliações.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERRO: O ficheiro {path} não foi encontrado.")
        return None

    # Remove colunas que estejam completamente vazias
    df = df.dropna(axis=1, how='all')
    
    if 'User' in df.columns:
        df = df.set_index('User')
    else:
        print("ERRO: Coluna 'User' não encontrada no CSV.")
        return None

    # Força todas as colunas de avaliação para numérico
    # 'errors='coerce'' transforma textos inválidos (ex: '2\t2.') em NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preenche todos os valores ausentes (NaN) com 0
    df = df.fillna(0)

    # Normaliza todas as notas para a escala [0, 1]
    num_cols = df.columns 
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df