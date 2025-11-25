import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def carregar_e_processar_dados(caminho="data/google_review_ratings.csv"):
    """
    Carrega o CSV, remove colunas vazias, trata erros numéricos 
    e normaliza os dados entre 0 e 1.
    """
    try:
        df = pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho} não encontrado.")
        return None
    df = df.dropna(axis=1, how='all')
    
    if 'User' in df.columns:
        df = df.set_index('User')
    else:
        print("Erro: Coluna 'User' não encontrada.")
        return None

    for coluna in df.columns:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    
    df = df.fillna(0)

    colunas_numericas = df.columns 
    escalador = MinMaxScaler()
    df[colunas_numericas] = escalador.fit_transform(df[colunas_numericas])

    return df