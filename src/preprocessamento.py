import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def carregar_e_processar_dados(caminho="data/google_review_ratings.csv"):
    """
    Carrega o CSV, limpa dados sujos, remove colunas inúteis 
    e normaliza as avaliações entre 0 e 1.
    """
    try:
        df = pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho} não encontrado.")
        return None

    # Remove coluna 'Unnamed: 25' e linhas com muitos nulos se houver
    df = df.dropna(axis=1, how='all')
    
    if 'User' in df.columns:
        df = df.set_index('User')
    elif 'Unique user id' in df.columns: 
        df = df.set_index('Unique user id')
    
    # Força conversão para numérico (trata erros como NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preenche valores faltantes com 0 (assumindo que não avaliou)
    df = df.fillna(0)

    # Normalização Min-Max (0 a 1)
    colunas_numericas = df.columns 
    escalador = MinMaxScaler()
    df[colunas_numericas] = escalador.fit_transform(df[colunas_numericas])

    return df