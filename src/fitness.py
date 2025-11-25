import numpy as np
import pandas as pd

PESOS = {
    "nota_usuario": 0.6,        # Peso Principal: O quanto o usuário gostou dessas categorias
    "fator_preferencia": 0.3,   # Peso: Preferência subjetiva (reforço)
    "popularidade_geral": 0.1   # Peso: O quanto essas categorias são bem avaliadas por todos
}

def decodificar_particula(posicao):
    """
    Converte as posições contínuas do PSO em índices ordenados (ranking de categorias).
    Retorna os índices das colunas que a partícula escolheu.
    """
    return np.argsort(posicao)

def calcular_fitness(posicao, df_completo, id_usuario, tamanho_roteiro, fator_preferencia_usuario):
    """
    Calcula a qualidade (score) da recomendação.
    Objetivo: Maximizar este valor.
    """
    
    # 1. Obter as notas do usuário alvo
    try:
        notas_usuario = df_completo.loc[id_usuario]
    except KeyError:
        return -np.inf 

    # 2. Decodificar quais categorias a partícula escolheu
    indices_ordenados = decodificar_particula(posicao)
    indices_escolhidos = indices_ordenados[:tamanho_roteiro]

    # A. Nota de Satisfação (Média das notas que o usuário deu para as categorias escolhidas)
    # Seleciona as colunas baseadas nos índices escolhidos
    score_nota_usuario = notas_usuario.iloc[indices_escolhidos].mean()

    # B. Popularidade Geral (Média da nota dessas categorias para TODOS os usuários)
    # df_completo.iloc[:, indices] pega todas as linhas das colunas escolhidas
    score_popularidade = df_completo.iloc[:, indices_escolhidos].mean().mean()
    
    # C. Fator Subjetivo (Reforço baseado na configuração do sistema)
    score_subjetivo = fator_preferencia_usuario * score_nota_usuario
    
    score_final = (
        PESOS['nota_usuario'] * score_nota_usuario + 
        PESOS['fator_preferencia'] * score_subjetivo +
        PESOS['popularidade_geral'] * score_popularidade
    )

    return score_final