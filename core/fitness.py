import numpy as np
from core.perfil import MAPA_GRUPOS

def decodificar_particula(posicao):
    return np.argsort(posicao)

def calcular_fitness(posicao, df, user_id, tamanho, pesos):
    indices = decodificar_particula(posicao)[:tamanho]
    notas_user = df.loc[user_id]

    score = 0.0
    for idx in indices:
        peso = 1.0
        for grupo, idxs in MAPA_GRUPOS.items():
            if idx in idxs:
                peso = pesos.get(grupo, 1.0)
                break

        popularidade = df.iloc[:, idx].mean()
        score += (notas_user.iloc[idx] * peso) + (0.2 * popularidade)

    return score
