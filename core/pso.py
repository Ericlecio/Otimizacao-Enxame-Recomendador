import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from core.fitness import calcular_fitness, decodificar_particula
from core.perfil import calcular_perfil_usuario
from core.categorias import CATEGORIAS

def gerar_roteiro_pso(df, user_id, tamanho=5, n_particulas=40, categorias_bloqueadas=None):
    if categorias_bloqueadas is None:
        categorias_bloqueadas = []

    dim = len(df.columns)
    pesos = calcular_perfil_usuario(df, user_id)

    colunas_validas = [
        i for i, c in enumerate(df.columns)
        if c not in categorias_bloqueadas
    ]

    def objective(particles):
        scores = []
        for p in particles:
            p_filtrado = p.copy()
            for i in range(dim):
                if i not in colunas_validas:
                    p_filtrado[i] = -999  # bloqueia categoria
            scores.append(
                -calcular_fitness(p_filtrado, df, user_id, tamanho, pesos)
            )
        return np.array(scores)

    optimizer = GlobalBestPSO(
        n_particles=n_particulas,
        dimensions=dim,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    )

    cost, pos = optimizer.optimize(objective, iters=100, verbose=False)

    indices = decodificar_particula(pos)[:tamanho]

    roteiro = []
    for idx in indices:
        col = df.columns[idx]
        roteiro.append({
            "categoria": CATEGORIAS.get(col, col),
            "nota_usuario": float(df.loc[user_id, col]),
            "popularidade": float(df[col].mean()),
            "motivo": "Alta afinidade + popularidade"
        })

    return {
        "usuario": user_id,
        "score": float(-cost),
        "roteiro": roteiro
    }
