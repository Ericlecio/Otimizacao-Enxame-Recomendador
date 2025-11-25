import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from fitness import calcular_fitness

def executar_pso(df, id_usuario, num_particulas, tamanho_roteiro, fator_preferencia, verbose=True):
    """
    Executa o otimizador PSO.
    Nome da função ajustado para bater com o main.py (executar_pso).
    """
    
    num_dimensoes = len(df.columns)

    def funcao_objetivo(particulas):
        """Função de custo que o pyswarms minimiza (Fitness * -1)."""
        scores = []
        for p in particulas:
            score = calcular_fitness(p, df, id_usuario, tamanho_roteiro, fator_preferencia)
            scores.append(-score) 
        return np.array(scores)

    options = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.7
    }

    optimizer = GlobalBestPSO(
        n_particles=num_particulas,
        dimensions=num_dimensoes,
        options=options
    )

    best_cost, best_pos = optimizer.optimize(
        funcao_objetivo, 
        iters=200, 
        verbose=verbose
    )

    return -best_cost, best_pos, optimizer.cost_history