import numpy as np
import time
from pyswarms.single.global_best import GlobalBestPSO
from fitness import calcular_fitness

def executar_pso(df, id_usuario, num_particulas, tamanho_roteiro, pesos_perfil, verbose=True):
    """
    Executa o PSO e mede o tempo de convergência.
    """
    
    num_dimensoes = len(df.columns)
    inicio_tempo = time.time()

    def funcao_objetivo(particulas):
        scores = []
        for p in particulas:
            score = calcular_fitness(p, df, id_usuario, tamanho_roteiro, pesos_perfil)
            scores.append(-score)
        return np.array(scores)

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    optimizer = GlobalBestPSO(
        n_particles=num_particulas,
        dimensions=num_dimensoes,
        options=options
    )

    best_cost, best_pos = optimizer.optimize(
        funcao_objetivo, 
        iters=150,
        verbose=verbose
    )
    
    fim_tempo = time.time()
    tempo_convergencia = fim_tempo - inicio_tempo

    return -best_cost, best_pos, optimizer.cost_history, tempo_convergencia