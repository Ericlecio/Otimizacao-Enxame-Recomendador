import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from src.fitness import fitness

def run_pso_recommender(df, user_id, n_particles, itinerary_size, verbose=True):
    """
    Executa o otimizador PSO com um número de partículas configurável.
    """
    
    num_dimensions = len(df.columns) # Uma dimensão para cada categoria

    def objective_function(particles):
        """
        Função de custo que o pyswarms irá minimizar.
        'particles' é um array com todas as partículas do enxame.
        """
        scores = []
        for p in particles:
            score = fitness(p, df, user_id, itinerary_size)
            # Invertemos o sinal, pois o PSO minimiza, e nós queremos maximizar
            scores.append(-score) 
        return np.array(scores)

    # Hiperparâmetros do PSO
    options = {
        'c1': 1.5,  # Fator cognitivo (individual)
        'c2': 1.5,  # Fator social (coletivo)
        'w': 0.7   # Inércia
    }

    # Inicializa o otimizador
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=num_dimensions,
        options=options
    )

    # Executa a otimização
    best_cost, best_pos = optimizer.optimize(
        objective_function, 
        iters=200, 
        verbose=verbose
    )

    # Retorna o melhor custo (negativo), a melhor posição e o histórico
    return best_cost, best_pos, optimizer.cost_history