import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from src.fitness import fitness

def run_pso_recommender(df, user_id, n_particles, itinerary_size, user_preference_score, verbose=True):
    """
    Executa o otimizador PSO, passando o fator de preferência para o cálculo do fitness.
    """
    
    num_dimensions = len(df.columns)

    def objective_function(particles):
        """Função de custo que o pyswarms minimiza (Fitness * -1)."""
        scores = []
        for p in particles:
            score = fitness(p, df, user_id, itinerary_size, user_preference_score)
            scores.append(-score) 
        return np.array(scores)

    options = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.7
    }

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=num_dimensions,
        options=options
    )

    best_cost, best_pos = optimizer.optimize(
        objective_function, 
        iters=200, 
        verbose=verbose
    )

    return best_cost, best_pos, optimizer.cost_history