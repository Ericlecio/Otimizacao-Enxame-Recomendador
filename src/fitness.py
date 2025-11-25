import numpy as np
import pandas as pd

PESOS_DE_FATOR = {
    "ratings": 0.5,           # Peso: Nota do usuário (Satisfação)
    "preferencia_usuario": 0.4, # Peso: Preferência subjetiva
    "popularidade_geral": 0.1,  # Peso: Popularidade (Simulada por Category 7)
    "custo_inverso": 0.5        # Peso: Custo (Simulado por Category 1)
}


def decode_particle(position):
    """Converte o vetor do PSO em uma ordem de índices (itinerário)."""
    return np.argsort(position)

def fitness(position, df, user_id, itinerary_size, user_preference_score):
    """
    Calcula o "fitness" (qualidade da solução) usando a fórmula PONDERADA.
    O objetivo é maximizar este score.
    """
    
    try:
        user_ratings = df.loc[user_id]
    except KeyError:
        return -np.inf 

    ordered_indices = decode_particle(position)
    itinerary_category_indices = ordered_indices[:itinerary_size]

    
    # 1. Nota de Preferência (Satisfação)
    ratings_score = user_ratings.iloc[itinerary_category_indices].mean()

    # 2. Custo (Usamos 1.0 - Custo para MINIMIZAR o preço)
    custo_score = 1.0 - df.iloc[itinerary_category_indices]['Category 1'].mean()

    # 3. Popularidade Geral (Simulada pela Category 7)
    popularidade_score = df.iloc[itinerary_category_indices]['Category 7'].mean()
    
    # 4. Fator Subjetivo
    preferencia_subjetiva = user_preference_score * ratings_score
    
    score = (
        PESOS_DE_FATOR['ratings'] * ratings_score + 
        PESOS_DE_FATOR['preferencia_usuario'] * preferencia_subjetiva +
        PESOS_DE_FATOR['popularidade_geral'] * popularidade_score +
        PESOS_DE_FATOR['custo_inverso'] * custo_score
    )

    return score