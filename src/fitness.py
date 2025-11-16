import numpy as np

def decode_particle(position):
    """
    Converte o vetor de posição contínua do PSO (ex: [0.5, 0.1, 0.9])
    numa ordem de índices (ex: [1, 0, 2]), que representa o itinerário.
    """
    # argsort() retorna os índices que ordenariam o array
    return np.argsort(position)

def fitness(position, df, user_id, itinerary_size):
    """
    Calcula o "fitness" (qualidade) de uma partícula (itinerário)
    para um utilizador específico.
    """
    try:
        # 1. Obtém a linha de notas do utilizador (já normalizadas)
        user_ratings = df.loc[user_id]
    
    except KeyError:
        print(f"ERRO: O utilizador '{user_id}' não foi encontrado.")
        return -np.inf 

    # 2. Converte a posição da partícula na ordem das categorias
    ordered_indices = decode_particle(position)
    
    # 3. Seleciona apenas as N primeiras categorias (tamanho do itinerário)
    itinerary_category_indices = ordered_indices[:itinerary_size]

    # 4. Calcula o score final somando as notas das categorias selecionadas
    score = user_ratings.iloc[itinerary_category_indices].sum()

    return score