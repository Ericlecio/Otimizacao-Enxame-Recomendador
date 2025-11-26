import numpy as np
import pandas as pd

MAPA_GRUPOS = {
    'Historico': [0, 5, 14, 22],       # Igrejas, Museus, Galerias, Monumentos
    'Natureza': [2, 3, 7, 21, 23],     # Praias, Parques, Zoo, Mirantes, Jardins
    'Gastronomia': [8, 9, 11, 13, 18, 20], # Restaurantes, Pubs, Burger, Sucos, Padarias, Cafés
    'Entretenimento': [4, 6, 15],      # Teatros, Shoppings, Baladas
    'Estadia': [1, 12],                # Resorts, Hoteis
    'BemEstar': [10, 16, 17, 19]       # Serviços Locais, Piscinas, Academias, Spas
}

def decodificar_particula(posicao):
    """Converte posição contínua em índices inteiros."""
    return np.argsort(posicao)

def calcular_perfil_usuario(df_completo, id_usuario):
    """
    FEATURE ENGINEERING AVANÇADA:
    Calcula a afinidade comparando o usuário com a POPULAÇÃO (Dataset Inteiro).
    
    Lógica: 
    Se o usuário dá nota 4.0 em algo que a média global é 2.0, ele tem ALTA afinidade.
    Se ele dá 4.0 em algo que a média global é 4.5, a afinidade é BAIXA/NORMAL.
    """
    try:
        notas_usuario = df_completo.loc[id_usuario]
    except KeyError:
        return None

    # 1. Calcular a média global de cada categoria (Dataset Inteiro)
    media_global_por_categoria = df_completo.mean()

    pesos_perfil = {}
    
    for nome_grupo, indices in MAPA_GRUPOS.items():
        # Notas do usuário para este grupo
        vals_user = notas_usuario.iloc[indices]
        
        # Média global para as categorias deste grupo
        vals_global = media_global_por_categoria.iloc[indices]
        
        # Calculamos a diferença entre o usuário e a média global
        # Ex: Usuário (0.8) - Global (0.4) = +0.4 (Afinidade Positiva)
        diferenca = vals_user.mean() - vals_global.mean()
        
        # Normalizamos isso para um peso positivo (base 1.0)
        # Se diferença for positiva, peso > 1.0 (Prioridade)
        # Se diferença for negativa, peso < 1.0 (Penalidade)
        peso_ajustado = 1.0 + diferenca
        
        # Garante que o peso nunca seja negativo ou zero
        pesos_perfil[nome_grupo] = max(0.1, peso_ajustado)
        
    return pesos_perfil

def calcular_fitness(posicao, df_completo, id_usuario, tamanho_roteiro, pesos_perfil_usuario):
    """
    Função Fitness Otimizada com Pesos Relativos.
    """
    
    indices_ordenados = decodificar_particula(posicao)
    indices_escolhidos = indices_ordenados[:tamanho_roteiro]
    
    notas_usuario = df_completo.loc[id_usuario]
    
    score_total = 0
    
    for idx in indices_escolhidos:
        nota_item = notas_usuario.iloc[idx]
        
        peso_grupo = 1.0 
        for grupo, indices_grupo in MAPA_GRUPOS.items():
            if idx in indices_grupo:
                peso_grupo = pesos_perfil_usuario.get(grupo, 1.0)
                break
        
        popularidade = df_completo.iloc[:, idx].mean()
        
        # Multiplicamos a nota pelo PESO RELATIVO.
        # Se o usuário gosta mais desse grupo do que a média das pessoas,
        # o 'peso_grupo' será alto, impulsionando esse item.
        score_item = (nota_item * peso_grupo) + (0.2 * popularidade)
        
        score_total += score_item

    return score_total