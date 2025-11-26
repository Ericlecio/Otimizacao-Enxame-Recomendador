import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from preprocessamento import carregar_e_processar_dados
from pso import executar_pso
from fitness import decodificar_particula, calcular_perfil_usuario, MAPA_GRUPOS

USUARIO_ALVO = "User 1"
TAMANHO_ROTEIRO = 5       
LIMIAR_GOSTO = 0.5        
NUM_EXECUCOES = 20        
TAMANHOS_ENXAME = [20, 40, 80] 
DIRETORIO_SAIDA = "resultados_otimizados"

MAPA_NOMES = {
    'Category 1': 'Igrejas (Histórico)', 'Category 2': 'Resorts (Estadia)', 
    'Category 3': 'Praias (Natureza)', 'Category 4': 'Parques (Natureza)', 
    'Category 5': 'Teatros (Entretenimento)', 'Category 6': 'Museus (Histórico)',
    'Category 7': 'Shoppings (Entretenimento)', 'Category 8': 'Zoológicos (Natureza)', 
    'Category 9': 'Restaurantes (Gastronomia)', 'Category 10': 'Pubs/Bares (Gastronomia)',
    'Category 11': 'Serviços Locais (BemEstar)', 'Category 12': 'Pizzarias (Gastronomia)',
    'Category 13': 'Hotéis (Estadia)', 'Category 14': 'Sucos (Gastronomia)', 
    'Category 15': 'Galerias Arte (Histórico)', 'Category 16': 'Baladas (BemEstar)', 
    'Category 17': 'Piscinas (BemEstar)', 'Category 18': 'Academias (BemEstar)',
    'Category 19': 'Padarias (Gastronomia)', 'Category 20': 'Spas (BemEstar)', 
    'Category 21': 'Cafés (Gastronomia)', 'Category 22': 'Mirantes (Natureza)', 
    'Category 23': 'Monumentos (Histórico)', 'Category 24': 'Jardins (Natureza)'
}

def criar_diretorio(caminho):
    os.makedirs(caminho, exist_ok=True)

def criar_gabarito_real(df, id_usuario, limiar):
    notas = df.loc[id_usuario]
    return (notas >= limiar).astype(int)

def criar_predicao(posicao, num_cats, tamanho):
    y_pred = np.zeros(num_cats)
    idx = decodificar_particula(posicao)[:tamanho]
    y_pred[idx] = 1
    return y_pred

def mostrar_perfil_engenharia(pesos):
    print("\n--- ENGENHARIA DE FEATURES: PERFIL DO USUÁRIO ---")
    print(f"Pesos calculados automaticamente para {USUARIO_ALVO}:")
    df_p = pd.DataFrame(list(pesos.items()), columns=['Grupo', 'Peso (Afinidade)'])
    print(df_p.sort_values(by='Peso (Afinidade)', ascending=False))
    print("--------------------------------------------------\n")

def mostrar_roteiro_final(posicao, df):
    print("\n--- MELHOR ROTEIRO OTIMIZADO ---")
    idx = decodificar_particula(posicao)[:TAMANHO_ROTEIRO]
    cols = df.columns[idx]
    for i, c in enumerate(cols):
        print(f"{i+1}. {MAPA_NOMES.get(c, c)}")

def main():
    print(">>> INICIANDO OTIMIZAÇÃO COM ENGENHARIA DE FEATURES <<<")
    criar_diretorio(DIRETORIO_SAIDA)

    caminho_arquivo = "google_review_ratings.csv" 
    if not os.path.exists(caminho_arquivo):
        caminho_arquivo = "data/google_review_ratings.csv"
        
    df = carregar_e_processar_dados(caminho_arquivo)
    if df is None: return
    
    pesos_perfil = calcular_perfil_usuario(df, USUARIO_ALVO)
    mostrar_perfil_engenharia(pesos_perfil)
    
    y_real = criar_gabarito_real(df, USUARIO_ALVO, LIMIAR_GOSTO)
    tabela_resumo = []
    
    melhor_global_pos = None
    melhor_global_fit = -np.inf

    for n_particulas in TAMANHOS_ENXAME:
        print(f"\n> Executando Enxame: {n_particulas} partículas")
        
        tempos = []
        f1s = []
        fits = []
        
        for i in range(NUM_EXECUCOES):
            fit, pos, hist, tempo = executar_pso(
                df, USUARIO_ALVO, n_particulas, TAMANHO_ROTEIRO, pesos_perfil, verbose=False
            )
            
            y_pred = criar_predicao(pos, len(df.columns), TAMANHO_ROTEIRO)
            f1 = f1_score(y_real, y_pred, zero_division=0)
            
            tempos.append(tempo)
            f1s.append(f1)
            fits.append(fit)
            
            if fit > melhor_global_fit:
                melhor_global_fit = fit
                melhor_global_pos = pos

        media_tempo = np.mean(tempos)
        media_f1 = np.mean(f1s)
        
        print(f"  Média F1: {media_f1:.4f} | Tempo Médio Convergência: {media_tempo:.4f}s")
        
        tabela_resumo.append({
            "Enxame": n_particulas,
            "F1_Score": media_f1,
            "Tempo_Convergencia_Seg": media_tempo,
            "Melhor_Fitness": np.max(fits)
        })

    df_res = pd.DataFrame(tabela_resumo)
    df_res.to_csv(os.path.join(DIRETORIO_SAIDA, "analise_final.csv"), index=False)
    
    print("\n--- TABELA FINAL DE PERFORMANCE ---")
    print(df_res)
    
    if melhor_global_pos is not None:
        mostrar_roteiro_final(melhor_global_pos, df)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_res, 
        x='Enxame', 
        y='Tempo_Convergencia_Seg', 
        hue='Enxame',   
        palette='magma', 
        legend=False        
    )
    plt.title("Tempo de Convergência por Tamanho do Enxame")
    plt.ylabel("Segundos")
    plt.savefig(os.path.join(DIRETORIO_SAIDA, "grafico_tempo.png"))
    print(f"\nGráficos salvos em {DIRETORIO_SAIDA}")

if __name__ == "__main__":
    main()