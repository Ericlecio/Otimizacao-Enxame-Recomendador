import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocessamento import carregar_e_processar_dados
from pso import executar_pso
from fitness import decodificar_particula

USUARIO_ALVO = "User 55"
TAMANHO_ROTEIRO = 5       
LIMIAR_GOSTO = 0.5        
NUM_EXECUCOES = 30        
TAMANHOS_ENXAME = [20, 40, 80] 
FATOR_PREFERENCIA = 0.9   
DIRETORIO_SAIDA = "resultados"

MAPA_CATEGORIAS = {
    'Category 1': 'Igrejas', 'Category 2': 'Resorts', 'Category 3': 'Praias',
    'Category 4': 'Parques', 'Category 5': 'Teatros', 'Category 6': 'Museus',
    'Category 7': 'Shoppings', 'Category 8': 'Zoológicos', 'Category 9': 'Restaurantes',
    'Category 10': 'Bares e Pubs', 'Category 11': 'Serviços Locais', 'Category 12': 'Pizzarias',
    'Category 13': 'Hotéis', 'Category 14': 'Casas de Suco', 'Category 15': 'Galerias de Arte',
    'Category 16': 'Baladas', 'Category 17': 'Piscinas', 'Category 18': 'Academias',
    'Category 19': 'Padarias', 'Category 20': 'Spas', 'Category 21': 'Cafés',
    'Category 22': 'Mirantes', 'Category 23': 'Monumentos', 'Category 24': 'Jardins'
}

def criar_diretorio(caminho):
    os.makedirs(caminho, exist_ok=True)

def criar_gabarito_real(df, id_usuario, limiar):
    notas_usuario = df.loc[id_usuario]
    y_real = (notas_usuario >= limiar).astype(int)
    return y_real

def criar_predicao(posicao_particula, num_categorias, tamanho_roteiro):
    y_pred = np.zeros(num_categorias)
    indices_ordenados = decodificar_particula(posicao_particula)
    indices_recomendados = indices_ordenados[:tamanho_roteiro]
    y_pred[indices_recomendados] = 1
    return y_pred

def salvar_graficos_individuais(output_dir, tamanho_enxame, lista_f1, historico_convergencia):
    sns.set(style="whitegrid")
    
    # 1. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=lista_f1, palette="pastel")
    sns.swarmplot(data=lista_f1, color=".25")
    plt.title(f'Estabilidade do F1-Score (Enxame={tamanho_enxame})', fontsize=16)
    plt.ylabel('F1-Score')
    plt.savefig(os.path.join(output_dir, 'grafico_boxplot_f1.png'))
    plt.close()

    # 2. Convergência
    plt.figure(figsize=(10, 6))
    plt.plot(historico_convergencia, linewidth=2, color='blue')
    plt.title(f'Convergência (Melhor Execução, Enxame={tamanho_enxame})', fontsize=16)
    plt.xlabel('Iterações')
    plt.ylabel('Fitness')
    plt.savefig(os.path.join(output_dir, 'grafico_convergencia.png'))
    plt.close()

def gerar_grafico_comparativo(df_resumo, output_dir):
    """
    Gera um gráfico de barras comparando a média do F1-Score
    entre os diferentes tamanhos de enxame (20, 40, 80).
    """
    plt.figure(figsize=(10, 6))
    
    x = [str(t) for t in df_resumo['Tamanho_Enxame']] 
    y = df_resumo['Media_F1']
    erro = df_resumo['Desvio_Padrao_F1']
    
    barras = plt.bar(x, y, yerr=erro, capsize=5, color=['lightblue', 'steelblue', 'darkblue'], edgecolor='black')
    
    plt.title('Comparação Final: Impacto do Tamanho do Enxame', fontsize=16)
    plt.xlabel('Número de Partículas', fontsize=12)
    plt.ylabel('Média do F1-Score', fontsize=12)
    plt.ylim(0, 1.1) 
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.bar_label(barras, fmt='%.3f', padding=3)
    
    caminho_arquivo = os.path.join(output_dir, 'grafico_final_comparativo.png')
    plt.savefig(caminho_arquivo)
    plt.close()
    print(f"\n[OK] Gráfico Comparativo salvo em: {caminho_arquivo}")

def mostrar_roteiro_final(melhor_posicao_global, df, tamanho_roteiro):
    print("\n" + "="*50)
    print(f" MELHOR ROTEIRO ENCONTRADO (TOP {tamanho_roteiro}) ")
    print("="*50)
    indices_ordenados = decodificar_particula(melhor_posicao_global)
    indices_escolhidos = indices_ordenados[:tamanho_roteiro]
    nomes_colunas = df.columns[indices_escolhidos]
    for i, col_name in enumerate(nomes_colunas):
        nome_real = MAPA_CATEGORIAS.get(col_name, col_name)
        print(f"{i+1}º Lugar: {nome_real}")
    print("="*50 + "\n")

def main():
    print("--- INICIANDO SISTEMA DE RECOMENDAÇÃO (PSO) ---")
    criar_diretorio(DIRETORIO_SAIDA)

    df = carregar_e_processar_dados("data/google_review_ratings.csv")
    if df is None: return

    num_categorias = len(df.columns)
    y_real = criar_gabarito_real(df, USUARIO_ALVO, LIMIAR_GOSTO)
    
    tabela_resumo = []
    
    melhor_posicao_de_todas = None
    melhor_fitness_de_todos = -np.inf

    for tamanho_enxame in TAMANHOS_ENXAME:
        print(f"\n>>> Testando Enxame com {tamanho_enxame} Partículas...")
        
        scores_f1 = []
        melhor_historico_local = []
        melhor_fitness_local = -np.inf

        for i in range(NUM_EXECUCOES):
            fitness, melhor_pos, historico = executar_pso(
                df, USUARIO_ALVO, tamanho_enxame, TAMANHO_ROTEIRO, FATOR_PREFERENCIA, verbose=False
            )

            y_pred = criar_predicao(melhor_pos, num_categorias, TAMANHO_ROTEIRO)
            f1 = f1_score(y_real, y_pred, zero_division=0)
            scores_f1.append(f1)

            if fitness > melhor_fitness_local:
                melhor_fitness_local = fitness
                melhor_historico_local = historico
            
            if fitness > melhor_fitness_de_todos:
                melhor_fitness_de_todos = fitness
                melhor_posicao_de_todas = melhor_pos
            
            if (i+1) % 10 == 0: 
                print(f"   -> Execução {i+1}/{NUM_EXECUCOES} concluída.")

        dir_resultado_atual = os.path.join(DIRETORIO_SAIDA, f"enxame_{tamanho_enxame}")
        criar_diretorio(dir_resultado_atual)
        salvar_graficos_individuais(dir_resultado_atual, tamanho_enxame, scores_f1, melhor_historico_local)

        tabela_resumo.append({
            "Tamanho_Enxame": tamanho_enxame,
            "Media_F1": np.mean(scores_f1),
            "Desvio_Padrao_F1": np.std(scores_f1),
            "Melhor_Fitness": melhor_fitness_local
        })

    df_resumo = pd.DataFrame(tabela_resumo)
    df_resumo.to_csv(os.path.join(DIRETORIO_SAIDA, "resumo_final.csv"), index=False)
    
    print("\n--- RESUMO ESTATÍSTICO ---")
    print(df_resumo)
    
    if melhor_posicao_de_todas is not None:
        mostrar_roteiro_final(melhor_posicao_de_todas, df, TAMANHO_ROTEIRO)
    
    gerar_grafico_comparativo(df_resumo, DIRETORIO_SAIDA)

if __name__ == "__main__":
    main()