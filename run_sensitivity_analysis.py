import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from src.preprocess import load_and_preprocess
from src.pso_recommender import run_pso_recommender
from src.fitness import decode_particle

TARGET_USER = "User 55"
ITINERARY_SIZE = 5
LIKE_THRESHOLD = 0.5
N_EXECUTIONS = 30
SWARM_SIZES_TO_TEST = [20, 40, 80]

USER_PREFERENCE_FACTOR = 0.9 

OUTPUT_DIR = "resultados"
EDA_DIR = os.path.join(OUTPUT_DIR, "1_analise_exploratoria")
SENSITIVITY_DIR = os.path.join(OUTPUT_DIR, "2_analise_sensibilidade")

def criar_diretorio_com_readme(path, descricao):
    os.makedirs(path, exist_ok=True)
    readme_path = os.path.join(path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(descricao)

def create_y_true(df, user_id, threshold):
    user_ratings = df.loc[user_id]
    y_true = (user_ratings >= threshold).astype(int)
    return y_true

def create_y_pred(best_pos, num_categories, itinerary_size):
    y_pred = np.zeros(num_categories)
    ordered_indices = decode_particle(best_pos)
    recommended_indices = ordered_indices[:itinerary_size]
    y_pred[recommended_indices] = 1
    return y_pred

def salvar_resultados_detalhados(output_dir, swarm_size, f1_scores, fitness_scores, best_history):
    """Salva a tabela de resultados brutos e os 3 gráficos detalhados de uma execução."""
    
    results_data = {
        'Execucao': np.arange(1, len(f1_scores) + 1),
        'Fitness_Score': fitness_scores,
        'F1_Score': f1_scores
    }
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(os.path.join(output_dir, 'tabela_resultados_brutos.csv'), index=False, float_format='%.4f')

    sns.set(style="whitegrid")
    
    # 1. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=f1_scores, palette="pastel"); sns.swarmplot(data=f1_scores, color=".25")
    plt.title(f'Estabilidade do F1-Score (Enxame={swarm_size}, {len(f1_scores)} Execuções)', fontsize=16)
    plt.ylabel('F1-Score', fontsize=12); plt.xlabel('PSO', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'grafico_boxplot_f1_score.png')); plt.close()

    # 2. Histograma
    plt.figure(figsize=(10, 6))
    f1_scores_str = [f"{score:.2f}" for score in f1_scores]
    sns.countplot(x=f1_scores_str, palette="coolwarm", order=sorted(list(set(f1_scores_str))))
    plt.title(f'Contagem de Resultados F1-Score (Enxame={swarm_size})', fontsize=16)
    plt.xlabel('F1-Score Obtido', fontsize=12); plt.ylabel('Número de Execuções (Contagem)', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'grafico_histograma_f1_scores.png')); plt.close()

    # 3. Curva de Convergência
    plt.figure(figsize=(10, 6))
    plt.plot([-c for c in best_history], linewidth=2, color='blue')
    plt.title(f'Curva de Convergência (Melhor Execução, Enxame={swarm_size})', fontsize=16)
    plt.xlabel('Iteração', fontsize=12); plt.ylabel('Melhor Fitness Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'grafico_convergencia.png')); plt.close()
    
    print(f"Gráficos e tabela para Enxame={swarm_size} salvos em '{output_dir}'.")


def executar_analise_exploratoria(output_dir):
    """Executa a Análise Exploratória de Dados (EDA)."""
    print("--- 1. Iniciando Análise Exploratória de Dados (EDA) ---")
    
    df_raw = pd.read_csv("data/google_review_ratings.csv")
    df_raw = df_raw.dropna(axis=1, how='all')
    if 'User' in df_raw.columns:
        df_raw = df_raw.set_index('User')
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw = df_raw.fillna(0)
    
    estatisticas = df_raw.describe().T
    estatisticas.to_csv(os.path.join(output_dir, 'tabela_estatisticas_descritivas.csv'))
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    mean_ratings = df_raw.mean().sort_values(ascending=False)
    sns.barplot(x=mean_ratings.values, y=mean_ratings.index, palette="viridis")
    plt.title('Popularidade Média por Categoria (Todos Utilizadores)', fontsize=16)
    plt.xlabel('Nota Média (0-5)', fontsize=12)
    plt.ylabel('Categoria', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grafico_popularidade_categorias.png')); plt.close()

    all_ratings = df_raw.values.flatten()
    all_ratings_sem_zeros = all_ratings[all_ratings > 0] 
    plt.figure(figsize=(10, 6))
    sns.histplot(all_ratings_sem_zeros, bins=20, kde=True)
    plt.title('Distribuição Geral das Notas (Excluindo Zeros)', fontsize=16)
    plt.xlabel('Nota', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'grafico_distribuicao_notas.png')); plt.close()
    
    print("--- Análise EDA Concluída ---")


def executar_analise_sensibilidade(output_dir):
    """Executa a Análise de Sensibilidade do PSO para diferentes tamanhos de enxame."""
    print("\n--- 2. Iniciando Análise de Sensibilidade (Tamanho do Enxame) ---")
    
    df = load_and_preprocess("data/google_review_ratings.csv")
    if df is None: return

    num_categories = len(df.columns)
    y_true = create_y_true(df, TARGET_USER, LIKE_THRESHOLD)

    summary_results = {
        'Tamanho do Enxame': [],
        'Media_F1': [], 'Std_Dev_F1': [],
        'Media_Precision': [], 'Std_Dev_Precision': [],
        'Media_Recall': [], 'Std_Dev_Recall': [],
        'Media_Fitness': [], 'Std_Dev_Fitness': []
    }

    for swarm_size in SWARM_SIZES_TO_TEST:
        print(f"\n--- EXECUTANDO TESTE COM {swarm_size} PARTÍCULAS ---")
        
        current_f1_scores = []; current_precision_scores = []; current_recall_scores = []; current_fitness_scores = []
        best_overall_fitness = -np.inf; best_run_history = [] 

        for i in range(N_EXECUTIONS):
            best_cost, best_pos, cost_history = run_pso_recommender(
                df,
                user_id=TARGET_USER,
                n_particles=swarm_size,
                itinerary_size=ITINERARY_SIZE,
                user_preference_score=USER_PREFERENCE_FACTOR, 
                verbose=False
            )
            
            y_pred = create_y_pred(best_pos, num_categories, ITINERARY_SIZE)
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            fitness_val = -best_cost
            
            current_f1_scores.append(f1); current_precision_scores.append(precision); current_recall_scores.append(recall)
            current_fitness_scores.append(fitness_val)
            
            if fitness_val > best_overall_fitness:
                best_overall_fitness = fitness_val
                best_run_history = cost_history 
            
            print(f"Enxame={swarm_size}, Exec {i+1}/{N_EXECUTIONS} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # SALVAR GRÁFICOS DETALHADOS
        run_output_dir = os.path.join(output_dir, f"{swarm_size}_particulas")
        criar_diretorio_com_readme(run_output_dir, f"# Resultados Detalhados (Enxame = {swarm_size})")
        salvar_resultados_detalhados(run_output_dir, swarm_size, current_f1_scores, current_fitness_scores, best_run_history)

        # Calcula estatísticas de SUMÁRIO
        summary_results['Tamanho do Enxame'].append(swarm_size)
        summary_results['Media_F1'].append(np.mean(current_f1_scores)); summary_results['Std_Dev_F1'].append(np.std(current_f1_scores))
        summary_results['Media_Precision'].append(np.mean(current_precision_scores)); summary_results['Std_Dev_Precision'].append(np.std(current_precision_scores))
        summary_results['Media_Recall'].append(np.mean(current_recall_scores)); summary_results['Std_Dev_Recall'].append(np.std(current_recall_scores))
        summary_results['Media_Fitness'].append(np.mean(current_fitness_scores)); summary_results['Std_Dev_Fitness'].append(np.std(current_fitness_scores))

    # Salva Tabela de Sumário FINAL e Gráfico de Comparação
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(output_dir, 'tabela_sumario_sensibilidade.csv'), index=False, float_format='%.4f')
    print("\n--- SUMÁRIO ESTATÍSTICO FINAL ---"); print(df_summary)

    # Gera Gráfico Comparativo Final
    x_labels = [str(size) for size in summary_results['Tamanho do Enxame']]; mean_f1_values = summary_results['Media_F1']; std_f1_values = summary_results['Std_Dev_F1']
    plt.figure(figsize=(10, 6)); bars = plt.bar(x_labels, mean_f1_values, yerr=std_f1_values, capsize=5, color=['lightblue', 'blue', 'darkblue'], edgecolor='black')
    plt.title('Comparação de Média do F1-Score por Tamanho do Enxame', fontsize=16); plt.xlabel('Tamanho do Enxame (Nº de Partículas)', fontsize=12); plt.ylabel('Média F1-Score (com Desvio Padrão)', fontsize=12)
    plt.ylim(0, 1.1); plt.bar_label(bars, fmt='%.4f'); plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.savefig(os.path.join(output_dir, 'grafico_comparacao_tamanho_enxame.png')); plt.close() 

def main():
    print("Iniciando a Execução Completa do TCC (PSO) com Função de Fitness Ponderada...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    criar_diretorio_com_readme(EDA_DIR, "# Análise Exploratória de Dados (EDA)")
    criar_diretorio_com_readme(SENSITIVITY_DIR, "# Análise de Sensibilidade do PSO")

    executar_analise_exploratoria(EDA_DIR)
    executar_analise_sensibilidade(SENSITIVITY_DIR)
    
    print("\n--- EXECUÇÃO COMPLETA FINALIZADA ---")

if __name__ == "__main__":
    main()