import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. PREENCHA MANUALMENTE COM OS RESULTADOS FINAIS ---

# Pegue estes valores do seu 'tabela_sumario_sensibilidade.csv'
# (Escolha o melhor resultado, ex: o de 80 partículas)
pso_scores = {
    'Precision': 0.9933, 
    'Recall': 0.9933,
    'F1-Score': 0.9933 
}

ag_scores = {
    'Precision': 0.9500, # Exemplo
    'Recall': 0.9400,    # Exemplo
    'F1-Score': 0.9450   # Exemplo
}
# -------------------------------------------------------------

# --- 2. Lógica do Gráfico ---
labels = list(pso_scores.keys())
pso_values = list(pso_scores.values())
ag_values = list(ag_scores.values())

x = np.arange(len(labels))
width = 0.35  # Largura das barras

fig, ax = plt.subplots(figsize=(12, 7))

# Plotar as barras
rects1 = ax.bar(x - width/2, pso_values, width, label='PSO (Seu Algoritmo)', color='darkblue')
rects2 = ax.bar(x + width/2, ag_values, width, label='Alg. Genético (AG)', color='green')

# Adicionar textos, títulos e legendas
ax.set_ylabel('Scores Médios (Média de 30 Execuções)', fontsize=14)
ax.set_title('Comparação de Desempenho: PSO vs. Algoritmo Genético', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)

# Adicionar rótulos de dados em cima das barras
ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

fig.tight_layout()

# Salvar na pasta principal de resultados
output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'grafico_comparacao_final_pso_vs_ag.png')
plt.savefig(save_path)

print(f"Gráfico 'grafico_comparacao_final_pso_vs_ag.png' salvo em '{output_dir}'.")