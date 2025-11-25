import matplotlib.pyplot as plt
import numpy as np
import os


# pso_scores = {
#     'Precision': 0.9867, 
#     'Recall': 0.9867,
#     'F1-Score': 0.9867 
# }

# ag_scores = {
#     'Precision': 0.9500,
#     'Recall': 0.9400,
#     'F1-Score': 0.9450 
# }

labels = list(pso_scores.keys())
pso_values = list(pso_scores.values())
ag_values = list(ag_scores.values())

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, pso_values, width, label='PSO (Seu Algoritmo)', color='darkblue')
rects2 = ax.bar(x + width/2, ag_values, width, label='Alg. Genético (AG)', color='green')

ax.set_ylabel('Scores Médios (Média de 30 Execuções)', fontsize=14)
ax.set_title('Comparação de Desempenho Final: PSO vs. Algoritmo Genético', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)

ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'grafico_comparacao_final_pso_vs_ag.png')
plt.savefig(save_path)

print(f"Gráfico de Comparação Final salvo em '{output_dir}'.")