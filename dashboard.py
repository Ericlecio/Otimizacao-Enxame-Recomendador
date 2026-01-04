import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

from core.categorias import CATEGORIAS
from core.dados import carregar_dados

API_URL = "http://127.0.0.1:8000/gerar-roteiro"

# =============================
# CONFIGURAÇÃO DA PÁGINA
# =============================
st.set_page_config(
    page_title="Sistema de Priorização",
    layout="wide"
)

st.title("Sistema Inteligente de Priorização")
st.caption(
    "Geração automática de prioridades baseada em otimização (PSO), preferências e restrições"
)

# =============================
# DADOS
# =============================
df = carregar_dados("data/google_review_ratings.csv")

# =============================
# SIDEBAR – CONFIGURAÇÕES
# =============================
st.sidebar.header("Configurações da Análise")

usuario = st.sidebar.selectbox(
    "Entidade analisada",
    df.index.tolist()
)

tamanho = st.sidebar.slider(
    "Quantidade de itens priorizados",
    min_value=1,
    max_value=10,
    value=5
)

bloqueadas = st.sidebar.multiselect(
    "Categorias excluídas da análise",
    options=df.columns.tolist(),
    format_func=lambda x: CATEGORIAS.get(x, x)
)

gerar = st.sidebar.button("Gerar Prioridade")

# =============================
# EXECUÇÃO
# =============================
if gerar:
    payload = {
        "user_id": usuario,
        "tamanho": tamanho,
        "categorias_bloqueadas": bloqueadas
    }

    resposta = requests.post(API_URL, json=payload).json()

    st.subheader("Ranking de Prioridades")

    # -----------------------------
    # TABELA DE RESULTADOS
    # -----------------------------
    dados = []
    for i, item in enumerate(resposta["roteiro"], 1):
        dados.append({
            "Prioridade": i,
            "Elemento Avaliado": item["categoria"],
            "Nota da Entidade": round(item["nota_usuario"], 3),
            "Relevância Global": round(item["popularidade"], 3),
            "Justificativa": item["motivo"]
        })

    df_resultado = pd.DataFrame(dados)
    st.dataframe(df_resultado, use_container_width=True)

    # -----------------------------
    # SCORE GLOBAL
    # -----------------------------
    st.markdown("### Score Global da Solução")
    st.write(
        f"O conjunto priorizado apresenta um score agregado de **{resposta['score']:.3f}**, "
        "indicando o nível geral de adequação às preferências e critérios definidos."
    )

    # -----------------------------
    # GRÁFICO – PERFIL
    # -----------------------------
    st.subheader("Perfil da Entidade Avaliada")

    categorias_plot = [
        c for c in df.columns if c not in bloqueadas
    ]

    notas_usuario = df.loc[usuario, categorias_plot]

    fig, ax = plt.subplots(figsize=(10, 4))
    notas_usuario.plot(kind="bar", ax=ax)

    ax.set_ylabel("Valor Normalizado")
    ax.set_xlabel("Categorias")
    ax.set_title("Distribuição de Avaliações")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    st.pyplot(fig)

    # -----------------------------
    # EXPLICAÇÃO
    # -----------------------------
    st.subheader("Critério de Priorização")
    st.write(
        "A priorização é obtida por meio de um algoritmo de Otimização por Enxame de Partículas (PSO), "
        "que busca maximizar a adequação entre as preferências da entidade analisada, a relevância média "
        "das categorias no conjunto de dados e as restrições impostas pelo usuário."
    )

else:
    st.info(
        "Defina os parâmetros da análise no painel lateral e execute a geração de prioridades."
    )
