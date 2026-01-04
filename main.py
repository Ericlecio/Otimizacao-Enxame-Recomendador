from core.dados import carregar_dados
from core.pso import gerar_roteiro_pso

df = carregar_dados("data/google_review_ratings.csv")

user_id = df.index[0]

resultado = gerar_roteiro_pso(
    df=df,
    user_id=user_id,
    tamanho_roteiro=5
)

print(resultado)
