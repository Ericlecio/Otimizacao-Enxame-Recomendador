import pandas as pd
import numpy as np
from preprocessamento import carregar_e_processar_dados
from fitness import MAPA_GRUPOS

USUARIO_ALVO = "User 55"
GRUPO_TESTE = "Natureza"

def auditar_calculo():
    print(f"--- AUDITORIA DE CÁLCULO PARA: {USUARIO_ALVO} ---")
    
    # 1. Carregar Dados
    df = carregar_e_processar_dados("data/google_review_ratings.csv")
    if df is None: return

    # 2. Pegar índices do grupo "Natureza"
    indices_natureza = MAPA_GRUPOS[GRUPO_TESTE]
    print(f"\n1. Grupo '{GRUPO_TESTE}' usa as colunas de índice: {indices_natureza}")
    
    # 3. Dados do Usuário
    notas_usuario = df.loc[USUARIO_ALVO].iloc[indices_natureza]
    media_usuario = notas_usuario.mean()
    print(f"\n2. Notas do {USUARIO_ALVO} nestas colunas:")
    print(notas_usuario.to_string())
    print(f">> Média do Usuário (Mu): {media_usuario:.6f}")

    # 4. Dados Globais (Dataset Inteiro)
    todas_notas_natureza = df.iloc[:, indices_natureza]
    media_global = todas_notas_natureza.values.mean()
    print(f"\n3. Média Global de todos os usuários nestas colunas (Mg): {media_global:.6f}")

    # 5. O Cálculo Manual
    diferenca = media_usuario - media_global
    peso_calculado = 1.0 + diferenca
    
    print("\n4. Conferência da Fórmula:")
    print(f"   Diferença = {media_usuario:.6f} - {media_global:.6f} = {diferenca:.6f}")
    print(f"   Peso Final = 1.0 + ({diferenca:.6f})")
    print(f"   RESULTADO ESPERADO: {peso_calculado:.6f}")

    # 6. Comparar com o que o seu sistema calculou
    from fitness import calcular_perfil_usuario
    perfil_sistema = calcular_perfil_usuario(df, USUARIO_ALVO)
    peso_sistema = perfil_sistema[GRUPO_TESTE]
    
    print(f"\n5. Resultado do Sistema (fitness.py): {peso_sistema:.6f}")
    
    if abs(peso_calculado - peso_sistema) < 0.00001:
        print("\nSUCESSO! O cálculo bateu perfeitamente.")
    else:
        print("\nERRO! Os valores não batem.")

if __name__ == "__main__":
    auditar_calculo()