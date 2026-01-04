MAPA_GRUPOS = {
    'Historico': [0, 5, 14, 22],
    'Natureza': [2, 3, 7, 21, 23],
    'Gastronomia': [8, 9, 11, 13, 18, 20],
    'Entretenimento': [4, 6, 15],
    'Estadia': [1, 12],
    'BemEstar': [10, 16, 17, 19]
}

def calcular_perfil_usuario(df, user_id):
    notas_user = df.loc[user_id]
    media_global = df.mean()

    pesos = {}
    for grupo, idxs in MAPA_GRUPOS.items():
        idxs_validos = [i for i in idxs if i < len(df.columns)]
        if not idxs_validos:
            continue

        diff = notas_user.iloc[idxs_validos].mean() - media_global.iloc[idxs_validos].mean()
        pesos[grupo] = max(0.1, 1.0 + diff)

    return pesos
