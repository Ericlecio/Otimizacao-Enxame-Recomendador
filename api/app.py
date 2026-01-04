from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from core.dados import carregar_dados
from core.pso import gerar_roteiro_pso

app = FastAPI(title="API Recomendador PSO")

df = carregar_dados("data/google_review_ratings.csv")

class Requisicao(BaseModel):
    user_id: str
    tamanho: int = 5
    categorias_bloqueadas: List[str] = []

@app.post("/gerar-roteiro")
def gerar(req: Requisicao):
    return gerar_roteiro_pso(
        df=df,
        user_id=req.user_id,
        tamanho=req.tamanho,
        categorias_bloqueadas=req.categorias_bloqueadas
    )
