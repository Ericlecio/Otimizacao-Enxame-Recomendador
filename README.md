### Sistema Inteligente de Priorização e Recomendação com PSO

## Visão Geral

Este repositório apresenta um sistema inteligente de recomendação e priorização baseado no algoritmo de Otimização por Enxame de Partículas (Particle Swarm Optimization – PSO).

O sistema foi desenvolvido para resolver problemas de decisão onde é necessário ordenar e priorizar itens considerando múltiplos critérios, restrições e preferências específicas.

Embora o projeto utilize como exemplo um dataset de avaliações de categorias, sua arquitetura é genérica e extensível, podendo ser aplicada diretamente em contextos como:

- Priorização de inspeções estruturais
- Planejamento de manutenção preventiva
- Apoio à decisão em engenharia
- Sistemas de recomendação personalizados
- Análise de risco e criticidade

O projeto integra algoritmo, API e dashboard, formando uma solução completa de ponta a ponta.

## Problema Resolvido

Em cenários reais, é comum enfrentar situações como:

- Muitos itens para avaliar
- Recursos limitados (tempo, orçamento, equipe)
- Necessidade de justificar tecnicamente decisões

Este sistema responde à pergunta:
Quais itens devem ser priorizados e em qual ordem, considerando preferências, relevância global e restrições?

## Por que usar PSO?

A Otimização por Enxame de Partículas é adequada porque:

- Explora espaços de busca contínuos e combinatórios
- Não exige derivadas
- Converge rapidamente
- Permite múltiplos critérios no fitness
- Baixo custo computacional

Cada partícula representa uma solução candidata e o enxame evolui buscando maximizar um score global.

## Arquitetura

Dashboard (Streamlit) -> API (FastAPI) -> Núcleo PSO -> Dados

## Componentes

### Núcleo PSO

- Implementação Global Best
- Fitness baseado em preferências, relevância global e penalizações

### API FastAPI

- Exposição do algoritmo
- Recebe parâmetros e retorna ranking priorizado

### Dashboard Streamlit

- Interface limpa
- Visualização de rankings e gráficos

## Dataset

Linhas representam entidades e colunas representam categorias ou elementos avaliados.
Valores indicam relevância ou severidade normalizada.

## Execução

Instalar dependências:
pip install -r requirements.txt

Iniciar API:
uvicorn api.app:app --reload

Executar dashboard:
streamlit run dashboard/dashboard.py

## Aplicação em Inspeção Estrutural

O sistema pode ser adaptado para priorização de inspeções com base em risco, auxiliando engenheiros na tomada de decisão.

## Autor

Ericlecio Thiago
