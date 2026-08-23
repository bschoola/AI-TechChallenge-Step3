# Podcast — Episódio: Otimizando um Classificador de Câncer com Algoritmo Genético

> Roteiro solo, ~4 min, público técnico (devs/dados). Projeto: `1.GenAIPrediction`.

---

E aí, pessoal, bem-vindo a mais um episódio. Hoje eu quero falar sobre um projeto que mistura duas coisas que eu adoro: machine learning aplicado a um problema real — diagnóstico de câncer de mama — e otimização por Algoritmo Genético, feita do zero, sem biblioteca pronta. É o módulo `1.GenAIPrediction` de um Tech Challenge de pós-graduação em IA.

Vamos começar pelo problema. A base é o Wisconsin Breast Cancer Dataset — imagens de biópsias transformadas em variáveis numéricas: raio, textura, área, concavidade do contorno do tumor, entre outras. O objetivo é classificar cada caso como benigno ou maligno. E aqui já tem uma decisão de design interessante: em vez de jogar as trinta e poucas colunas do dataset original no modelo, o time selecionou só quatro features — `area_pior`, `textura_pior`, `pontos_concavos_pior` e `concavidade_pior`. Menos ruído, modelo mais enxuto, e como a gente vai ver depois com SHAP, é exatamente `area_pior` que carrega o maior peso na decisão final.

O modelo em si é uma Regressão Logística — simples, interpretável, e um baseline sólido pra contexto clínico, onde explicabilidade importa tanto quanto performance. Mas Regressão Logística tem hiperparâmetros que não são aprendidos durante o treino: `C`, que controla a força da regularização; `penalty`, l1 ou l2; `max_iter`; e `class_weight`. Encontrar a combinação ideal é um problema de busca. E é aí que entra o Algoritmo Genético.

A lógica é a metáfora biológica de sempre: cada indivíduo é um cromossomo de quatro genes — índices que apontam pra listas de valores possíveis em `config.py`. A população evolui geração após geração usando seleção por torneio, crossover de ponto único e mutação gene a gene, com elitismo garantindo que o melhor indivíduo nunca se perca. E o fitness — a nota que cada indivíduo recebe — é o F1-Score médio em cross-validation de cinco folds, calculado só em cima do treino, pra não vazar informação do teste.

Por que F1 e não Accuracy? Porque em diagnóstico médico, Accuracy mente. Um Falso Negativo — dizer que o tumor é benigno quando é maligno — é o pior erro possível: o paciente não recebe tratamento. Então Recall é a métrica que mais importa clinicamente, e o F1 entra como fitness porque equilibra Recall com Precision, evitando que o AG otimize um Recall alto às custas de alarmes falsos em excesso.

O projeto roda três experimentos com configurações diferentes de população, gerações e taxa de mutação — de população pequena e mutação alta, que explora bastante o espaço de busca, até população grande e mutação baixa, que explota mais o que já encontrou. No fim, compara os três contra o baseline sem otimização numa tabela de F1, Recall, Accuracy e ROC-AUC, e salva o modelo vencedor com `joblib`, pronto pra ser consumido por uma API.

E pra fechar o ciclo de confiança, tem uma camada de explicabilidade com SHAP, usando `LinearExplainer` em cima do pipeline treinado. Isso permite mostrar, feature por feature, o quanto cada variável empurrou a predição pra um lado ou pro outro — essencial quando o resultado vai influenciar uma decisão médica real.

Resumindo: é um pipeline compacto, mas que junta busca evolutiva implementada do zero, uma métrica de fitness escolhida com propósito clínico, e explicabilidade no fim da linha. Bom material pra estudar como Algoritmo Genético resolve otimização de hiperparâmetros sem depender de GridSearch ou de bibliotecas de AutoML.

É isso por hoje. Valeu, e até o próximo episódio.
