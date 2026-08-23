# Roteiro de Estudo — Pipeline de Classificação com Algoritmo Genético

> Guia completo do código para desenvolvedores .NET em estudo de Python e IA.

---

## Visão Geral do Fluxo

Antes de entrar no detalhe de cada arquivo, é útil visualizar o que acontece quando você roda `python main.py`:

```
Carrega dados (CSV remoto)
        ↓
Seleciona features e codifica o alvo (M=1, B=0)
        ↓
Divide em treino (80%) e teste (20%)
        ↓
Treina Regressão Logística com hiperparâmetros padrão  ← Baseline
        ↓
Avalia no conjunto de teste (Accuracy, Precision, Recall, F1, ROC-AUC)
        ↓
Para cada um dos 3 experimentos do Algoritmo Genético:
    ├─ Cria população aleatória de "configurações de hiperparâmetros"
    ├─ Avalia cada configuração com cross-validation (só usa dados de treino)
    ├─ Evolui a população ao longo das gerações (seleção → cruzamento → mutação)
    └─ Treina o modelo final com a melhor configuração encontrada e avalia no teste
        ↓
Exibe tabela comparativa: Baseline vs Exp1 vs Exp2 vs Exp3
```

---

## `config.py` — O arquivo de constantes

Equivalente a um `appsettings.json` no .NET: centraliza tudo que é configurável.

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
```

- **`RANDOM_STATE = 42`**: semente do gerador de números aleatórios. Garante que toda vez que você rodar o código, a divisão treino/teste e os embaralhamentos internos produzam os mesmos resultados. É como fixar o `new Random(42)` no C#.
- **`TEST_SIZE = 0.2`**: 20% dos dados vão para o conjunto de teste; 80% para treino.
- **`CV_FOLDS = 5`**: o dataset de treino será dividido em 5 partes para a validação cruzada (detalhado mais adiante).

### Espaço de busca do Algoritmo Genético

```python
HP_C_VALUES        = [0.001, 0.01, 0.1, 1, 10, 100]
HP_PENALTY_VALUES  = ["l1", "l2"]
HP_MAX_ITER_VALUES = [100, 200, 500, 1000, 2000]
HP_CLASS_WEIGHT_VALUES = [None, "balanced"]

HP_GENE_SIZES = [6, 2, 5, 2]   # tamanho de cada lista acima
```

Esses são os **valores possíveis para cada hiperparâmetro** da Regressão Logística que o AG vai explorar. Pense nisso como o "universo de possibilidades" que o algoritmo vai varrer. Em vez de testar todas as 6×2×5×2 = **120 combinações** de forma exaustiva (como um `GridSearch`), o AG usa evolução para chegar a boas combinações com muito menos avaliações.

### Configurações dos 3 experimentos

```python
GA_EXPERIMENTS = [
    { "name": "Exp1", "population_size": 10, "generations": 20, "mutation_rate": 0.30, ... },
    { "name": "Exp2", "population_size": 20, "generations": 15, "mutation_rate": 0.20, ... },
    { "name": "Exp3", "population_size": 30, "generations": 25, "mutation_rate": 0.10, ... },
]
```

Cada experimento é um `dict` Python — equivalente a um `Dictionary<string, object>` no C#. Os três experimentos diferem intencionalmente nos parâmetros do próprio AG para observar como essas configurações afetam o resultado final.

---

## `src/data_loader.py` — Carregamento dos dados

```python
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    return df
```

- **`pd.read_csv`**: lê o arquivo CSV diretamente de uma URL remota (GitHub) e retorna um `DataFrame` — pense nisso como um `DataTable` do .NET, com linhas e colunas nomeadas.
- **`rename(columns=...)`**: traduz os nomes das colunas do inglês para o português usando o dicionário `COLUMN_RENAME_MAP`. É como fazer um `.ToDictionary()` e re-mapear propriedades.
- **`drop(columns=["Unnamed: 32"])`**: o CSV original tem uma coluna extra vazia gerada pelo Excel/Kaggle. `errors="ignore"` equivale a um `if (dict.ContainsKey(key))` antes de remover.

O retorno `pd.DataFrame` é o tipo central do pandas — todas as operações de dados do projeto passam por ele.

---

## `src/preprocessing.py` — Preparação dos dados

```python
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_bi = df.copy()
    df_bi["diagnostico_binario"] = df_bi["diagnostico"].map({"M": 1, "B": 0})
    X = df_bi[SELECTED_FEATURES]
    y = df_bi["diagnostico_binario"]
    return X, y
```

- **`df.copy()`**: cria uma cópia para não modificar o DataFrame original — boa prática equivalente a trabalhar com objetos imutáveis ou `Clone()`.
- **`.map({"M": 1, "B": 0})`**: transforma a coluna de texto `"M"/"B"` em numérica `1/0`. Modelos de ML precisam de números — o algoritmo não entende texto.
- **`X`** (features): as 4 colunas de entrada (`area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`). É a "entrada" do modelo.
- **`y`** (alvo/label): a coluna que o modelo deve aprender a prever (0=benigno, 1=maligno).

```python
def split_data(X, y) -> tuple[...]:
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- **`train_test_split`**: divide os dados em treino e teste. O parâmetro `stratify=y` garante que a proporção de tumores malignos/benignos seja a mesma nos dois conjuntos — sem isso, por azar, o conjunto de teste poderia ter muito mais casos de um tipo do que o outro, distorcendo a avaliação.

---

## `src/models.py` — Definição do modelo

```python
def build_baseline_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
```

### O que é um `Pipeline`?

`Pipeline` é uma cadeia de transformações + modelo final, executadas em sequência. É similar a um padrão de Builder ou Middleware no .NET. Aqui temos dois passos:

1. **`StandardScaler`** — normalização: transforma cada feature para ter média 0 e desvio padrão 1. É indispensável para a Regressão Logística porque o algoritmo é sensível à escala dos dados. Sem isso, uma feature com valores na casa dos milhares dominaria as que têm valores entre 0 e 1.

2. **`LogisticRegression`** — o modelo classificador (detalhado na próxima seção).

A vantagem do `Pipeline` é que, ao chamar `.fit(X_train, y_train)`, ele aplica o `StandardScaler` nos dados de treino e, automaticamente, usa os parâmetros aprendidos (média, desvio) para transformar os dados de teste durante o `.predict()`. Isso evita um erro clássico chamado **data leakage** (vazamento de dados): calcular a normalização com base nos dados de teste.

```python
def build_model_from_hyperparams(hyperparams: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**hyperparams)),
    ])
```

- **`**hyperparams`**: o `**` desempacota o dicionário como argumentos nomeados. É equivalente a passar um objeto de configurações para um construtor via reflexão no C#. Por exemplo, `{"C": 0.1, "penalty": "l1"}` vira `LogisticRegression(C=0.1, penalty="l1")`.

---

## Regressão Logística — Entendendo o modelo

### O que ela faz?

Apesar do nome "regressão", ela é um **classificador**. Ela calcula a probabilidade de um tumor ser maligno dado os valores das features. A saída é um número entre 0 e 1 — se for ≥ 0.5, classifica como maligno (1); caso contrário, benigno (0).

Matematicamente, ela aplica a **função sigmoide** sobre uma combinação linear das features:

```
P(maligno) = 1 / (1 + e^-(w1*area_pior + w2*textura_pior + ...))
```

Durante o treino, o modelo ajusta os pesos `w1, w2, ...` para minimizar os erros de classificação.

### Hiperparâmetros relevantes (os genes do AG)

| Hiperparâmetro | O que controla | Analogia |
|---|---|---|
| **`C`** | Força da regularização (inverso da penalidade). `C` pequeno = modelo mais simples/conservador. | Quanto menor, mais você "pune" o modelo por ser complexo — evita overfitting. |
| **`penalty`** | Tipo de regularização: `l1` tende a zerar pesos (feature selection automática); `l2` distribui pesos menores por todas as features. | `l1` é como um filtro que elimina variáveis; `l2` é como suavizar todas elas. |
| **`max_iter`** | Número máximo de iterações do otimizador interno para convergir. | Como um `timeout` para o loop de treino. |
| **`class_weight`** | `"balanced"` faz o modelo prestar mais atenção à classe minoritária. | Útil quando há poucos casos malignos comparados a benignos. |
| **`solver`** | Algoritmo de otimização interno. `liblinear` é necessário para `l1`; `lbfgs` é o padrão para `l2`. | Não é exposto diretamente; é derivado automaticamente a partir do `penalty`. |

---

## `src/evaluation.py` — Avaliação do modelo

### `evaluate_model`

```python
def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)         # previsões binárias: 0 ou 1
    y_prob = model.predict_proba(X_test)[:, 1]  # probabilidade de ser maligno
    ...
```

- **`predict`**: retorna a classificação final (0 ou 1) para cada amostra.
- **`predict_proba`**: retorna a probabilidade de cada classe. `[:, 1]` pega a coluna da classe positiva (maligno). Necessário para calcular o ROC-AUC.

### As métricas e por que cada uma importa

Para entender as métricas, é preciso conhecer os 4 tipos de resultado possíveis:

```
                   Predição: Benigno    Predição: Maligno
Real: Benigno   →  Verdadeiro Negativo  |  Falso Positivo (alarme falso)
Real: Maligno   →  Falso Negativo (!)   |  Verdadeiro Positivo
```

O **Falso Negativo** é o caso mais grave neste domínio: o modelo diz "benigno" mas é maligno — o paciente não recebe tratamento.

| Métrica | Fórmula simplificada | Pergunta que responde |
|---|---|---|
| **Accuracy** | (acertos) / (total) | "De tudo que foi predito, quanto acertou?" |
| **Precision** | VP / (VP + FP) | "Dos que foram classificados como malignos, quantos realmente eram?" |
| **Recall** | VP / (VP + FN) | "Dos tumores malignos reais, quantos foram detectados?" |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | "Equilíbrio entre Precision e Recall" |
| **ROC-AUC** | Área sob a curva ROC | "Quão bem o modelo separa as duas classes?" — 1.0 = perfeito |

> O **Recall** é a métrica mais crítica neste contexto clínico. Um Recall de 0.90 significa que 10% dos tumores malignos passaram despercebidos. O **F1-Score** é usado como fitness do AG por equilibrar os dois lados.

### `cross_validate_model`

```python
def cross_validate_model(name, model, X, y) -> None:
    scores = cross_validate(model, X, y, cv=5, scoring={"recall": "recall", "f1": "f1"})
    print(f"  Mean Recall:   {scores['test_recall'].mean():.4f}")
    print(f"  Mean F1-score: {scores['test_f1'].mean():.4f}")
```

**Cross-validation (validação cruzada)** é uma técnica para avaliar o modelo de forma mais robusta do que uma única divisão treino/teste.

Com `cv=5`, o que acontece:
1. O dataset inteiro é dividido em 5 partes iguais (folds).
2. O modelo é treinado 5 vezes: em cada rodada, 4 partes são usadas para treino e 1 parte (diferente a cada vez) é usada para teste.
3. As 5 métricas resultantes são calculadas e o código exibe a média.

```
Fold 1: [TESTE ] [treino] [treino] [treino] [treino]
Fold 2: [treino] [TESTE ] [treino] [treino] [treino]
Fold 3: [treino] [treino] [TESTE ] [treino] [treino]
Fold 4: [treino] [treino] [treino] [TESTE ] [treino]
Fold 5: [treino] [treino] [treino] [treino] [TESTE ]
                                              → média dos 5 resultados
```

Isso é muito mais confiável do que uma única avaliação, porque a performance do modelo é medida em diferentes subconjuntos dos dados. No contexto médico, dá mais segurança de que o modelo não "memorizou" os dados.

---

## `src/genetic_algorithm.py` — O coração do projeto

Esta é a implementação mais complexa. Vamos entender cada conceito antes de olhar o código.

### Por que usar Algoritmo Genético?

A Regressão Logística tem hiperparâmetros que precisam ser configurados antes do treino (não são aprendidos pelo modelo automaticamente). Encontrar a melhor combinação é um problema de **otimização**.

Abordagens possíveis:
- **Grid Search**: testa todas as combinações. Certo, mas lento (120 combinações × 5 folds = 600 treinos).
- **Random Search**: testa combinações aleatórias. Rápido, mas sem inteligência.
- **Algoritmo Genético**: inspira-se na evolução natural para convergir para boas soluções eficientemente.

### A metáfora biológica

| Conceito biológico | Equivalente no código |
|---|---|
| Organismo/Indivíduo | Uma configuração de hiperparâmetros |
| Cromossomo | A lista `[c_idx, penalty_idx, max_iter_idx, class_weight_idx]` |
| Gene | Um elemento da lista (ex: o índice que representa `C=0.1`) |
| Aptidão (fitness) | F1-Score do modelo com aquela configuração |
| População | Conjunto de várias configurações em uma geração |
| Geração | Uma rodada de evolução |
| Seleção natural | Indivíduos com maior F1 têm mais chance de se reproduzir |
| Reprodução/Cruzamento | Dois indivíduos trocam genes para gerar filhos |
| Mutação | Um gene muda aleatoriamente |
| Elitismo | O melhor indivíduo sempre passa para a próxima geração |

---

### Representação dos genes: `decode_individual`

```python
# Individual layout: [c_idx, penalty_idx, max_iter_idx, class_weight_idx]
#                     [  0,       1,           2,              3        ]
```

Cada indivíduo é uma lista de 4 inteiros — **índices** apontando para as listas de valores em `config.py`.

```python
def decode_individual(individual: list[int]) -> dict:
    penalty = HP_PENALTY_VALUES[individual[1]]      # ex: "l2"
    solver  = "liblinear" if penalty == "l1" else "lbfgs"
    return {
        "C":            HP_C_VALUES[individual[0]],         # ex: 0.1
        "penalty":      penalty,                            # ex: "l2"
        "solver":       solver,                             # ex: "lbfgs"
        "max_iter":     HP_MAX_ITER_VALUES[individual[2]],  # ex: 500
        "class_weight": HP_CLASS_WEIGHT_VALUES[individual[3]], # ex: None
        "random_state": RANDOM_STATE,
    }
```

**Por que usar índices e não os valores diretamente?**
Trabalhar com inteiros facilita muito os operadores genéticos (cruzamento, mutação). É mais simples dizer "troca o gene de posição 2" do que lidar com strings ou floats de domínios diferentes.

**Exemplo concreto:**
```
individual = [2, 1, 3, 0]
               ↓  ↓  ↓  ↓
decode →  C=0.1, penalty="l2", max_iter=1000, class_weight=None
```

O `solver` é derivado automaticamente do `penalty` porque a scikit-learn exige essa compatibilidade: `l1` só funciona com `liblinear` ou `saga`; `l2` funciona com `lbfgs`.

---

### Inicialização da população

```python
def _random_individual() -> list[int]:
    return [random.randrange(size) for size in HP_GENE_SIZES]

def _initialize_population(size: int) -> list[list[int]]:
    return [_random_individual() for _ in range(size)]
```

- `random.randrange(size)` gera um inteiro aleatório entre 0 e `size-1`.
- A list comprehension `[... for _ in range(size)]` é o equivalente Python de um `Enumerable.Range(0, size).Select(_ => ...)` em LINQ.

Para `population_size=10`, isso cria uma lista com 10 indivíduos, cada um com 4 genes sorteados aleatoriamente:

```python
[
  [3, 0, 2, 1],   # C=1, penalty="l1", max_iter=500, class_weight="balanced"
  [5, 1, 4, 0],   # C=100, penalty="l2", max_iter=2000, class_weight=None
  ...              # 8 outros indivíduos aleatórios
]
```

---

### Função Fitness: `_fitness`

```python
def _fitness(individual: list[int], X_train, y_train) -> float:
    hyperparams = decode_individual(individual)
    model = build_model_from_hyperparams(hyperparams)
    scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    return float(scores.mean())
```

Esta é a função mais importante do AG. Ela responde: **"quão bom é este indivíduo?"**

O processo:
1. Decodifica os índices nos hiperparâmetros reais
2. Constrói um Pipeline com esses hiperparâmetros
3. Avalia com 5-fold cross-validation **usando apenas `X_train`** — o conjunto de teste nunca é visto durante a busca
4. Retorna o F1-Score médio das 5 dobras

> **Atenção ao custo computacional:** para cada indivíduo avaliado, o modelo é treinado 5 vezes (cross-validation). Com população de 30 e 25 gerações, são 30×25×5 = **3.750 treinos** só no Exp3.

---

### Seleção por Torneio: `_tournament_select`

```python
def _tournament_select(population, fitnesses, tournament_size) -> list[int]:
    candidates = random.sample(range(len(population)), tournament_size)
    winner = max(candidates, key=lambda i: fitnesses[i])
    return population[winner][:]
```

O torneio funciona assim:
1. Sorteia aleatoriamente `tournament_size` (=3) índices da população
2. Entre esses 3 candidatos, retorna o que tem maior F1-Score
3. O `[:]` no final cria uma cópia da lista (equivalente a `.ToList()` no C#)

**Por que torneio e não simplesmente pegar os melhores?**
Se sempre pegássemos os 2 melhores para reproduzir, a população convergiria muito rápido para uma solução local (premature convergence). O torneio mantém diversidade: indivíduos medianos ainda têm chance de ser selecionados, o que permite explorar mais o espaço de busca.

```
População: [A=0.91, B=0.94, C=0.88, D=0.95, E=0.90]
Torneio com size=3: sorteia {B, C, E} → vencedor: B (0.94)
Próximo torneio:    sorteia {A, D, E} → vencedor: D (0.95)
```

---

### Cruzamento (Crossover): `_crossover`

```python
def _crossover(parent1, parent2, crossover_rate) -> tuple[list, list]:
    if random.random() > crossover_rate or _GENE_COUNT < 2:
        return parent1[:], parent2[:]           # sem cruzamento: cópia direta
    point = random.randint(1, _GENE_COUNT - 1)  # ponto de corte entre 1 e 3
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

O cruzamento de ponto único divide dois pais em um ponto aleatório e combina as partes:

```
Pai 1:  [2, 0, 3, 1]     Pai 2:  [5, 1, 0, 0]
               ↑ ponto de corte = 2

Filho 1: [2, 0 | 0, 0]   (genes 0-1 do pai1 + genes 2-3 do pai2)
Filho 2: [5, 1 | 3, 1]   (genes 0-1 do pai2 + genes 2-3 do pai1)
```

- `random.random()` retorna float entre 0 e 1. Se for > `crossover_rate`, não há cruzamento e os pais são copiados sem alteração.
- Com `crossover_rate=0.80`, 80% das reproduções geram filhos com material genético misto.

---

### Mutação: `_mutate`

```python
def _mutate(individual, mutation_rate) -> list[int]:
    return [
        random.randrange(HP_GENE_SIZES[i]) if random.random() < mutation_rate else gene
        for i, gene in enumerate(individual)
    ]
```

Para cada gene: com probabilidade `mutation_rate`, substitui por um valor aleatório; caso contrário, mantém o original.

```
Antes:   [2, 0, 3, 1]    (mutation_rate=0.30)
Gene 0: random=0.72 > 0.30 → mantém 2
Gene 1: random=0.18 < 0.30 → muta  → sorteia novo valor (ex: 1)
Gene 2: random=0.55 > 0.30 → mantém 3
Gene 3: random=0.25 < 0.30 → muta  → sorteia novo valor (ex: 0)
Depois:  [2, 1, 3, 0]
```

A mutação introduz **diversidade** e permite que o AG escape de ótimos locais. Taxa alta (Exp1=0.30) explora mais o espaço; taxa baixa (Exp3=0.10) explora menos mas explota melhor o que já foi encontrado.

---

### Loop principal: `run_ga`

```python
def run_ga(X_train, y_train, config) -> tuple[dict, float, list[float]]:
    population = _initialize_population(population_size)
    best_individual = population[0]
    best_f1 = 0.0

    for gen in range(1, generations + 1):
        # 1. Avalia todos os indivíduos da geração atual
        fitnesses = [_fitness(ind, X_train, y_train) for ind in population]

        # 2. Registra o melhor global
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_f1:
            best_f1 = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx][:]

        # 3. Constrói a próxima geração
        new_population = [best_individual[:]]   # elitismo: garante o melhor

        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitnesses, tournament_size)
            p2 = _tournament_select(population, fitnesses, tournament_size)
            c1, c2 = _crossover(p1, p2, crossover_rate)
            new_population.append(_mutate(c1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(_mutate(c2, mutation_rate))

        population = new_population

    return decode_individual(best_individual), best_f1, history
```

**Fluxo de uma geração:**

```
Geração N:
  população atual → avalia fitness de cada um
                  → encontra o melhor da geração
                  → inicia nova população com o melhor (elitismo)
                  → enquanto nova pop não está cheia:
                       seleciona 2 pais por torneio
                       faz cruzamento (80-90% das vezes)
                       aplica mutação em cada filho
                       adiciona filhos à nova população
                  → nova população substitui a anterior
  Geração N+1 começa com população renovada
```

**Elitismo** (linha `new_population = [best_individual[:]]`): o melhor indivíduo encontrado até agora sempre vai direto para a próxima geração sem sofrer cruzamento nem mutação. Isso garante que o resultado nunca piore entre gerações.

**Retorno da função:** uma tupla com 3 elementos:
```python
(
    {"C": 0.1, "penalty": "l1", ...},   # melhores hiperparâmetros decodificados
    0.9605,                              # melhor F1-Score encontrado
    [0.9553, 0.9605, 0.9605, ...]       # histórico de F1 por geração
)
```

---

## `main.py` — Orquestração do pipeline

```python
def main() -> None:
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
```

Python suporta retorno múltiplo de funções via tuplas. O `split_data` retorna 4 valores de uma vez, que são atribuídos diretamente — equivalente a `(var1, var2) = metodo()` com tuplas nomeadas no C# 7+.

### Baseline

```python
baseline = build_baseline_model()
baseline.fit(X_train, y_train)
baseline_metrics = evaluate_model("Regressão Logística — Baseline", baseline, X_test, y_test)
cross_validate_model("Regressão Logística — Baseline", build_baseline_model(), X, y)
```

- **`.fit(X_train, y_train)`**: treina o modelo — é aqui que o `StandardScaler` aprende a média/desvio e a `LogisticRegression` aprende os pesos.
- Note que `build_baseline_model()` é chamado **duas vezes**: uma para treinar e avaliar no teste, outra para o cross-validate. Isso porque o cross-validate precisa de um modelo "virgem" (não treinado) para treinar e avaliar nas 5 dobras internamente.

### Loop dos experimentos GA

```python
for experiment in GA_EXPERIMENTS:
    best_hyperparams, best_cv_f1, _ = run_ga(X_train, y_train, experiment)

    optimized_model = build_model_from_hyperparams(best_hyperparams)
    optimized_model.fit(X_train, y_train)
    opt_metrics = evaluate_model(f"LR Otimizada ({experiment['name']})", optimized_model, X_test, y_test)
```

- O `_` no `best_hyperparams, best_cv_f1, _` descarta o terceiro elemento da tupla retornada por `run_ga` (o histórico por geração). Em Python, `_` é convenção para "não me importa este valor".
- O modelo otimizado é treinado com **todos os dados de treino** (`X_train`) usando os melhores hiperparâmetros encontrados pelo AG. O AG usou cross-validation **dentro** de `X_train` para avaliar — o `X_test` só é usado aqui, ao final, para a avaliação definitiva.

### Tabela comparativa

```python
def _print_comparison_table(results: list[dict]) -> None:
    col_widths = [max(len(r["name"]) for r in results) + 2, 10, 10, 10, 10]
    ...
    best = max(results, key=lambda r: r["metrics"]["f1"])
```

- `max(..., key=lambda r: r["metrics"]["f1"])`: encontra o dicionário com maior F1. O parâmetro `key` é como o `IComparer<T>` do C# — define o critério de comparação. `lambda r: ...` é uma função anônima, equivalente ao `x => x.Metrics.F1` em LINQ.

---

## Os 3 Experimentos — Diferenças e Trade-offs

| | Exp1 | Exp2 | Exp3 |
|---|---|---|---|
| `population_size` | 10 | 20 | 30 |
| `generations` | 20 | 15 | 25 |
| `mutation_rate` | 0.30 | 0.20 | 0.10 |
| `crossover_rate` | 0.80 | 0.80 | 0.90 |
| Total de avaliações | 10×20 = 200 | 20×15 = 300 | 30×25 = 750 |

**Exp1 — Pop. Pequena / Mutação Alta:**
- Menos indivíduos, mas evolui por mais gerações com alta diversidade (mutação 30%).
- Converge mais devagar, explora bastante o espaço — bom para escapar de mínimos locais.

**Exp2 — Pop. Média / Mutação Média:**
- Balanceado. População maior que Exp1 garante mais diversidade inicial; mutação moderada.
- Boa relação entre exploração e explotação.

**Exp3 — Pop. Grande / Mutação Baixa:**
- Muitos indivíduos por geração; mutação conservadora (10%) — explota bem o que encontrou mas pode convergir prematuramente.
- Mais caro computacionalmente (750 avaliações × 5 folds = 3.750 treinos).

---

## Resumo: Por que o AG funciona aqui?

O Algoritmo Genético funciona bem para este problema porque:

1. **O espaço de busca é discreto e finito** — 120 combinações possíveis são facilmente representadas por índices inteiros.
2. **Avaliar uma combinação é relativamente caro** — cada fitness requer 5 treinos de modelo; o AG avalia muito menos combinações que o Grid Search.
3. **Soluções similares tendem a ter fitness similar** — `C=0.1` e `C=1` provavelmente dão resultados parecidos, o que torna o cruzamento útil (herdar genes bons de dois pais).
4. **O elitismo garante convergência** — o resultado nunca piora, apenas mantém ou melhora ao longo das gerações.

---

## `ChoosenModel/` — Modelo salvo e seus metadados

Após os 3 experimentos, o melhor modelo é persistido em disco para ser consumido pela API sem precisar retreinar.

```
ChoosenModel/
├── cancer_model.joblib    ← pipeline serializado (StandardScaler + LogisticRegression)
└── model_metadata.json   ← metadados do experimento vencedor
```

### `model_metadata.json`

```json
{
  "experiment": "Exp1 — Pop. Pequena / Mutação Alta",
  "hyperparameters": {
    "C": 0.1, "penalty": "l1", "solver": "liblinear",
    "max_iter": 1000, "class_weight": null, "random_state": 42
  },
  "features": ["area_pior", "textura_pior", "pontos_concavos_pior", "concavidade_pior"],
  "metrics": {
    "accuracy": 0.9649,
    "precision": 1.0,
    "recall": 0.9048,
    "f1": 0.95,
    "roc_auc": 0.9993
  }
}
```

**Por que `joblib` e não `pickle`?** O `joblib` é mais eficiente para arrays NumPy — que é exatamente o que o scikit-learn usa internamente. Pense nele como um `BinaryFormatter` mais performático para objetos de ML.

O `Pipeline` serializado contém **tanto o scaler quanto o modelo** juntos. Quando a API carregar o arquivo e chamar `.predict()`, os dados passarão automaticamente pela normalização antes de chegarem ao classificador — nenhum pré-processamento manual necessário.

---

## `2.PredictionApi/` — API REST + Fila Assíncrona

Esta é a camada de serviço do projeto: recebe os dados do frontend, faz a predição com o modelo salvo e gera um laudo clínico via LLM (Ollama).

### Arquitetura da camada

```
Browser
   │
   ▼
FastAPI (main.py)          ← porta 8000
   │  ├─ POST /predict/breastCancer         → predição síncrona (< 50ms)
   │  └─ POST /predict/breastCancer/laudo  → predição + dispara task no Celery
   │
   ├──── Redis (broker) ─────────────────── porta 6379
   │
   └──── Celery Worker (tasks.py)
              └─ chama Ollama → llama3.2:3b → gera laudo
                                              porta 11434
```

O ponto central do design é a **separação entre predição e geração do laudo**. A Regressão Logística responde em milissegundos, mas o LLM pode levar 30–60 segundos. Se tudo fosse síncrono, o browser daria timeout. A solução é processar o laudo em background.

---

### `main.py` — FastAPI

#### Lifespan: carregamento do modelo na inicialização

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    _model = joblib.load(MODEL_PATH)           # carrega o Pipeline do disco
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)
    yield                                      # API fica disponível
```

O `lifespan` é o equivalente ao `Startup.cs` / `Program.cs` no ASP.NET Core: código executado uma vez antes da API começar a atender requisições. O `yield` separa o que acontece antes (startup) do que acontece depois (shutdown). O modelo é carregado **uma única vez** em memória e reutilizado em todas as requisições — exatamente o que um singleton faria no .NET.

#### Schemas Pydantic — validação de entrada e saída

```python
class BreastCancerInput(BaseModel):
    area_pior: float = Field(..., gt=0, ...)
    textura_pior: float = Field(..., gt=0, ...)
    pontos_concavos_pior: float = Field(..., ge=0, ...)
    concavidade_pior: float = Field(..., ge=0, ...)
```

`BaseModel` do Pydantic é o equivalente a um `record` ou DTO com `DataAnnotations` no C#. O FastAPI valida automaticamente o JSON recebido contra esse schema e retorna HTTP 422 com detalhes se algum campo for inválido — sem nenhum código de validação manual.

- `Field(..., gt=0)`: o `...` significa campo obrigatório; `gt=0` significa "maior que zero". É como `[Required]` + `[Range(min, double.MaxValue)]` juntos.
- `ge=0`: "maior ou igual a zero" (`>=`).

#### Endpoint principal: `POST /predict/breastCancer/laudo`

```python
def predict_breast_cancer_laudo(data: BreastCancerInput) -> LaudoAsyncResponse:
    pred = _run_prediction(data)        # 1. predição síncrona com o modelo

    task = gerar_laudo_task.delay(      # 2. enfileira a geração do laudo
        area_pior=data.area_pior,
        diagnostico=pred.diagnostico,
        confianca=pred.confianca,
        ...
    )

    return LaudoAsyncResponse(**pred.model_dump(), task_id=task.id)  # 3. retorna imediatamente
```

O `.delay()` do Celery enfileira a tarefa no Redis e retorna um `task_id` instantaneamente — sem esperar o LLM terminar. O cliente recebe a predição em < 100ms e usa o `task_id` para fazer polling.

#### Função de predição: `_run_prediction`

```python
def _run_prediction(data: BreastCancerInput) -> PredictionResponse:
    features = [[data.area_pior, data.textura_pior,
                 data.pontos_concavos_pior, data.concavidade_pior]]

    classe         = int(_model.predict(features)[0])
    probabilidades = _model.predict_proba(features)[0]
    prob_benigno   = round(float(probabilidades[0]), 4)
    prob_maligno   = round(float(probabilidades[1]), 4)
    confianca      = round(float(probabilidades[classe]) * 100, 2)
    ...
```

- `features` é uma lista de listas — o scikit-learn espera uma matriz 2D onde cada linha é uma amostra. Mesmo com uma única amostra, é necessário o `[[...]]`.
- `predict_proba(features)[0]` retorna um array com a probabilidade de cada classe: `[prob_benigno, prob_maligno]`.
- `probabilidades[classe]` pega a probabilidade da classe prevista — que é usada como "confiança do modelo".

#### Endpoint de polling: `GET /predict/breastCancer/laudo/status/{task_id}`

```python
def laudo_status(task_id: str) -> LaudoStatusResponse:
    result = AsyncResult(task_id, app=celery_app)
    state_map = {
        "PENDING": "pending", "STARTED": "processing",
        "SUCCESS": "completed", "FAILURE": "failed",
    }
    status = state_map.get(result.state, "processing")
    laudo = result.result if status == "completed" else None
    ...
```

O `AsyncResult` consulta o Redis para saber o estado da tarefa. O frontend chama esse endpoint a cada 2 segundos até receber `"completed"` — padrão clássico de **polling assíncrono**, similar ao `IAsyncResult` + loop de verificação no .NET.

---

### `tasks.py` — Celery Worker e Geração do Laudo

#### Configuração do Celery

```python
celery_app = Celery("oncotech", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    result_expires=3600,           # laudos ficam no Redis por 1 hora
    worker_prefetch_multiplier=1,  # processa uma tarefa por vez — respeita GPU única
)
```

O Celery usa o Redis como **broker** (fila de tarefas) e como **backend** (armazena resultados). O `worker_prefetch_multiplier=1` é crítico aqui: garante que o worker não pegue duas tarefas ao mesmo tempo, evitando que dois processos disputem a GPU do Ollama.

#### A tarefa: `gerar_laudo_task`

```python
@celery_app.task(name="tasks.gerar_laudo", bind=True, max_retries=2)
def gerar_laudo_task(self, area_pior, textura_pior, ..., diagnostico, confianca, ...):
    client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
    ...
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)   # tenta novamente em 5s
```

**Por que `OpenAI` client para chamar o Ollama?** O Ollama expõe uma API compatível com o formato da OpenAI em `/v1`. Isso permite usar o SDK oficial da OpenAI apontando para o servidor local — `base_url` e `api_key="ollama"` (qualquer string) fazem o redirecionamento.

**`temperature=0.2`**: quanto mais próximo de 0, mais determinístico e conservador o LLM fica. Para um laudo médico, isso é desejável — queremos consistência, não criatividade.

**`max_retries=2` + `self.retry(countdown=5)`**: se o Ollama estiver ocupado ou retornar erro, a tarefa é recolocada na fila após 5 segundos, até 2 tentativas adicionais. É o equivalente a um `Polly RetryPolicy` no .NET.

#### O prompt

O prompt instrui o LLM a agir como assistente médico em oncologia mamária e estrutura o laudo em quatro parágrafos fixos: **Achados morfológicos**, **Interpretação**, **Conduta sugerida** e **Observação importante**. As regras explícitas ("NÃO inclua campos em branco", "máximo 200 palavras") são essenciais para controlar a saída do modelo — LLMs sem restrições tendem a produzir conteúdo variável e imprevisível.

---

## `3.Front/` — Frontend Angular

O frontend é uma SPA (Single Page Application) em Angular que consome a API e apresenta o resultado ao usuário.

### Fluxo completo na interface

```
Usuário preenche os 4 campos e clica "Iniciar Análise"
        ↓
POST /predict/breastCancer/laudo
        ↓
Exibe imediatamente: diagnóstico, confiança, probabilidades
        ↓
Inicia polling a cada 2s → GET /predict/breastCancer/laudo/status/{task_id}
        ↓
Quando status = "completed" → exibe o laudo clínico no documento
```

### `prediction.service.ts` — Service Angular

O service encapsula toda a comunicação com a API. Equivale a um `HttpClient` wrapper no .NET.

```typescript
requestLaudo(input: PredictionInput): Observable<LaudoAsyncResponse> {
    return this.http.post<LaudoAsyncResponse>(
        `${this.apiUrl}/predict/breastCancer/laudo`, input
    );
}

pollLaudo(taskId: string): Observable<LaudoStatusResponse> {
    return interval(POLL_INTERVAL_MS).pipe(          // emite a cada 2s
        switchMap(() =>
            this.http.get<LaudoStatusResponse>(
                `${this.apiUrl}/predict/breastCancer/laudo/status/${taskId}`
            ).pipe(catchError(() => EMPTY))          // ignora erros de rede
        ),
        takeWhile(
            (r) => r.status !== 'completed' && r.status !== 'failed',
            true   // inclui o último emit que encerrou o loop
        )
    );
}
```

`interval(2000).pipe(switchMap(...))` é o padrão RxJS para polling: a cada 2 segundos, cancela a requisição anterior (se ainda estiver em andamento) e inicia uma nova. `takeWhile(..., true)` para o Observable assim que o status terminal for recebido, evitando requisições infinitas.

### `app.component.html` — Interface

A interface é dividida em duas áreas principais que aparecem em sequência:

**Antes da análise:** formulário com os 4 campos do exame (`area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`) e validação Angular (`[(ngModel)]`, `required`, `min`).

**Após a análise:** duas colunas lado a lado — à esquerda, o diagnóstico com barras de probabilidade e os parâmetros informados; à direita, o laudo clínico formatado como documento médico (com cabeçalho, corpo em Markdown renderizado e rodapé com aviso de responsabilidade e botão de impressão).

O banner de diagnóstico muda visualmente dependendo do resultado: vermelho para Maligno, verde para Benigno — usando classes CSS condicionais (`[class.diagnosis-banner--maligno]="isMaligno"`).

---

## Infraestrutura — `docker-compose.yml`

O projeto inteiro sobe com um único `docker compose up`. A ordem de inicialização é controlada por `depends_on` com `healthcheck`.

```
docker compose up
        ↓
ollama (porta 11434)          ← healthcheck: "ollama list"
        ↓
ollama-setup                  ← baixa llama3.2:3b (one-shot)
        ↓
ollama-warmup                 ← pré-carrega o modelo na VRAM (one-shot)
        ↓
redis (porta 6379)            ← healthcheck: "redis-cli ping"
        ↓
celery-worker + api           ← dependem do warmup + redis
        ↓
front (porta 4200)            ← só sobe após a API passar no healthcheck
```

### Por que o `ollama-warmup`?

O `keep_alive` padrão do Ollama descarrega o modelo da VRAM após 5 minutos de inatividade. Sem o warmup, a **primeira requisição** após o cold start carregaria o modelo (~5–10s de espera). O warmup resolve isso: o modelo já está na VRAM quando a API e o worker sobem, garantindo latência baixa desde a primeira chamada.

### Variáveis de ambiente relevantes

| Variável | Valor | Onde é usada |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` | ollama-setup e warmup |
| `OLLAMA_URL` | `http://ollama:11434` | celery-worker e api |
| `OLLAMA_MODEL` | `llama3.2:3b` | celery-worker e api |
| `REDIS_URL` | `redis://redis:6379/0` | celery-worker e api |

Os containers se comunicam pelo nome do serviço (ex: `ollama`, `redis`) — o Docker resolve esses nomes como DNS interno na rede criada pelo Compose.

---

## Fluxo completo de ponta a ponta

Para fechar, o caminho percorrido por uma única requisição do usuário até o laudo final:

```
1. Usuário preenche o formulário Angular e clica em "Iniciar Análise"

2. Angular → POST /predict/breastCancer/laudo (FastAPI)
   └─ FastAPI chama _run_prediction()
      └─ Pipeline.predict([[880.5, 25.38, 0.2654, 0.3001]])
         └─ StandardScaler normaliza os dados
         └─ LogisticRegression classifica → "Maligno" (confiança: 94.7%)
   └─ FastAPI enfileira gerar_laudo_task.delay(...) no Redis
   └─ Retorna imediatamente: diagnóstico + task_id

3. Angular exibe o diagnóstico e inicia polling a cada 2s

4. Celery Worker pega a tarefa do Redis
   └─ Monta o prompt com os dados do exame e o diagnóstico
   └─ Chama Ollama (llama3.2:3b) via API compatível com OpenAI
   └─ LLM gera o laudo em Markdown (~30–60s)
   └─ Celery salva o laudo no Redis com o task_id

5. Polling do Angular detecta status = "completed"
   └─ Exibe o laudo renderizado como documento médico na interface
```
