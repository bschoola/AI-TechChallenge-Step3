DATA_URL = (
    "https://raw.githubusercontent.com/bschoola/FIAP-Pos-AI"
    "/refs/heads/main/Data/data.csv"
)

COLUMN_RENAME_MAP = {
    "diagnosis": "diagnostico",
    "radius_mean": "raio_medio",
    "texture_mean": "textura_media",
    "perimeter_mean": "perimetro_medio",
    "area_mean": "area_media",
    "smoothness_mean": "suavidade_media",
    "compactness_mean": "compacidade_media",
    "concavity_mean": "concavidade_media",
    "concave points_mean": "pontos_concavos_media",
    "symmetry_mean": "simetria_media",
    "fractal_dimension_mean": "dimensao_fractal_media",
    "radius_se": "raio_erro_padrao",
    "texture_se": "textura_erro_padrao",
    "perimeter_se": "perimetro_erro_padrao",
    "area_se": "area_erro_padrao",
    "smoothness_se": "suavidade_erro_padrao",
    "compactness_se": "compacidade_erro_padrao",
    "concavity_se": "concavidade_erro_padrao",
    "concave points_se": "pontos_concavos_erro_padrao",
    "symmetry_se": "simetria_erro_padrao",
    "fractal_dimension_se": "dimensao_fractal_erro_padrao",
    "radius_worst": "raio_pior",
    "texture_worst": "textura_pior",
    "perimeter_worst": "perimetro_pior",
    "area_worst": "area_pior",
    "smoothness_worst": "suavidade_pior",
    "compactness_worst": "compacidade_pior",
    "concavity_worst": "concavidade_pior",
    "concave points_worst": "pontos_concavos_pior",
    "symmetry_worst": "simetria_pior",
    "fractal_dimension_worst": "dimensao_fractal_pior",
}

SELECTED_FEATURES = [
    "area_pior",
    "textura_pior",
    "pontos_concavos_pior",
    "concavidade_pior",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Espaço de busca para otimização por algoritmo genético
# Cada gene é um índice na lista correspondente
HP_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
HP_PENALTY_VALUES = ["l1", "l2"]
HP_MAX_ITER_VALUES = [100, 200, 500, 1000, 2000]
HP_CLASS_WEIGHT_VALUES = [None, "balanced"]

# Gene lengths for bounds checking: [len(C), len(penalty), len(max_iter), len(class_weight)]
HP_GENE_SIZES = [
    len(HP_C_VALUES),
    len(HP_PENALTY_VALUES),
    len(HP_MAX_ITER_VALUES),
    len(HP_CLASS_WEIGHT_VALUES),
]

# 3 experimentos com diferentes configurações do algoritmo genético
GA_EXPERIMENTS = [
    {
        "name": "Exp1 — Pop. Pequena / Mutação Alta",
        "population_size": 10,
        "generations": 20,
        "mutation_rate": 0.30,
        "crossover_rate": 0.80, # troca de cromossomos entre indivíduos
        "tournament_size": 3, # pega pessoas aleatorias para reproduzir mesmo que não seja vencedora
    },
    {
        "name": "Exp2 — Pop. Média / Mutação Média",
        "population_size": 20,
        "generations": 15,
        "mutation_rate": 0.20,
        "crossover_rate": 0.80,
        "tournament_size": 3,
    },
    {
        "name": "Exp3 — Pop. Grande / Mutação Baixa",
        "population_size": 30,
        "generations": 25,
        "mutation_rate": 0.10,
        "crossover_rate": 0.90,
        "tournament_size": 3,
    },
]
