import random

import pandas as pd
from sklearn.model_selection import cross_val_score

from config import (
    CV_FOLDS,
    HP_C_VALUES,
    HP_CLASS_WEIGHT_VALUES,
    HP_GENE_SIZES,
    HP_MAX_ITER_VALUES,
    HP_PENALTY_VALUES,
    RANDOM_STATE,
)
from src.models import build_model_from_hyperparams

# Individual layout: [c_idx, penalty_idx, max_iter_idx, class_weight_idx]
_GENE_COUNT = len(HP_GENE_SIZES)


def decode_individual(individual: list[int]) -> dict:
    penalty = HP_PENALTY_VALUES[individual[1]]
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return {
        "C": HP_C_VALUES[individual[0]],
        "penalty": penalty,
        "solver": solver,
        "max_iter": HP_MAX_ITER_VALUES[individual[2]],
        "class_weight": HP_CLASS_WEIGHT_VALUES[individual[3]],
        "random_state": RANDOM_STATE,
    }


def _random_individual() -> list[int]:
    return [random.randrange(size) for size in HP_GENE_SIZES]


def _initialize_population(size: int) -> list[list[int]]:
    return [_random_individual() for _ in range(size)]


def _fitness(individual: list[int], X_train: pd.DataFrame, y_train: pd.Series) -> float:
    hyperparams = decode_individual(individual)
    model = build_model_from_hyperparams(hyperparams)
    scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    return float(scores.mean())


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
) -> list[int]:
    candidates = random.sample(range(len(population)), tournament_size)
    winner = max(candidates, key=lambda i: fitnesses[i])
    return population[winner][:]


def _crossover(
    parent1: list[int],
    parent2: list[int],
    crossover_rate: float,
) -> tuple[list[int], list[int]]:
    if random.random() > crossover_rate or _GENE_COUNT < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, _GENE_COUNT - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def _mutate(individual: list[int], mutation_rate: float) -> list[int]:
    return [
        random.randrange(HP_GENE_SIZES[i]) if random.random() < mutation_rate else gene
        for i, gene in enumerate(individual)
    ]


def run_ga(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple[dict, float, list[float]]:
    """Run the genetic algorithm and return (best_hyperparams, best_f1, history_per_generation)."""
    population_size: int = config["population_size"]
    generations: int = config["generations"]
    mutation_rate: float = config["mutation_rate"]
    crossover_rate: float = config["crossover_rate"]
    tournament_size: int = config["tournament_size"]

    population = _initialize_population(population_size)
    history: list[float] = []

    best_individual: list[int] = population[0]
    best_f1: float = 0.0

    for gen in range(1, generations + 1):
        fitnesses = [_fitness(ind, X_train, y_train) for ind in population]

        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_f1 = fitnesses[gen_best_idx]

        if gen_best_f1 > best_f1:
            best_f1 = gen_best_f1
            best_individual = population[gen_best_idx][:]

        history.append(best_f1)
        print(f"  Geração {gen:>{len(str(generations))}}/{generations} | Melhor F1: {best_f1:.4f}")

        # Build next generation preserving the current best (elitism)
        new_population: list[list[int]] = [best_individual[:]]

        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitnesses, tournament_size)
            p2 = _tournament_select(population, fitnesses, tournament_size)
            c1, c2 = _crossover(p1, p2, crossover_rate)
            new_population.append(_mutate(c1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(_mutate(c2, mutation_rate))

        population = new_population

    return decode_individual(best_individual), best_f1, history
