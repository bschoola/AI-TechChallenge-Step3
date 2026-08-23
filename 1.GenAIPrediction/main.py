import io
import json
import sys
from datetime import datetime
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib

from config import GA_EXPERIMENTS, SELECTED_FEATURES
from src.data_loader import load_data
from src.evaluation import cross_validate_model, evaluate_model
from src.genetic_algorithm import run_ga
from src.models import build_baseline_model, build_model_from_hyperparams
from src.preprocessing import prepare_features, split_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ChoosenModel"


def _print_comparison_table(results: list[dict]) -> None:
    col_widths = [max(len(r["name"]) for r in results) + 2, 10, 10, 10, 10]
    header = (
        f"{'Modelo':<{col_widths[0]}}"
        f"{'F1-Score':>{col_widths[1]}}"
        f"{'Recall':>{col_widths[2]}}"
        f"{'Accuracy':>{col_widths[3]}}"
        f"{'ROC-AUC':>{col_widths[4]}}"
    )
    separator = "-" * len(header)
    print(f"\n{header}")
    print(separator)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['name']:<{col_widths[0]}}"
            f"{m['f1']:>{col_widths[1]}.4f}"
            f"{m['recall']:>{col_widths[2]}.4f}"
            f"{m['accuracy']:>{col_widths[3]}.4f}"
            f"{m['roc_auc']:>{col_widths[4]}.4f}"
        )
    best = max(results, key=lambda r: r["metrics"]["f1"])
    print(separator)
    print(f"Melhor F1-Score: {best['name']} ({best['metrics']['f1']:.4f})")
    return best


def _save_best_model(best: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "cancer_model.joblib"
    joblib.dump(best["model"], model_path)

    metadata = {
        "experiment": best["name"],
        "hyperparameters": best["hyperparams"],
        "features": SELECTED_FEATURES,
        "metrics": best["metrics"],
        "trained_at": datetime.now().isoformat(),
        "model_file": "cancer_model.joblib",
    }
    with open(OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Modelo salvo em:    {model_path}")
    print(f"  Metadados salvos em: {OUTPUT_DIR / 'model_metadata.json'}")


def main() -> None:
    print("=== Pipeline de Predição de Câncer ===")

    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    all_results: list[dict] = []

    # Baseline — Regressão Logística sem otimização
    print("\n=== Modelo Baseline (sem otimização) ===")
    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model("Regressão Logística — Baseline", baseline, X_test, y_test)
    cross_validate_model("Regressão Logística — Baseline", build_baseline_model(), X, y)
    all_results.append({"name": "Baseline LR", "metrics": baseline_metrics, "model": baseline, "hyperparams": {}})

    # Algoritmo Genético — 3 experimentos
    print("\n=== Algoritmo Genético — Otimização de Hiperparâmetros ===")
    for experiment in GA_EXPERIMENTS:
        print(
            f"\n--- {experiment['name']} "
            f"(pop={experiment['population_size']}, "
            f"ger={experiment['generations']}, "
            f"mut={experiment['mutation_rate']:.2f}) ---"
        )

        best_hyperparams, best_cv_f1, _ = run_ga(X_train, y_train, experiment)

        print(f"\n  Melhores hiperparâmetros encontrados:")
        for key, value in best_hyperparams.items():
            if key != "random_state":
                print(f"    {key}: {value}")
        print(f"  Melhor F1 (CV treino): {best_cv_f1:.4f}")

        optimized_model = build_model_from_hyperparams(best_hyperparams)
        optimized_model.fit(X_train, y_train)
        opt_metrics = evaluate_model(
            f"LR Otimizada ({experiment['name']})", optimized_model, X_test, y_test
        )
        all_results.append({"name": experiment["name"], "metrics": opt_metrics, "model": optimized_model, "hyperparams": best_hyperparams})

    # Tabela comparativa final
    print("\n=== Comparação Final ===")
    best = _print_comparison_table(all_results)

    # Salva o melhor modelo em ChoosenModel/
    print("\n=== Salvando Melhor Modelo ===")
    _save_best_model(best)


if __name__ == "__main__":
    main()
