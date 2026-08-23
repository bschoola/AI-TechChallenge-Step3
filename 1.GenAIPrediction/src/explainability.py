import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def shap_analysis(
    pipe_lr: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    scaler = pipe_lr.named_steps["scaler"]
    model = pipe_lr.named_steps["model"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values_test = explainer.shap_values(X_test_scaled)
    shap_values_train = explainer.shap_values(X_train_scaled)

    print("\nSHAP Summary — Test Set")
    shap.summary_plot(
        shap_values_test, X_test_scaled, feature_names=X_test.columns.tolist(), show=False
    )
    plt.tight_layout()
    plt.show()

    print("\nSHAP Summary — Train Set")
    shap.summary_plot(shap_values_train, X_train, show=False)
    plt.tight_layout()
    plt.show()


def feature_importance(pipe_lr: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coef = pipe_lr.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"variable": feature_names, "coefficient": coef})
    coef_df = coef_df.sort_values(by="coefficient", ascending=False).reset_index(drop=True)
    print("\nFeature Importance (Logistic Regression coefficients):")
    print(coef_df.to_string(index=False))
    return coef_df
