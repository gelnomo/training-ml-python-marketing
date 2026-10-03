"""
Improved training pipeline for the Target Marketing for Canadian Bank competition.

What it does:
1. Builds the features in src/features.py (K-Means fitted on training folds only).
2. Runs stratified K-fold cross-validation for each model and keeps the
   out-of-fold (OOF) probabilities, so every model is scored on the same rows.
3. Optionally tunes LightGBM, XGBoost and CatBoost with Optuna (--tune).
4. Picks the probability threshold that maximises F1 on the OOF predictions,
   because the competition is scored with F1 rather than accuracy.
5. Averages the models into an ensemble and writes one submission per model.

Usage:
    python -m src.train                          # default models, 5 folds
    python -m src.train --tune --n-trials 50     # Optuna tuning first
    python -m src.train --models lgbm catboost --seeds 3
"""
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import ClusterFeatures, ID_COLUMN, build_features, cluster_columns

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DATA_DIR = Path("../input/target-marketing-for-canadian-bank-2023w-aml1413")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_logreg(params: dict, seed: int, n_jobs: int):
    def make_preprocessor():
        return ColumnTransformer(
            [("clusters", OneHotEncoder(handle_unknown="ignore"), cluster_columns())],
            remainder=StandardScaler(),
        )

    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(C=params.get("C", 0.5), max_iter=2000)),
        ]
    )


def make_random_forest(params: dict, seed: int, n_jobs: int):
    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=20, max_features=0.3, n_jobs=n_jobs, random_state=seed,
            )),
        ]
    )


def make_extra_trees(params: dict, seed: int, n_jobs: int):
    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=20, max_features=0.5, n_jobs=n_jobs, random_state=seed,
            )),
        ]
    )


def make_hist_gb(params: dict, seed: int, n_jobs: int):
    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, random_state=seed,
            )),
        ]
    )


LGBM_DEFAULTS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
)


def make_lgbm(params: dict, seed: int, n_jobs: int):
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", LGBMClassifier(**{**LGBM_DEFAULTS, **params}, random_state=seed, n_jobs=n_jobs, verbose=-1)),
        ]
    )


XGB_DEFAULTS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
)


def make_xgb(params: dict, seed: int, n_jobs: int):
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", XGBClassifier(
                **{**XGB_DEFAULTS, **params}, tree_method="hist", eval_metric="logloss",
                random_state=seed, n_jobs=n_jobs,
            )),
        ]
    )


CATBOOST_DEFAULTS = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0)


def make_catboost(params: dict, seed: int, n_jobs: int):
    from catboost import CatBoostClassifier

    return Pipeline(
        [
            ("clusters", ClusterFeatures(random_state=seed)),
            ("model", CatBoostClassifier(
                **{**CATBOOST_DEFAULTS, **params}, random_seed=seed, thread_count=n_jobs,
                verbose=0, allow_writing_files=False,
            )),
        ]
    )


MODELS: Dict[str, Callable] = {
    "logreg": make_logreg,
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "hist_gb": make_hist_gb,
    "lgbm": make_lgbm,
    "xgb": make_xgb,
    "catboost": make_catboost,
}
DEFAULT_MODELS = ["logreg", "random_forest", "hist_gb", "lgbm", "xgb", "catboost"]


# ---------------------------------------------------------------------------
# Optuna search spaces (only for the boosting libraries)
# ---------------------------------------------------------------------------
def suggest_params(name: str, trial) -> dict:
    if name == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        )
    if name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        )
    if name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 4, 9),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        )
    raise ValueError(f"No search space for {name}")


TUNABLE = {"lgbm", "xgb", "catboost"}


def tune(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int, seed: int, n_jobs: int) -> dict:
    """
    Tune a boosting model with Optuna, maximising 3-fold CV ROC AUC.
    AUC is used instead of F1 because it does not depend on the threshold,
    which is chosen afterwards.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        params = suggest_params(name, trial)
        scores = []
        for train_idx, valid_idx in folds:
            model = MODELS[name](params, seed, n_jobs)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(roc_auc_score(y.iloc[valid_idx], model.predict_proba(X.iloc[valid_idx])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{name}] best CV AUC {study.best_value:.5f} with {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Cross-validation, thresholds and metrics
# ---------------------------------------------------------------------------
def cross_validate(name: str, params: dict, X, y, X_test, n_folds: int, seeds: List[int], n_jobs: int):
    """
    Out-of-fold probabilities for train and fold-averaged probabilities for test
    """
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, valid_idx in skf.split(X, y):
            model = MODELS[name](params, seed, n_jobs)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[valid_idx] += model.predict_proba(X.iloc[valid_idx])[:, 1] / len(seeds)
            test_pred += model.predict_proba(X_test)[:, 1] / (n_folds * len(seeds))
    return oof, test_pred, model


def best_f1_threshold(y_true, proba) -> tuple:
    """
    Threshold that maximises F1 on the given predictions
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def evaluate(y_true, proba) -> dict:
    threshold, best_f1 = best_f1_threshold(y_true, proba)
    return {
        "auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "accuracy@0.5": accuracy_score(y_true, proba >= 0.5),
        "f1@0.5": f1_score(y_true, proba >= 0.5),
        "best_threshold": threshold,
        "f1@best": best_f1,
        "predicted_positive_rate@best": float(np.mean(proba >= threshold)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_data(data_dir: Path):
    if data_dir is None:
        data_dir = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.is_dir() else ROOT / "data"
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def write_submission(path: Path, ids: pd.Series, labels: np.ndarray, id_header: str) -> None:
    pd.DataFrame({id_header: ids, "Target": labels.astype(int)}).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1, help="number of CV seeds to average")
    parser.add_argument("--tune", action="store_true", help="tune lgbm/xgb/catboost with Optuna first")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--id-header", default="Customer_id",
        help="ID column name in the submission. The original submissions used Customer_id; "
             "the competition page shows CustomerID.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42 + i for i in range(args.seeds)]

    df_train, df_test = load_data(args.data_dir)
    X, y, X_test, test_ids = build_features(df_train, df_test)
    print(f"Train {X.shape}, test {X_test.shape}, positive rate {y.mean():.3f}")

    results, oof_all, test_all, best_params = {}, {}, {}, {}

    # Reference point: F1 when every customer is predicted as a responder.
    results["baseline_all_positive"] = {"f1@0.5": f1_score(y, np.ones(len(y)))}

    for name in args.models:
        start = time.time()
        params = {}
        if args.tune and name in TUNABLE:
            print(f"Tuning {name} ({args.n_trials} trials)...")
            params = tune(name, X, y, args.n_trials, seeds[0], args.n_jobs)
            best_params[name] = params
        print(f"Cross-validating {name}...")
        oof, test_pred, last_model = cross_validate(name, params, X, y, X_test, args.folds, seeds, args.n_jobs)
        metrics = evaluate(y, oof)
        metrics["minutes"] = (time.time() - start) / 60
        results[name], oof_all[name], test_all[name] = metrics, oof, test_pred
        print(f"  {name}: AUC {metrics['auc']:.4f} | F1@0.5 {metrics['f1@0.5']:.4f} | "
              f"F1@{metrics['best_threshold']:.3f} {metrics['f1@best']:.4f}")

        write_submission(args.output_dir / f"submission_{name}.csv", test_ids,
                         test_pred >= metrics["best_threshold"], args.id_header)

        estimator = last_model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            names = last_model.named_steps["clusters"].transform(X.head()).columns
            pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False) \
                .to_csv(args.output_dir / f"feature_importance_{name}.csv", header=["importance"])

    # Ensemble: average of the models' probabilities
    if len(args.models) > 1:
        oof = np.mean([oof_all[m] for m in args.models], axis=0)
        test_pred = np.mean([test_all[m] for m in args.models], axis=0)
        metrics = evaluate(y, oof)
        results["ensemble_mean"] = metrics
        print(f"  ensemble_mean: AUC {metrics['auc']:.4f} | F1@0.5 {metrics['f1@0.5']:.4f} | "
              f"F1@{metrics['best_threshold']:.3f} {metrics['f1@best']:.4f}")
        write_submission(args.output_dir / "submission_ensemble.csv", test_ids,
                         test_pred >= metrics["best_threshold"], args.id_header)

    table = pd.DataFrame(results).T.sort_values("auc", ascending=False, na_position="last")
    table.to_csv(args.output_dir / "results.csv")
    pd.DataFrame(oof_all).assign(Target=y.values).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(test_all).assign(**{ID_COLUMN: test_ids.values}).to_csv(
        args.output_dir / "test_probabilities.csv", index=False)
    if best_params:
        (args.output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print("\nResults (out-of-fold, sorted by AUC):")
    print(table.round(4).to_string())
    print(f"\nSubmissions and reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
