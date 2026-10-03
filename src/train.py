"""
Improved training pipeline for the Target Marketing for Canadian Bank competition.

What it does:
1. Builds the features in src/features.py (optional K-Means clusters, fitted on
   training folds only).
2. Runs stratified K-fold cross-validation for each model and keeps the
   out-of-fold (OOF) probabilities, so every model is scored on the same rows.
3. Optionally tunes the models with Optuna (--tune), or reuses parameters saved
   by an earlier tuning run (--params-file).
4. Picks the probability threshold that maximises F1 on the OOF predictions,
   because the competition is scored with F1 rather than accuracy.
5. Combines the models four ways (mean, rank mean, weighted blend, stacking) and
   writes one submission per model and per ensemble, plus a lift table.

Usage:
    python -m src.train                                   # default models, 5 folds
    python -m src.train --tune --n-trials 50              # Optuna tuning first
    python -m src.train --params-file outputs/best_params.json --seeds 3
    python -m src.train --models lgbm catboost --no-clusters
"""
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from src.features import ClusterFeatures, ID_COLUMN, build_features, cluster_columns

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DATA_DIR = Path("../input/target-marketing-for-canadian-bank-2023w-aml1413")


# ---------------------------------------------------------------------------
# Models
# Each factory returns the estimator only; build_model() adds the cluster step.
# ---------------------------------------------------------------------------
def scaled(estimator, use_clusters: bool, quantile: bool = False):
    """
    Scale numeric columns (and one-hot encode cluster ids) for linear models and networks
    """
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=1000) if quantile else StandardScaler()
    if not use_clusters:
        return [("scale", scaler), ("model", estimator)]
    preprocess = ColumnTransformer(
        [("clusters", OneHotEncoder(handle_unknown="ignore"), cluster_columns())],
        remainder=scaler,
    )
    return [("preprocess", preprocess), ("model", estimator)]


def make_logreg(params, seed, n_jobs, use_clusters):
    return scaled(LogisticRegression(C=params.get("C", 0.5), max_iter=2000), use_clusters)


def make_mlp(params, seed, n_jobs, use_clusters):
    model = MLPClassifier(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
        alpha=params.get("alpha", 1e-3),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        batch_size=512, max_iter=200, early_stopping=True, n_iter_no_change=10,
        random_state=seed,
    )
    # Quantile scaling copes with the very heavy-tailed transaction columns
    return scaled(model, use_clusters, quantile=True)


def make_random_forest(params, seed, n_jobs, use_clusters):
    defaults = dict(n_estimators=500, min_samples_leaf=20, max_features=0.3)
    return [("model", RandomForestClassifier(**{**defaults, **params}, n_jobs=n_jobs, random_state=seed))]


def make_extra_trees(params, seed, n_jobs, use_clusters):
    defaults = dict(n_estimators=500, min_samples_leaf=20, max_features=0.5)
    return [("model", ExtraTreesClassifier(**{**defaults, **params}, n_jobs=n_jobs, random_state=seed))]


HIST_GB_DEFAULTS = dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)


def make_hist_gb(params, seed, n_jobs, use_clusters):
    return [("model", HistGradientBoostingClassifier(**{**HIST_GB_DEFAULTS, **params}, random_state=seed))]


LGBM_DEFAULTS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
)


def make_lgbm(params, seed, n_jobs, use_clusters):
    from lightgbm import LGBMClassifier

    return [("model", LGBMClassifier(**{**LGBM_DEFAULTS, **params}, random_state=seed, n_jobs=n_jobs, verbose=-1))]


XGB_DEFAULTS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
)


def make_xgb(params, seed, n_jobs, use_clusters):
    from xgboost import XGBClassifier

    return [("model", XGBClassifier(
        **{**XGB_DEFAULTS, **params}, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
    ))]


CATBOOST_DEFAULTS = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0)


def make_catboost(params, seed, n_jobs, use_clusters):
    from catboost import CatBoostClassifier

    return [("model", CatBoostClassifier(
        **{**CATBOOST_DEFAULTS, **params}, random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False,
    ))]


MODELS: Dict[str, Callable] = {
    "logreg": make_logreg,
    "mlp": make_mlp,
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "hist_gb": make_hist_gb,
    "lgbm": make_lgbm,
    "xgb": make_xgb,
    "catboost": make_catboost,
}
DEFAULT_MODELS = ["logreg", "mlp", "random_forest", "hist_gb", "lgbm", "xgb", "catboost"]


def build_model(name: str, params: dict, seed: int, n_jobs: int, use_clusters: bool) -> Pipeline:
    steps = MODELS[name](params, seed, n_jobs, use_clusters)
    if use_clusters:
        steps = [("clusters", ClusterFeatures(random_state=seed))] + steps
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Optuna search spaces
# The ranges include small, strongly regularised trees: on this data they did
# better than the deeper default trees.
# ---------------------------------------------------------------------------
def suggest_params(name: str, trial) -> dict:
    if name == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        )
    if name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        )
    if name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 3, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 50, log=True),
        )
    if name == "hist_gb":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 1500),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 1000, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 100, log=True),
        )
    if name == "logreg":
        return dict(C=trial.suggest_float("C", 1e-3, 10, log=True))
    raise ValueError(f"No search space for {name}")


TUNABLE = {"lgbm", "xgb", "catboost", "hist_gb", "logreg"}


def tune(name, X, y, n_trials, seed, n_jobs, use_clusters) -> dict:
    """
    Tune a model with Optuna, maximising 3-fold CV ROC AUC.
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
            model = build_model(name, params, seed, n_jobs, use_clusters)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(roc_auc_score(y.iloc[valid_idx], model.predict_proba(X.iloc[valid_idx])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{name}] best 3-fold AUC {study.best_value:.5f} with {study.best_params}", flush=True)
    return study.best_params


# ---------------------------------------------------------------------------
# Cross-validation, thresholds and metrics
# ---------------------------------------------------------------------------
def cross_validate(name, params, X, y, X_test, n_folds, seeds, n_jobs, use_clusters):
    """
    Out-of-fold probabilities for train, fold-averaged probabilities for test,
    and the AUC of each fold (to show how noisy the score is)
    """
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, valid_idx in skf.split(X, y):
            model = build_model(name, params, seed, n_jobs, use_clusters)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            valid_pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[valid_idx], valid_pred))
            oof[valid_idx] += valid_pred / len(seeds)
            test_pred += model.predict_proba(X_test)[:, 1] / (n_folds * len(seeds))
    return oof, test_pred, model, fold_aucs


def best_f1_threshold(y_true, proba) -> tuple:
    """
    Threshold that maximises F1 on the given predictions
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def evaluate(y_true, proba, fold_aucs=None) -> dict:
    threshold, best_f1 = best_f1_threshold(y_true, proba)
    metrics = {
        "auc": roc_auc_score(y_true, proba),
        "auc_fold_std": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "accuracy@0.5": accuracy_score(y_true, proba >= 0.5),
        "f1@0.5": f1_score(y_true, proba >= 0.5),
        "best_threshold": threshold,
        "f1@best": best_f1,
        "predicted_positive_rate@best": float(np.mean(proba >= threshold)),
    }
    return metrics


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """
    Responders found when contacting the top 10%, 20%, ... of customers ranked by probability
    """
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    rows = []
    for k in range(1, n_bins + 1):
        n = int(round(len(y_sorted) * k / n_bins))
        top = y_sorted[:n]
        rows.append({
            "contacted_share": k / n_bins,
            "response_rate": top.mean(),
            "share_of_all_responders_found": top.sum() / y_sorted.sum(),
            "lift": top.mean() / y_sorted.mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------
def to_logit(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def blend_weights(oof_matrix: np.ndarray, y) -> np.ndarray:
    """
    Non-negative weights (summing to 1) that minimise the log loss of the blended OOF probabilities
    """
    n = oof_matrix.shape[1]

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return log_loss(y, np.clip(oof_matrix @ w, 1e-6, 1 - 1e-6))

    result = minimize(loss, np.full(n, 1 / n), method="Nelder-Mead", options={"maxiter": 2000})
    return np.abs(result.x) / np.abs(result.x).sum()


def build_ensembles(oof_all: dict, test_all: dict, y, seed: int) -> dict:
    """
    Four ways to combine the models. Each returns (oof, test) probabilities.
    The stacking OOF score uses cross_val_predict, so it is not scored on rows it was fitted on.
    """
    names = list(oof_all)
    oof_m = np.column_stack([oof_all[m] for m in names])
    test_m = np.column_stack([test_all[m] for m in names])
    ensembles = {"ensemble_mean": (oof_m.mean(axis=1), test_m.mean(axis=1))}

    # Rank average: robust when models output differently calibrated probabilities.
    # Ranks are rescaled to 0..1, so the test ranks use the OOF distribution as reference.
    oof_rank = np.column_stack([rankdata(c) / len(c) for c in oof_m.T]).mean(axis=1)
    test_rank = np.column_stack(
        [np.searchsorted(np.sort(o), t) / len(o) for o, t in zip(oof_m.T, test_m.T)]
    ).mean(axis=1)
    ensembles["ensemble_rank_mean"] = (oof_rank, test_rank)

    weights = blend_weights(oof_m, y)
    print("  blend weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights)), flush=True)
    ensembles["ensemble_weighted"] = (oof_m @ weights, test_m @ weights)

    stacker = LogisticRegression(C=1.0, max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 100)
    stack_oof = cross_val_predict(stacker, to_logit(oof_m), y, cv=skf, method="predict_proba")[:, 1]
    stack_test = stacker.fit(to_logit(oof_m), y).predict_proba(to_logit(test_m))[:, 1]
    ensembles["ensemble_stacking"] = (stack_oof, stack_test)
    return ensembles


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
    parser.add_argument("--no-clusters", action="store_true", help="skip the K-Means cluster features")
    parser.add_argument("--tune", action="store_true", help=f"tune {sorted(TUNABLE)} with Optuna first")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--params-file", type=Path, default=None,
                        help="JSON with parameters per model, e.g. best_params.json from an earlier --tune run")
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
    use_clusters = not args.no_clusters
    saved_params = json.loads(args.params_file.read_text()) if args.params_file else {}

    df_train, df_test = load_data(args.data_dir)
    X, y, X_test, test_ids = build_features(df_train, df_test)
    print(f"Train {X.shape}, test {X_test.shape}, positive rate {y.mean():.3f}, clusters {use_clusters}", flush=True)

    results, oof_all, test_all, best_params = {}, {}, {}, dict(saved_params)

    # Reference point: F1 when every customer is predicted as a responder.
    results["baseline_all_positive"] = {"f1@0.5": f1_score(y, np.ones(len(y))), "auc": 0.5}

    for name in args.models:
        start = time.time()
        params = saved_params.get(name, {})
        if args.tune and name in TUNABLE:
            print(f"Tuning {name} ({args.n_trials} trials)...", flush=True)
            params = tune(name, X, y, args.n_trials, seeds[0], args.n_jobs, use_clusters)
            best_params[name] = params
        print(f"Cross-validating {name}...", flush=True)
        oof, test_pred, last_model, fold_aucs = cross_validate(
            name, params, X, y, X_test, args.folds, seeds, args.n_jobs, use_clusters)
        metrics = evaluate(y, oof, fold_aucs)
        metrics["minutes"] = (time.time() - start) / 60
        results[name], oof_all[name], test_all[name] = metrics, oof, test_pred
        print(f"  {name}: AUC {metrics['auc']:.4f} (fold std {metrics['auc_fold_std']:.4f}) | "
              f"F1@0.5 {metrics['f1@0.5']:.4f} | F1@{metrics['best_threshold']:.3f} {metrics['f1@best']:.4f}",
              flush=True)

        write_submission(args.output_dir / f"submission_{name}.csv", test_ids,
                         test_pred >= metrics["best_threshold"], args.id_header)

        estimator = last_model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            names = last_model[:-1].transform(X.head()).columns if len(last_model) > 1 else X.columns
            pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False) \
                .to_csv(args.output_dir / f"feature_importance_{name}.csv", header=["importance"])

    if len(args.models) > 1:
        for ens_name, (oof, test_pred) in build_ensembles(oof_all, test_all, y, seeds[0]).items():
            metrics = evaluate(y, oof)
            results[ens_name] = metrics
            print(f"  {ens_name}: AUC {metrics['auc']:.4f} | F1@0.5 {metrics['f1@0.5']:.4f} | "
                  f"F1@{metrics['best_threshold']:.3f} {metrics['f1@best']:.4f}", flush=True)
            write_submission(args.output_dir / f"submission_{ens_name}.csv",
                             test_ids, test_pred >= metrics["best_threshold"], args.id_header)
            oof_all[ens_name] = oof

    table = pd.DataFrame(results).T.sort_values("auc", ascending=False, na_position="last")
    table.to_csv(args.output_dir / "results.csv")

    best_name = table.drop(index="baseline_all_positive").index[0]
    lift_table(y, oof_all[best_name]).to_csv(args.output_dir / "lift.csv", index=False)

    pd.DataFrame({k: v for k, v in oof_all.items()}).assign(Target=y.values).to_csv(
        args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(test_all).assign(**{ID_COLUMN: test_ids.values}).to_csv(
        args.output_dir / "test_probabilities.csv", index=False)
    if best_params:
        (args.output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print("\nResults (out-of-fold, sorted by AUC):")
    print(table.round(4).to_string())
    print(f"\nLift table for {best_name} (saved to lift.csv):")
    print(lift_table(y, oof_all[best_name]).round(3).to_string(index=False))
    print(f"\nSubmissions and reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
