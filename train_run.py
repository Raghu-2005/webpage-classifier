"""

What it does:
  1. Collects all features.json from output/ where extract_status=success
  2. Builds a training DataFrame — ONLY successful extractions
  3. Trains XGBoost with cross-validation + GridSearch
  4. Evaluates with classification report, confusion matrix
  5. Saves model, label encoder, feature list, and SHAP importance to models/
  6. Saves a training_report.json with full metadata
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import click
import joblib
import numpy as np
import pandas as pd
import shap
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import (
    console, get_logger, load_checkpoint, load_config, now_iso
)

warnings.filterwarnings("ignore")
logger = get_logger("train_run")
cfg = load_config()
MODEL_CFG = cfg["model"]


# Load training data 
def _purge_stale_features(output_root: Path, checkpoint: Dict) -> int:
    """
    Find and delete features.json files that belong to URLs where
    scrape_status != success OR extract_status != success.
    These files should NEVER be used for training.
    Returns count of files deleted.
    """
    deleted = 0
    for url, meta in checkpoint.items():
        label  = meta.get("label", "")
        folder = meta.get("folder", "")
        if not label or not folder:
            continue
        feat_path = output_root / label / folder / "features.json"
        if not feat_path.exists():
            continue
        scrape_ok  = meta.get("scrape_status")  == "success"
        extract_ok = meta.get("extract_status") == "success"
        if not scrape_ok or not extract_ok:
            feat_path.unlink()
            logger.warning(
                f"Deleted stale features.json for {url} "
                f"(scrape={meta.get('scrape_status')}, extract={meta.get('extract_status')})"
            )
            deleted += 1
    return deleted


def _load_training_data(output_root: Path, checkpoint: Dict) -> pd.DataFrame:
    """
    Walk checkpoint for success extractions, load their features.json.
    ONLY success scrape + success extract records are used.
    Automatically purges any stale features.json from failed URLs first.
    """
    # Purge stale files before loading — prevents contamination
    stale = _purge_stale_features(output_root, checkpoint)
    if stale:
        logger.warning(f"Purged {stale} stale features.json files from failed/skipped URLs")

    records: List[Dict] = []
    missing = 0
    skipped_failed = 0

    for url, meta in checkpoint.items():
        if meta.get("scrape_status") != "success":
            skipped_failed += 1
            continue
        if meta.get("extract_status") != "success":
            skipped_failed += 1
            continue

        label  = meta.get("label")
        folder = meta.get("folder")
        if not label or not folder:
            continue

        features_path = output_root / label / folder / "features.json"
        if not features_path.exists():
            missing += 1
            continue

        try:
            feat  = json.loads(features_path.read_text(encoding="utf-8"))
            clean = {k: v for k, v in feat.items() if not k.startswith("_")}
            clean["label"] = label
            clean["_url"] = url
            records.append(clean)
        except Exception as e:
            logger.warning(f"Could not read features for {url}: {e}")

    if missing:
        logger.warning(f"{missing} features.json files missing despite checkpoint mark")
    if skipped_failed:
        logger.info(f"Skipped {skipped_failed} failed/incomplete URLs (not used for training)")

    df = pd.DataFrame(records)
    logger.info(f"Training data shape: {df.shape} | Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str]]:
    """Prepare X, y, label encoder, and feature column list."""
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # Drop non-numeric / label columns
    drop_cols = {"label"}
    feature_cols = [c for c in df.columns
                    if c not in drop_cols
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()

    # Fill NaN with 0 (should be rare; extractor always returns values)
    X = X.fillna(0)

    logger.info(f"Feature columns: {len(feature_cols)}")
    return X, pd.Series(y), le, feature_cols


def _train_model(X_train, y_train) -> XGBClassifier:
    """Train XGBoost with cross-validation and light hyperparameter search."""
    base_params = {k: v for k, v in MODEL_CFG["xgb_params"].items()
                   if k not in ("use_label_encoder", "eval_metric")}
    base_params["eval_metric"] = MODEL_CFG["xgb_params"].get("eval_metric", "mlogloss")
    base_params["random_state"] = MODEL_CFG["random_state"]
    base_params["verbosity"] = 0

    # Param grid for GridSearchCV
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=MODEL_CFG["random_state"],
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=MODEL_CFG["cv_folds"], shuffle=True,
                         random_state=MODEL_CFG["random_state"])

    logger.info("Starting GridSearchCV — this may take a few minutes...")
    grid = GridSearchCV(
        base_clf, param_grid, cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV F1-macro: {grid.best_score_:.4f}")
    return grid.best_estimator_


def _evaluate(model, X_test, y_test, le: LabelEncoder) -> Dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}


def _shap_importance(model, X_test, feature_cols: List[str]) -> List[Dict]:
    """
    Compute SHAP feature importance (mean |SHAP value| across test set).
    Handles both old shap API (list of arrays) and new API (3-D ndarray).
    """
    try:
        explainer = shap.TreeExplainer(model)
        sample = X_test[:200]
        shap_values = explainer.shap_values(sample)

        # shap_values shape variants:
        sv = np.array(shap_values)

        if sv.ndim == 3:
            # shape: (n_samples, n_features, n_classes)  — new API multi-class
            mean_abs = np.abs(sv).mean(axis=0).mean(axis=1)   # → (n_features,)
        elif sv.ndim == 2 and sv.shape[0] == len(feature_cols):
            # shape: (n_features, n_samples) — some versions return transposed
            mean_abs = np.abs(sv).mean(axis=1)
        elif sv.ndim == 2:
            # shape: (n_samples, n_features) — binary or single-output
            mean_abs = np.abs(sv).mean(axis=0)
        else:
            # list of 2-D arrays — old multi-class API
            arrays = [np.abs(s).mean(axis=0) for s in shap_values]
            mean_abs = np.mean(arrays, axis=0)

        # Align length with feature_cols (safety guard)
        mean_abs = np.array(mean_abs).flatten()
        if len(mean_abs) != len(feature_cols):
            raise ValueError(
                f"SHAP output length {len(mean_abs)} != feature_cols length {len(feature_cols)}"
            )

        importance = sorted(
            [{"feature": f, "importance": round(float(v), 6)}
             for f, v in zip(feature_cols, mean_abs)],
            key=lambda x: x["importance"], reverse=True
        )
        logger.info("SHAP feature importance computed successfully.")
        return importance

    except Exception as e:
        logger.warning(f"SHAP failed: {e}. Using XGBoost built-in importance instead.")
        scores = model.get_booster().get_score(importance_type="gain")
        if not scores:
            scores = model.get_booster().get_fscore()
        total = sum(scores.values()) or 1
        # Fill missing features with 0
        all_importance = {f: 0.0 for f in feature_cols}
        for k, v in scores.items():
            if k in all_importance:
                all_importance[k] = round(v / total, 6)
        return sorted(
            [{"feature": k, "importance": v} for k, v in all_importance.items()],
            key=lambda x: x["importance"], reverse=True
        )


@click.command()
@click.option("--no-gridsearch", is_flag=True, help="Skip GridSearchCV and use default params")
@click.option("--show-results", is_flag=True, help="Print saved training results without retraining")
def main(no_gridsearch: bool, show_results: bool):
    """Train the webpage classifier model."""
    output_root = Path(cfg["paths"]["output_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Show saved results without retraining 
    if show_results:
        report_path = models_dir / "training_report.json"
        if not report_path.exists():
            console.print("[red]No training report found. Run python train_run.py first.")
            sys.exit(1)
        saved = json.loads(report_path.read_text())
        report = saved["evaluation"]["classification_report"]
        cm = saved["evaluation"]["confusion_matrix"]
        classes = saved["classes"]
        print("\n" + "=" * 60)
        print("  SAVED TRAINING RESULTS")
        print(f"  Trained at: {saved.get('trained_at', '?')[:19]}")
        print(f"  Samples — train: {saved['num_training_samples']} | test: {saved['num_test_samples']}")
        print("=" * 60)
        print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*54}")
        for cls in classes:
            r = report.get(cls, {})
            print(f"  {cls:<12} {r.get('precision',0):>10.3f} {r.get('recall',0):>10.3f} {r.get('f1-score',0):>10.3f} {int(r.get('support',0)):>10}")
        ma = report.get("macro avg", {})
        print(f"  {'-'*54}")
        print(f"  {'macro avg':<12} {ma.get('precision',0):>10.3f} {ma.get('recall',0):>10.3f} {ma.get('f1-score',0):>10.3f} {'—':>10}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>12}" + "".join(f"{c:>10}" for c in classes))
        print(f"  {'-' * (12 + 10 * len(classes))}")
        for i, cls in enumerate(classes):
            print(f"  {cls:>12}" + "".join(f"{cm[i][j]:>10}" for j in range(len(classes))))
        print("\n  Top 10 features:")
        for rank, item in enumerate(saved.get("top_features", [])[:10], 1):
            print(f"  {rank:2}. {item['feature']:<40} {item['importance']:.6f}")
        print("=" * 60 + "\n")
        return

    console.rule("[bold blue]Webpage Classifier — Train Step")
    checkpoint = load_checkpoint()

    # Load data
    df = _load_training_data(output_root, checkpoint)

    if df.empty:
        logger.error("No training data found. Run scrape_run.py and extract_run.py first.")
        sys.exit(1)

    min_samples = 3
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        if count < min_samples:
            logger.error(
                f"Class '{label}' has only {count} sample(s). "
                f"Minimum {min_samples} required per class."
            )
            sys.exit(1)

    # Prepare
    X, y, le, feature_cols = _prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CFG["test_size"],
        random_state=MODEL_CFG["random_state"],
        stratify=y,
    )

    console.print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train 
    if no_gridsearch:
        params = {k: v for k, v in MODEL_CFG["xgb_params"].items()
                  if k not in ("use_label_encoder",)}
        params["random_state"] = MODEL_CFG["random_state"]
        params["verbosity"] = 0
        model = XGBClassifier(use_label_encoder=False, **params)
        model.fit(X_train, y_train)
    else:
        model = _train_model(X_train, y_train)

    # Evaluate
    eval_results = _evaluate(model, X_test, y_test, le)
    acc = eval_results["accuracy"]
    print(f"\n  Accuracy: {acc*100:.2f}%")
    report = eval_results["classification_report"]
    cm = eval_results["confusion_matrix"]

    # FULL MISCLASSIFICATION ANALYSIS
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Decode labels
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # Get URLs (IMPORTANT: requires _url in df)
    if "_url" in df.columns:
        urls = df.loc[X_test.index, "_url"]
    else:
        urls = pd.Series(["NA"] * len(X_test), index=X_test.index)

    results = []

    for i, idx in enumerate(X_test.index):
        actual = y_test_labels[i]
        pred = y_pred_labels[i]
        url = urls.loc[idx]
        conf = float(np.max(y_prob[i]))

        results.append({
            "url": url,
            "actual": actual,
            "predicted": pred,
            "confidence": round(conf, 4)
        })

    df_results = pd.DataFrame(results)

    # All misclassified 
    mis_all = df_results[df_results["actual"] != df_results["predicted"]]

    print("\n===== ALL MISCLASSIFIED URLS =====")
    print(mis_all)
    print("Total test samples:", len(df_results))
    print("Total misclassified:", len(mis_all))
    # Class-wise breakdown
    mis_list = df_results[(df_results["actual"] == "list") & (df_results["predicted"] != "list")]
    mis_detail = df_results[(df_results["actual"] == "detail") & (df_results["predicted"] != "detail")]
    mis_others = df_results[(df_results["actual"] == "others") & (df_results["predicted"] != "others")]

    print("\n===== LIST → WRONG =====")
    print(mis_list)

    print("\n===== DETAIL → WRONG =====")
    print(mis_detail)

    print("\n===== OTHERS → WRONG =====")
    print(mis_others)

    # Save all outputs
    mis_all.to_csv("misclassified_all.csv", index=False)
    mis_list.to_csv("misclassified_list.csv", index=False)
    mis_detail.to_csv("misclassified_detail.csv", index=False)
    mis_others.to_csv("misclassified_others.csv", index=False)

    # Always print plain-text results to stdout FIRST (never gets lost)
    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")
    for cls in le.classes_:
        r = report.get(cls, {})
        print(
            f"  {cls:<12}"
            f" {r.get('precision', 0):>10.3f}"
            f" {r.get('recall', 0):>10.3f}"
            f" {r.get('f1-score', 0):>10.3f}"
            f" {int(r.get('support', 0)):>10}"
        )
    ma = report.get("macro avg", {})
    print(f"  {'-'*54}")
    print(
        f"  {'macro avg':<12}"
        f" {ma.get('precision', 0):>10.3f}"
        f" {ma.get('recall', 0):>10.3f}"
        f" {ma.get('f1-score', 0):>10.3f}"
        f" {'—':>10}"
    )
    wa = report.get("weighted avg", {})
    print(
        f"  {'weighted avg':<12}"
        f" {wa.get('precision', 0):>10.3f}"
        f" {wa.get('recall', 0):>10.3f}"
        f" {wa.get('f1-score', 0):>10.3f}"
        f" {'—':>10}"
    )

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = f"  {'':>12}" + "".join(f"{cls:>10}" for cls in le.classes_)
    print(header)
    print(f"  {'-' * (12 + 10 * len(le.classes_))}")
    for i, cls in enumerate(le.classes_):
        row_vals = "".join(f"{cm[i][j]:>10}" for j in range(len(le.classes_)))
        print(f"  {cls:>12}{row_vals}")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    table = Table(title="Classification Report", show_header=True)
    table.add_column("Class", style="bold")
    table.add_column("Precision", style="cyan")
    table.add_column("Recall", style="cyan")
    table.add_column("F1", style="green")
    table.add_column("Support", style="yellow")
    for cls in le.classes_:
        r = report.get(cls, {})
        table.add_row(
            cls,
            f"{r.get('precision', 0):.3f}",
            f"{r.get('recall', 0):.3f}",
            f"{r.get('f1-score', 0):.3f}",
            str(int(r.get('support', 0))),
        )
    overall = report.get("macro avg", {})
    table.add_row(
        "[bold]macro avg[/bold]",
        f"{overall.get('precision', 0):.3f}",
        f"{overall.get('recall', 0):.3f}",
        f"{overall.get('f1-score', 0):.3f}",
        "—",
    )
    console.print(table)

    console.print("\nConfusion Matrix (rows=actual, cols=predicted):")
    cm_table = Table(show_header=True)
    cm_table.add_column("Actual \\ Pred")
    for cls in le.classes_:
        cm_table.add_column(cls, style="cyan")
    for i, cls in enumerate(le.classes_):
        row = [cls] + [str(cm[i][j]) for j in range(len(le.classes_))]
        cm_table.add_row(*row)
    console.print(cm_table)

    # SHAP importance 
    importance = _shap_importance(model, X_test, feature_cols)
    console.print("\n[bold]Top 20 features by SHAP importance:[/bold]")
    imp_table = Table(show_header=True)
    imp_table.add_column("Rank")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("SHAP Importance", style="green")
    for rank, item in enumerate(importance[:20], 1):
        imp_table.add_row(str(rank), item["feature"], f"{item['importance']:.6f}")
    console.print(imp_table)

    # Save artefacts
    model_path = models_dir / "classifier.joblib"
    le_path = models_dir / "label_encoder.joblib"
    features_path = models_dir / "feature_columns.json"
    report_path = models_dir / "training_report.json"
    importance_path = models_dir / "feature_importance.json"

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    features_path.write_text(json.dumps(feature_cols, indent=2))
    importance_path.write_text(json.dumps(importance, indent=2))

    training_report = {
        "trained_at": now_iso(),
        "num_training_samples": len(X_train),
        "num_test_samples": len(X_test),
        "num_features": len(feature_cols),
        "classes": list(le.classes_),
        "label_distribution": df["label"].value_counts().to_dict(),
        "evaluation": eval_results,
        "model_params": model.get_params(),
        "top_features": importance[:30],
    }
    report_path.write_text(json.dumps(training_report, indent=2))

    console.print(f"\n[green]✓ Model saved → {model_path}")
    console.print(f"[green]✓ Training report → {report_path}")
    console.print("\n[bold]Run [cyan]python predict_run.py --url <URL>[/cyan] to classify a webpage.")

if __name__ == "__main__":
    main()