"""
eval_pipeline.py
────────────────
Evaluates and compares THREE accuracy modes on your already-scraped test data:

  Mode A — ML model ONLY (baseline)
  Mode B — Pre-classifier ONLY  (how many rules fire, are they correct?)
  Mode C — Full pipeline: Pre-classifier → ML fallback  (production accuracy)

Usage:
    python eval_pipeline.py
    python eval_pipeline.py --split 0.2          # 20% test split (default)
    python eval_pipeline.py --split 0.3          # 30% for larger test set
    python eval_pipeline.py --verbose            # show every misclassified URL
    python eval_pipeline.py --export results.csv # export detailed results

What it does:
  1. Loads ALL successfully scraped+extracted URLs from output/<label>/<folder>/
     using the checkpoint (same data your model was trained on)
  2. Holds out a stratified test split (never seen during training)
  3. For each test URL it runs:
       a. ML model alone (features.json → XGBoost)
       b. Pre-classifier alone (raw.html → deterministic rules)
       c. Full pipeline (pre-classifier first; ML if pre-classifier returns None)
  4. Prints a side-by-side comparison table + per-class breakdown
  5. Lists every URL where pre-classifier and ML disagree

Run from your project root:
    source venv/bin/activate
    python eval_pipeline.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Make sure project modules are importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import load_checkpoint, load_config
from extraction.feature_extractor import extract_features, get_feature_columns
from extraction.pre_classifier import pre_classify

console = Console()
cfg = load_config()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_samples(output_root: Path, checkpoint: Dict) -> List[Dict]:
    """
    Load every URL that has:
      - scrape_status  = success
      - extract_status = success
      - raw.html       exists
      - features.json  exists

    Returns list of dicts:
      { url, label, folder, html, features, raw_html_path }
    """
    samples = []
    missing_html = 0
    missing_feat = 0

    for url, meta in checkpoint.items():
        if meta.get("scrape_status") != "success":
            continue
        if meta.get("extract_status") != "success":
            continue

        label  = meta.get("label")
        folder = meta.get("folder")
        if not label or not folder:
            continue

        folder_path   = output_root / label / folder
        html_path     = folder_path / "raw.html"
        features_path = folder_path / "features.json"

        if not html_path.exists():
            missing_html += 1
            continue
        if not features_path.exists():
            missing_feat += 1
            continue

        try:
            features = json.loads(features_path.read_text(encoding="utf-8"))
            samples.append({
                "url":          url,
                "label":        label,
                "folder":       folder,
                "html_path":    html_path,
                "features":     features,
            })
        except Exception as e:
            console.print(f"[yellow]  Warning: could not load {url}: {e}[/yellow]")

    if missing_html:
        console.print(f"[yellow]  {missing_html} URLs skipped — missing raw.html[/yellow]")
    if missing_feat:
        console.print(f"[yellow]  {missing_feat} URLs skipped — missing features.json[/yellow]")

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def predict_ml_only(
    features: Dict,
    model,
    le: LabelEncoder,
    feature_cols: List[str],
) -> Tuple[str, float]:
    """Run ML model on pre-computed features. Returns (label, confidence)."""
    row   = {col: features.get(col, 0) for col in feature_cols}
    X     = pd.DataFrame([row]).fillna(0)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    return le.inverse_transform([idx])[0], float(proba[idx])


def predict_pre_classifier_only(html: str, url: str) -> Optional[Tuple[str, float, str]]:
    """
    Run ONLY the pre-classifier (no ML fallback).
    Returns (label, confidence, method) or None if no rule fired.
    """
    result = pre_classify(html=html, url=url, http_status=None)
    if result:
        return result["label"], result["confidence"], result["method"]
    return None


def predict_full_pipeline(
    html: str,
    url: str,
    features: Dict,
    model,
    le: LabelEncoder,
    feature_cols: List[str],
) -> Tuple[str, float, str]:
    """
    Production pipeline:
      1. Try pre-classifier
      2. If no rule fired → ML model
    Returns (label, confidence, method)
    """
    pre = pre_classify(html=html, url=url, http_status=None)
    if pre:
        return pre["label"], pre["confidence"], pre["method"]

    # ML fallback
    lbl, conf = predict_ml_only(features, model, le, feature_cols)
    return lbl, conf, "ml_model"


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: List[str], y_pred: List[str], classes: List[str]) -> Dict:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / max(len(y_true), 1)
    report   = classification_report(
        y_true, y_pred, labels=classes, target_names=classes,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "n": len(y_true)}


def print_comparison_table(
    ml_metrics: Dict,
    pre_metrics: Dict,
    full_metrics: Dict,
    classes: List[str],
) -> None:
    console.print()
    console.rule("[bold cyan]Pipeline Accuracy Comparison")

    # ── Overall summary ───────────────────────────────────────────────────────
    summary = Table(title="Overall Accuracy", show_header=True, expand=False)
    summary.add_column("Mode",            style="bold", min_width=35)
    summary.add_column("Accuracy",        style="cyan",  min_width=10, justify="right")
    summary.add_column("Macro F1",        style="green", min_width=10, justify="right")
    summary.add_column("Correct / Total", style="white", min_width=16, justify="right")

    for name, m in [
        ("A — ML Model Only (baseline)",         ml_metrics),
        ("B — Pre-classifier Only (rules fired)", pre_metrics),
        ("C — Full Pipeline (pre + ML)",          full_metrics),
    ]:
        ma  = m["report"].get("macro avg", {})
        acc = m["accuracy"]
        n   = m["n"]
        correct = round(acc * n)
        f1  = ma.get("f1-score", 0)

        color = "green" if name.startswith("C") else ("yellow" if name.startswith("B") else "white")
        summary.add_row(
            f"[{color}]{name}[/{color}]",
            f"[{color}]{acc:.2%}[/{color}]",
            f"[{color}]{f1:.4f}[/{color}]",
            f"{correct} / {n}",
        )
    console.print(summary)

    # ── Per-class breakdown ───────────────────────────────────────────────────
    console.print()
    per_class = Table(title="Per-Class F1 Score", show_header=True, expand=False)
    per_class.add_column("Class",        style="bold",  min_width=10)
    per_class.add_column("ML Only",      style="cyan",  min_width=12, justify="right")
    per_class.add_column("Pre-cls Only", style="yellow",min_width=12, justify="right")
    per_class.add_column("Full Pipeline",style="green", min_width=14, justify="right")
    per_class.add_column("Support",      style="white", min_width=10, justify="right")

    for cls in classes:
        ml_f1   = ml_metrics["report"].get(cls, {}).get("f1-score", 0)
        pre_f1  = pre_metrics["report"].get(cls, {}).get("f1-score", 0)
        full_f1 = full_metrics["report"].get(cls, {}).get("f1-score", 0)
        sup     = ml_metrics["report"].get(cls, {}).get("support", 0)
        per_class.add_row(
            cls,
            f"{ml_f1:.4f}",
            f"{pre_f1:.4f}",
            f"[bold green]{full_f1:.4f}[/bold green]",
            str(int(sup)),
        )
    console.print(per_class)

    # ── Confusion matrices ────────────────────────────────────────────────────
    for label, m in [("ML Only", ml_metrics), ("Full Pipeline", full_metrics)]:
        cm = m["confusion_matrix"]
        cm_table = Table(title=f"Confusion Matrix — {label} (rows=actual, cols=predicted)", show_header=True)
        cm_table.add_column("Actual \\ Pred")
        for c in classes:
            cm_table.add_column(c, justify="right", style="cyan")
        for i, cls in enumerate(classes):
            cm_table.add_row(cls, *[str(cm[i][j]) for j in range(len(classes))])
        console.print(cm_table)
        console.print()


def print_pre_classifier_breakdown(results: List[Dict], classes: List[str]) -> None:
    """Show which pre-classifier rules fired and their accuracy."""
    console.rule("[bold cyan]Pre-Classifier Rule Breakdown")

    method_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        pre_method = r.get("pre_method")
        if pre_method and pre_method != "no_rule":
            method_stats[pre_method]["total"]   += 1
            method_stats[pre_method]["correct"] += int(r["pre_label"] == r["true_label"])

    fired_total   = sum(s["total"]   for s in method_stats.values())
    fired_correct = sum(s["correct"] for s in method_stats.values())
    n_total       = len(results)

    table = Table(title=f"Rules fired on {fired_total}/{n_total} samples ({fired_total/n_total:.1%})",
                  show_header=True, expand=False)
    table.add_column("Rule / Method",  style="bold", min_width=30)
    table.add_column("Fired",          style="cyan",  min_width=8,  justify="right")
    table.add_column("Correct",        style="green", min_width=10, justify="right")
    table.add_column("Accuracy",       style="white", min_width=10, justify="right")

    for method, stats in sorted(method_stats.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / max(stats["total"], 1)
        color = "green" if acc >= 0.95 else ("yellow" if acc >= 0.85 else "red")
        table.add_row(
            method,
            str(stats["total"]),
            str(stats["correct"]),
            f"[{color}]{acc:.1%}[/{color}]",
        )

    # Totals row
    overall_acc = fired_correct / max(fired_total, 1)
    table.add_row(
        "[bold]TOTAL (rules fired)[/bold]",
        str(fired_total),
        str(fired_correct),
        f"[bold green]{overall_acc:.1%}[/bold green]",
    )
    console.print(table)

    # URLs where pre-classifier was WRONG
    pre_wrong = [r for r in results
                 if r.get("pre_method") and r.get("pre_method") != "no_rule"
                 and r["pre_label"] != r["true_label"]]

    if pre_wrong:
        console.print(f"\n[red bold]Pre-classifier ERRORS ({len(pre_wrong)} URLs):[/red bold]")
        err_table = Table(show_header=True, expand=True)
        err_table.add_column("URL",       style="cyan", max_width=55)
        err_table.add_column("True",      style="green", min_width=8)
        err_table.add_column("Predicted", style="red",   min_width=10)
        err_table.add_column("Method",    style="yellow",min_width=20)
        err_table.add_column("Conf.",     style="white", min_width=8, justify="right")
        for r in pre_wrong:
            err_table.add_row(
                r["url"][:55],
                r["true_label"],
                r["pre_label"],
                r["pre_method"],
                f"{r['pre_conf']:.1%}",
            )
        console.print(err_table)
    else:
        console.print("\n[green]✓ Pre-classifier made ZERO errors on fired rules![/green]")


def print_disagreements(results: List[Dict]) -> None:
    """Show URLs where pre-classifier and ML disagree."""
    disagree = [
        r for r in results
        if r.get("pre_method") and r["pre_method"] != "no_rule"
        and r["pre_label"] != r["ml_label"]
    ]

    if not disagree:
        console.print("[green]✓ No disagreements between pre-classifier and ML model.[/green]")
        return

    console.rule(f"[bold yellow]Pre-classifier vs ML Disagreements ({len(disagree)} URLs)")
    table = Table(show_header=True, expand=True)
    table.add_column("URL",       style="cyan",   max_width=50)
    table.add_column("True",      style="green",  min_width=8)
    table.add_column("Pre-cls",   style="yellow", min_width=10)
    table.add_column("ML",        style="blue",   min_width=10)
    table.add_column("Full",      style="bold",   min_width=10)
    table.add_column("Pre Method",style="white",  min_width=18)

    for r in sorted(disagree, key=lambda x: x["true_label"]):
        pre_ok  = "✓" if r["pre_label"] == r["true_label"] else "✗"
        ml_ok   = "✓" if r["ml_label"]  == r["true_label"] else "✗"
        full_ok = "✓" if r["full_label"] == r["true_label"] else "✗"
        table.add_row(
            r["url"][:50],
            r["true_label"],
            f"{pre_ok} {r['pre_label']}",
            f"{ml_ok} {r['ml_label']}",
            f"{full_ok} {r['full_label']}",
            r["pre_method"],
        )
    console.print(table)


def print_verbose_misclassified(results: List[Dict], mode: str = "full") -> None:
    """Print all misclassified URLs for a given mode."""
    key = {"ml": "ml_label", "full": "full_label"}[mode]
    wrong = [r for r in results if r[key] != r["true_label"]]

    label_map = {"ml": "ML Only", "full": "Full Pipeline"}
    console.rule(f"[bold red]All Misclassified — {label_map[mode]} ({len(wrong)} URLs)")

    if not wrong:
        console.print(f"[green]✓ Zero errors in {label_map[mode]} mode![/green]")
        return

    table = Table(show_header=True, expand=True)
    table.add_column("URL",       style="cyan",  max_width=55)
    table.add_column("True",      style="green", min_width=8)
    table.add_column("Predicted", style="red",   min_width=10)
    table.add_column("Conf.",     style="white", min_width=8, justify="right")
    table.add_column("Method",    style="yellow",min_width=20)

    conf_key   = {"ml": "ml_conf",   "full": "full_conf"}[mode]
    method_key = {"ml": "ml_method", "full": "full_method"}[mode]

    for r in sorted(wrong, key=lambda x: x["true_label"]):
        table.add_row(
            r["url"][:55],
            r["true_label"],
            r[key],
            f"{r[conf_key]:.1%}",
            r.get(method_key, "ml_model"),
        )
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@click.command()
@click.option("--split",   default=None,  help="Test split fraction (default: reads from settings.yaml)")
@click.option("--seed",    default=None,  type=int, help="Random seed (default: reads from settings.yaml)")
@click.option("--verbose", is_flag=True,  help="Show all misclassified URLs for both modes")
@click.option("--export",  default=None,  help="Export detailed per-URL results to CSV")
@click.option("--no-html", is_flag=True,  help="Skip pre-classifier (only report ML accuracy)")
def main(split, seed, verbose: bool, export: Optional[str], no_html: bool):
    """
    Evaluate ML-only vs Full Pipeline (pre-classifier + ML) accuracy.
    Uses your existing scraped data — no new scraping needed.

    IMPORTANT: Uses the SAME split + seed as train_run.py (from settings.yaml)
    so the test set is truly held-out data the model never saw during training.
    """
    # ── Read split/seed from settings.yaml to stay in sync with train_run.py ──
    # This is CRITICAL — using a different seed would leak training data into
    # the test set and give fake inflated accuracy numbers.
    cfg_split = cfg["model"]["test_size"]      # 0.2
    cfg_seed  = cfg["model"]["random_state"]   # 42

    split = float(split) if split is not None else cfg_split
    seed  = int(seed)    if seed  is not None else cfg_seed

    console.rule("[bold blue]Webpage Classifier — Pipeline Evaluation")
    console.print(f"  [dim]Split: {split:.0%}  |  Seed: {seed}  "
                  f"(matched to settings.yaml → same test set as train_run.py)[/dim]")

    output_root = Path(cfg["paths"]["output_dir"])
    models_dir  = Path(cfg["paths"]["models_dir"])

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = models_dir / "classifier.joblib"
    le_path    = models_dir / "label_encoder.joblib"
    fc_path    = models_dir / "feature_columns.json"

    if not model_path.exists():
        console.print("[red]Model not found. Run: python train_run.py first.[/red]")
        sys.exit(1)

    console.print("[dim]Loading model artefacts...[/dim]")
    model        = joblib.load(model_path)
    le           = joblib.load(le_path)
    feature_cols = json.loads(fc_path.read_text())
    classes      = list(le.classes_)
    console.print(f"  Classes: [cyan]{classes}[/cyan]")
    console.print(f"  Features: [cyan]{len(feature_cols)}[/cyan]")

    # ── Cross-check: confirm training report seed matches ────────────────────
    report_path = models_dir / 'training_report.json'
    if report_path.exists():
        saved        = json.loads(report_path.read_text())
        saved_random = saved.get('model_params', {}).get('random_state', None)
        saved_acc    = saved.get('evaluation', {}).get('accuracy', None)
        saved_train  = saved.get('num_training_samples', '?')
        saved_test   = saved.get('num_test_samples', '?')
        console.print()
        console.print('  [dim]Saved training report:[/dim]')
        console.print(f"    Trained at:    {saved.get('trained_at', '?')[:19]}")
        console.print(f"    Train samples: {saved_train}  |  Test: {saved_test}")
        if saved_acc:
            console.print(f'    Saved ML accuracy: [cyan]{saved_acc:.2%}[/cyan]')
        if saved_random is not None and saved_random != seed:
            console.print(f'  [red bold]WARNING: model trained with random_state={saved_random} but eval uses seed={seed}[/red bold]')
            console.print('  [red]  The test sets DIFFER — accuracy will be INFLATED. Run with --seed ' + str(saved_random) + '[/red]')
        else:
            console.print(f'  [green]  Seed matches ({seed}) — test set is genuinely held-out data[/green]')



    # ── Load data ─────────────────────────────────────────────────────────────
    console.print("\n[dim]Loading samples from checkpoint...[/dim]")
    checkpoint = load_checkpoint()
    all_samples = load_all_samples(output_root, checkpoint)

    if not all_samples:
        console.print("[red]No samples found. Run scrape_run.py + extract_run.py first.[/red]")
        sys.exit(1)

    console.print(f"  Total samples loaded: [cyan]{len(all_samples)}[/cyan]")

    # Label distribution
    label_dist = defaultdict(int)
    for s in all_samples: label_dist[s["label"]] += 1
    for lbl, cnt in sorted(label_dist.items()):
        console.print(f"    {lbl}: {cnt}")

    # ── Stratified train/test split ───────────────────────────────────────────
    # Use the SAME split logic as train_run.py so we test on truly held-out data
    urls    = [s["url"]   for s in all_samples]
    labels  = [s["label"] for s in all_samples]

    _, test_urls, _, test_labels = train_test_split(
        urls, labels,
        test_size=split,
        random_state=seed,
        stratify=labels,
    )
    test_set_urls = set(test_urls)

    test_samples = [s for s in all_samples if s["url"] in test_set_urls]
    console.print(f"\n  Test samples (split={split:.0%}): [cyan]{len(test_samples)}[/cyan]")

    test_dist = defaultdict(int)
    for s in test_samples: test_dist[s["label"]] += 1
    for lbl, cnt in sorted(test_dist.items()):
        console.print(f"    {lbl}: {cnt}")

    if not test_samples:
        console.print("[red]Test set is empty. Reduce --split or add more data.[/red]")
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────────────
    console.print(f"\n[bold]Running evaluation on {len(test_samples)} test samples...[/bold]")

    results: List[Dict] = []

    # Pre-classifier needs HTML — load it only if needed
    need_html = not no_html

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Evaluating...", total=len(test_samples))

        for sample in test_samples:
            url      = sample["url"]
            true_lbl = sample["label"]
            features = sample["features"]

            # Mode A: ML Only
            ml_lbl, ml_conf = predict_ml_only(features, model, le, feature_cols)

            # Mode B + C: Pre-classifier (needs HTML)
            pre_lbl    = None
            pre_conf   = 0.0
            pre_method = "no_rule"
            full_lbl   = ml_lbl
            full_conf  = ml_conf
            full_method = "ml_model"

            if need_html:
                try:
                    html = sample["html_path"].read_text(encoding="utf-8", errors="replace")

                    pre_result = predict_pre_classifier_only(html, url)
                    if pre_result:
                        pre_lbl, pre_conf, pre_method = pre_result
                        # Full pipeline: pre-classifier wins
                        full_lbl    = pre_lbl
                        full_conf   = pre_conf
                        full_method = pre_method
                    else:
                        # Pre-classifier passed → ML decides
                        pre_lbl  = ml_lbl   # for "pre only" stats, record ML result
                        pre_conf = ml_conf
                        full_lbl    = ml_lbl
                        full_conf   = ml_conf
                        full_method = "ml_model"

                except Exception as e:
                    # HTML read error → fall back to ML
                    pre_lbl    = ml_lbl
                    pre_conf   = ml_conf
                    pre_method = f"html_error: {str(e)[:40]}"
                    full_lbl   = ml_lbl
                    full_conf  = ml_conf
                    full_method = "ml_model"
            else:
                pre_lbl    = ml_lbl
                pre_conf   = ml_conf

            results.append({
                "url":         url,
                "true_label":  true_lbl,
                # ML
                "ml_label":    ml_lbl,
                "ml_conf":     ml_conf,
                "ml_method":   "ml_model",
                # Pre-classifier
                "pre_label":   pre_lbl,
                "pre_conf":    pre_conf,
                "pre_method":  pre_method,
                # Full pipeline
                "full_label":  full_lbl,
                "full_conf":   full_conf,
                "full_method": full_method,
            })

            progress.advance(task)

    # ── Compute metrics ───────────────────────────────────────────────────────
    y_true = [r["true_label"]  for r in results]
    y_ml   = [r["ml_label"]    for r in results]
    y_full = [r["full_label"]  for r in results]

    # For "pre only": treat no_rule as "passed to ML" → use ML label there
    # This measures: among URLs where a rule FIRED, how accurate were those rules?
    pre_fired = [r for r in results if r["pre_method"] != "no_rule" and not r["pre_method"].startswith("html_error")]
    if pre_fired:
        y_pre_true = [r["true_label"] for r in pre_fired]
        y_pre_pred = [r["pre_label"]  for r in pre_fired]
        pre_metrics = compute_metrics(y_pre_true, y_pre_pred, classes)
        pre_metrics["fired_count"] = len(pre_fired)
    else:
        # No rules fired — fill with zeros
        pre_metrics = {"accuracy": 0.0, "report": {}, "confusion_matrix": [], "n": 0, "fired_count": 0}

    ml_metrics   = compute_metrics(y_true, y_ml,   classes)
    full_metrics = compute_metrics(y_true, y_full, classes)

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison_table(ml_metrics, pre_metrics, full_metrics, classes)

    if need_html:
        print_pre_classifier_breakdown(results, classes)
        print_disagreements(results)

    if verbose:
        print_verbose_misclassified(results, mode="ml")
        print_verbose_misclassified(results, mode="full")
    else:
        # Always show full pipeline errors
        print_verbose_misclassified(results, mode="full")

    # ── Key insight summary ───────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Summary")

    ml_acc   = ml_metrics["accuracy"]
    full_acc = full_metrics["accuracy"]
    gain     = full_acc - ml_acc
    n_fired  = pre_metrics.get("fired_count", 0)
    n_total  = len(results)

    console.print(f"  ML model accuracy:        [cyan]{ml_acc:.2%}[/cyan]")
    console.print(f"  Full pipeline accuracy:   [green bold]{full_acc:.2%}[/green bold]")
    console.print(f"  Accuracy gain from rules: [{'green' if gain > 0 else 'red'}]{gain:+.2%}[/{'green' if gain > 0 else 'red'}]")
    console.print(f"  Pre-classifier fired on:  [cyan]{n_fired}/{n_total}[/cyan] samples ({n_fired/max(n_total,1):.1%})")
    console.print(f"  Remaining for ML model:   [cyan]{n_total-n_fired}/{n_total}[/cyan] samples ({(n_total-n_fired)/max(n_total,1):.1%})")

    if gain > 0:
        console.print(f"\n  [green]✓ Pre-classifier IMPROVES accuracy by {gain:+.2%}[/green]")
    elif gain == 0:
        console.print(f"\n  [yellow]= Pre-classifier has no net effect on accuracy[/yellow]")
    else:
        console.print(f"\n  [red]✗ Pre-classifier HURTS accuracy by {gain:.2%} — check wrong-fired rules[/red]")

    # ── Export ────────────────────────────────────────────────────────────────
    if export:
        export_path = Path(export)
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "url", "true_label",
                "ml_label", "ml_conf",
                "pre_label", "pre_conf", "pre_method",
                "full_label", "full_conf", "full_method",
                "ml_correct", "full_correct", "pre_fired", "pre_correct",
            ], extrasaction="ignore")
            writer.writeheader()
            for r in results:
                r["ml_correct"]   = int(r["ml_label"]   == r["true_label"])
                r["full_correct"] = int(r["full_label"]  == r["true_label"])
                r["pre_fired"]    = int(r["pre_method"]  != "no_rule" and not r["pre_method"].startswith("html_error"))
                r["pre_correct"]  = int(r["pre_label"]   == r["true_label"]) if r["pre_fired"] else ""
                writer.writerow(r)
        console.print(f"\n[green]✓ Detailed results exported → {export_path}[/green]")

    console.print()


if __name__ == "__main__":
    main()