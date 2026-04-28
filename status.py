"""

Prints a live dashboard of the entire pipeline state:
  - Checkpoint summary (scrape + extract status per label)
  - Output folder file counts
  - Model info (if trained)
  - Recent prediction history
  - Any warnings (mismatches, missing files, etc.)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import console, get_logger, load_checkpoint, load_config, load_mapping

logger = get_logger("status")
cfg = load_config()


def _checkpoint_summary(checkpoint: dict) -> None:
    console.rule("[bold cyan]Checkpoint Summary")

    scrape_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    extract_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_counts: dict[str, int] = defaultdict(int)

    for url, meta in checkpoint.items():
        label = meta.get("label", "unknown")
        label_counts[label] += 1
        scrape_stats[label][meta.get("scrape_status", "unknown")] += 1
        extract_stats[label][meta.get("extract_status", "unknown")] += 1

    from rich.table import Table

    # Scrape table
    scrape_table = Table(title="Scrape Status", show_header=True, expand=False)
    scrape_table.add_column("Label", style="bold")
    scrape_table.add_column("Total", style="white")
    scrape_table.add_column("✓ success", style="green")
    scrape_table.add_column("✗ failed", style="red")
    scrape_table.add_column("⏳ pending", style="yellow")
    scrape_table.add_column("⊘ skipped", style="dim")

    for label in sorted(label_counts.keys()):
        ss = scrape_stats[label]
        scrape_table.add_row(
            label,
            str(label_counts[label]),
            str(ss.get("success", 0)),
            str(ss.get("failed", 0)),
            str(ss.get("pending", 0)),
            str(ss.get("skipped", 0)),
        )

    total = len(checkpoint)
    all_scrape = defaultdict(int)
    for d in scrape_stats.values():
        for k, v in d.items():
            all_scrape[k] += v

    scrape_table.add_row(
        "[bold]TOTAL[/bold]",
        str(total),
        str(all_scrape.get("success", 0)),
        str(all_scrape.get("failed", 0)),
        str(all_scrape.get("pending", 0)),
        str(all_scrape.get("skipped", 0)),
    )
    console.print(scrape_table)

    # Extract table
    extract_table = Table(title="Extraction Status", show_header=True, expand=False)
    extract_table.add_column("Label", style="bold")
    extract_table.add_column("Total", style="white")
    extract_table.add_column("✓ success", style="green")
    extract_table.add_column("✗ failed", style="red")
    extract_table.add_column("⏳ pending", style="yellow")

    for label in sorted(label_counts.keys()):
        es = extract_stats[label]
        extract_table.add_row(
            label,
            str(label_counts[label]),
            str(es.get("success", 0)),
            str(es.get("failed", 0)),
            str(es.get("pending", 0)),
        )

    all_extract = defaultdict(int)
    for d in extract_stats.values():
        for k, v in d.items():
            all_extract[k] += v

    extract_table.add_row(
        "[bold]TOTAL[/bold]",
        str(total),
        str(all_extract.get("success", 0)),
        str(all_extract.get("failed", 0)),
        str(all_extract.get("pending", 0)),
    )
    console.print(extract_table)

    # Ready for training
    ready = sum(
        1 for meta in checkpoint.values()
        if meta.get("scrape_status") == "success"
        and meta.get("extract_status") == "success"
    )
    console.print(f"\n  [green bold]{ready}[/green bold] URLs ready for training\n")


def _output_folder_summary() -> None:
    from rich.table import Table
    console.rule("[bold cyan]Output Folder Contents")
    output_root = Path(cfg["paths"]["output_dir"])

    table = Table(title="Output Files", show_header=True, expand=False)
    table.add_column("Label", style="bold")
    table.add_column("Folders", style="cyan")
    table.add_column("raw.html", style="white")
    table.add_column("page.json", style="white")
    table.add_column("features.json", style="green")
    table.add_column("Complete", style="bold green")

    for label in ["list", "detail", "others"]:
        label_dir = output_root / label
        if not label_dir.exists():
            table.add_row(label, "0", "0", "0", "0", "0")
            continue

        folders = [d for d in label_dir.iterdir() if d.is_dir()]
        html_count = sum(1 for f in folders if (f / "raw.html").exists())
        json_count = sum(1 for f in folders if (f / "page.json").exists())
        feat_count = sum(1 for f in folders if (f / "features.json").exists())
        complete = sum(
            1 for f in folders
            if (f / "raw.html").exists()
            and (f / "page.json").exists()
            and (f / "features.json").exists()
        )
        table.add_row(
            label,
            str(len(folders)),
            str(html_count),
            str(json_count),
            str(feat_count),
            str(complete),
        )

    console.print(table)


def _model_summary() -> None:
    from rich.table import Table
    console.rule("[bold cyan]Model Status")
    models_dir = Path(cfg["paths"]["models_dir"])
    report_path = models_dir / "training_report.json"

    if not report_path.exists():
        console.print("  [yellow]No trained model found. Run: python train_run.py[/yellow]\n")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    console.print(f"  Trained at:    [cyan]{report.get('trained_at', 'unknown')}")
    console.print(f"  Train samples: [cyan]{report.get('num_training_samples', '?')}")
    console.print(f"  Test samples:  [cyan]{report.get('num_test_samples', '?')}")
    console.print(f"  Features used: [cyan]{report.get('num_features', '?')}")
    console.print(f"  Classes:       [cyan]{', '.join(report.get('classes', []))}")

    cr = report.get("evaluation", {}).get("classification_report", {})
    table = Table(title="Test Set Performance", show_header=True, expand=False)
    table.add_column("Class", style="bold")
    table.add_column("Precision", style="cyan")
    table.add_column("Recall", style="cyan")
    table.add_column("F1-Score", style="green")
    table.add_column("Support", style="yellow")

    for cls in report.get("classes", []):
        r = cr.get(cls, {})
        table.add_row(
            cls,
            f"{r.get('precision', 0):.3f}",
            f"{r.get('recall', 0):.3f}",
            f"{r.get('f1-score', 0):.3f}",
            str(int(r.get('support', 0))),
        )
    ma = cr.get("macro avg", {})
    table.add_row(
        "[bold]macro avg[/bold]",
        f"{ma.get('precision', 0):.3f}",
        f"{ma.get('recall', 0):.3f}",
        f"{ma.get('f1-score', 0):.3f}",
        "—",
    )
    console.print(table)

    # Top 10 features
    imp_path = models_dir / "feature_importance.json"
    if imp_path.exists():
        importance = json.loads(imp_path.read_text())[:10]
        console.print("\n  [bold]Top 10 features by SHAP importance:[/bold]")
        for i, item in enumerate(importance, 1):
            bar = "█" * max(1, int(item["importance"] * 100))
            console.print(f"    {i:2}. [cyan]{item['feature']:<40}[/cyan] {item['importance']:.4f}  {bar}")
    console.print()


def _prediction_summary() -> None:
    from rich.table import Table
    console.rule("[bold cyan]Recent Predictions")
    pred_mapping_path = Path(cfg["paths"]["predicted_dir"]) / "mapping.json"

    if not pred_mapping_path.exists():
        console.print("  [dim]No predictions yet. Run: python predict_run.py --url <URL>[/dim]\n")
        return

    pred_map = json.loads(pred_mapping_path.read_text(encoding="utf-8"))
    if not pred_map:
        console.print("  [dim]No predictions yet.[/dim]\n")
        return

    # Sort by predicted_at descending, show last 10
    sorted_preds = sorted(
        pred_map.items(),
        key=lambda x: x[1].get("predicted_at", ""),
        reverse=True,
    )[:10]

    table = Table(title=f"Last {len(sorted_preds)} Predictions", show_header=True, expand=True)
    table.add_column("URL", style="cyan", max_width=55, no_wrap=True)
    table.add_column("Label", style="bold")
    table.add_column("Conf.", style="green")
    table.add_column("Fallback", style="yellow")
    table.add_column("Predicted At", style="dim")

    label_color = {"list": "blue", "detail": "green", "others": "yellow"}

    for folder, meta in sorted_preds:
        url = meta.get("url", "?")
        lbl = meta.get("predicted_label", "?")
        conf = meta.get("confidence", 0)
        fallback = "⚠ yes" if meta.get("fallback_used") else "no"
        ts = meta.get("predicted_at", "?")[:19].replace("T", " ")
        color = label_color.get(lbl, "white")
        table.add_row(
            url[:55],
            f"[{color}]{lbl}[/{color}]",
            f"{conf:.1%}",
            fallback,
            ts,
        )

    console.print(table)
    console.print(f"  [dim]Total predictions: {len(pred_map)}[/dim]\n")


def _warnings(checkpoint: dict) -> None:
    """Surface any data quality issues."""
    issues = []

    # Check for scrape success but extract pending (forgot to run extract_run)
    extract_pending = [
        url for url, meta in checkpoint.items()
        if meta.get("scrape_status") == "success"
        and meta.get("extract_status") == "pending"
    ]
    if extract_pending:
        issues.append(
            f"[yellow]⚠ {len(extract_pending)} URL(s) scraped but not yet extracted. "
            "Run: python extract_run.py[/yellow]"
        )

    # Check label balance
    from collections import Counter
    label_dist = Counter(
        meta.get("label") for meta in checkpoint.values()
        if meta.get("scrape_status") == "success"
        and meta.get("extract_status") == "success"
    )
    if label_dist:
        counts = list(label_dist.values())
        if max(counts) > min(counts) * 5:
            issues.append(
                f"[yellow]⚠ Class imbalance detected: {dict(label_dist)}. "
                "Consider adding more samples to minority classes.[/yellow]"
            )
        for lbl, cnt in label_dist.items():
            if cnt < 10:
                issues.append(
                    f"[red]✗ Class '{lbl}' has only {cnt} ready sample(s). "
                    "Minimum 30 recommended for good model performance.[/red]"
                )

    if issues:
        console.rule("[bold yellow]Warnings")
        for issue in issues:
            console.print(f"  {issue}")
        console.print()


@click.command()
@click.option("--failed", is_flag=True, help="List all failed URLs")
def main(failed: bool):
    """Show full pipeline status dashboard."""
    console.rule("[bold blue]Webpage Classifier — Pipeline Status")
    checkpoint = load_checkpoint()

    if not checkpoint:
        console.print(
            "[yellow]No checkpoint found. "
            "Add URLs to data/training_urls.csv and run: python scrape_run.py[/yellow]"
        )
    else:
        _checkpoint_summary(checkpoint)

    _output_folder_summary()
    _model_summary()
    _prediction_summary()

    if checkpoint:
        _warnings(checkpoint)

    if failed and checkpoint:
        console.rule("[bold red]Failed URLs")
        failed_urls = [
            (url, meta) for url, meta in checkpoint.items()
            if meta.get("scrape_status") == "failed"
            or meta.get("extract_status") == "failed"
        ]
        if not failed_urls:
            console.print("  [green]No failures found.[/green]")
        else:
            for url, meta in failed_urls:
                scrape_r = meta.get("scrape_reason", "")
                extract_r = meta.get("extract_reason", "")
                console.print(f"  [red]{url}[/red]")
                if scrape_r:
                    console.print(f"    scrape:  {scrape_r}")
                if extract_r:
                    console.print(f"    extract: {extract_r}")
        console.print()

if __name__ == "__main__":
    main()