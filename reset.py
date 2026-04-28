"""

Safely reset parts of the pipeline without losing data you want to keep.

Options allow selective reset:
  --checkpoint   Clear checkpoint (forces re-scrape of all URLs)
  --failed       Reset only failed/skipped URLs so they can be retried
  --extract      Reset all extract_status to pending (re-extract everything)
  --output       Delete all output folders (raw HTML + features)
  --model        Delete trained model artefacts
  --predicted    Delete all predictions
  --all          Full reset (everything above)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import (
    console, load_checkpoint, save_checkpoint,
    load_mapping, save_mapping, load_config
)

cfg = load_config()


def _confirm(message: str) -> bool:
    console.print(f"\n[yellow bold]{message}[/yellow bold]")
    answer = input("  Type 'yes' to confirm: ").strip().lower()
    return answer == "yes"


@click.command()
@click.option("--checkpoint", "reset_checkpoint", is_flag=True, help="Clear entire checkpoint")
@click.option("--failed", "reset_failed", is_flag=True, help="Reset only failed/skipped URLs for retry")
@click.option("--extract", "reset_extract", is_flag=True, help="Reset all extract_status to pending")
@click.option("--output", "reset_output", is_flag=True, help="Delete all output folders")
@click.option("--model", "reset_model", is_flag=True, help="Delete trained model artefacts")
@click.option("--predicted", "reset_predicted", is_flag=True, help="Delete all predictions")
@click.option("--all", "reset_all", is_flag=True, help="Full reset of everything")
@click.option("--yes", "auto_yes", is_flag=True, help="Skip confirmation prompts")
def main(
    reset_checkpoint, reset_failed, reset_extract,
    reset_output, reset_model, reset_predicted,
    reset_all, auto_yes
):
    """Selectively reset parts of the pipeline."""
    console.rule("[bold red]Webpage Classifier — Reset Tool")

    if not any([reset_checkpoint, reset_failed, reset_extract,
                reset_output, reset_model, reset_predicted, reset_all]):
        console.print("[yellow]No reset option specified. Use --help to see options.[/yellow]")
        return

    if reset_all:
        reset_checkpoint = reset_extract = reset_output = reset_model = reset_predicted = True

    actions = []
    if reset_checkpoint:
        actions.append("Delete data/checkpoint.json and data/mapping.json")
    if reset_failed:
        actions.append("Reset failed/skipped URLs in checkpoint for retry")
    if reset_extract:
        actions.append("Reset all extract_status to 'pending'")
    if reset_output:
        actions.append("DELETE all files in output/ (raw HTML, page.json, features.json)")
    if reset_model:
        actions.append("DELETE all trained model artefacts in models/")
    if reset_predicted:
        actions.append("DELETE all predictions in predicted/")

    console.print("\n[bold]Actions to perform:[/bold]")
    for a in actions:
        console.print(f"  • {a}")

    if not auto_yes:
        if not _confirm("Are you sure you want to perform these resets?"):
            console.print("[green]Reset cancelled.[/green]")
            return

    #  Execute 
    checkpoint_path = Path(cfg["paths"]["checkpoint"])
    mapping_path = Path(cfg["paths"]["mapping"])
    output_root = Path(cfg["paths"]["output_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    predicted_dir = Path(cfg["paths"]["predicted_dir"])

    if reset_checkpoint:
        for p in [checkpoint_path, mapping_path]:
            if p.exists():
                p.unlink()
                console.print(f"  [red]Deleted: {p}[/red]")
        console.print("  [green]✓ Checkpoint cleared[/green]")

    if reset_failed and not reset_checkpoint:
        checkpoint = load_checkpoint()
        reset_count = 0
        for url, meta in checkpoint.items():
            if meta.get("scrape_status") in {"failed", "skipped"}:
                meta["scrape_status"] = "pending"
                meta["extract_status"] = "pending"
                meta.pop("scrape_reason", None)
                reset_count += 1
            elif meta.get("extract_status") == "failed":
                meta["extract_status"] = "pending"
                meta.pop("extract_reason", None)
                reset_count += 1
        save_checkpoint(checkpoint)
        console.print(f"  [green]✓ Reset {reset_count} failed/skipped URL(s) to pending[/green]")

    if reset_extract and not reset_checkpoint:
        checkpoint = load_checkpoint()
        count = 0
        for url, meta in checkpoint.items():
            if meta.get("scrape_status") == "success":
                meta["extract_status"] = "pending"
                meta.pop("extract_reason", None)
                meta.pop("extract_completed_at", None)
                meta.pop("feature_count", None)
                count += 1
        save_checkpoint(checkpoint)
        # Also reset mapping extract status
        mapping = load_mapping()
        for folder, meta in mapping.items():
            if meta.get("scrape_status") == "success":
                meta["extract_status"] = "pending"
                meta["files"] = [f for f in meta.get("files", []) if f != "features.json"]
        save_mapping(mapping)
        console.print(f"  [green]✓ Reset extract_status to pending for {count} URL(s)[/green]")

    if reset_output:
        deleted = 0
        for label in ["list", "detail", "others"]:
            label_dir = output_root / label
            if label_dir.exists():
                shutil.rmtree(label_dir)
                label_dir.mkdir(parents=True, exist_ok=True)
                deleted += 1
        console.print(f"  [red]Deleted and recreated {deleted} output label directories[/red]")

    if reset_model:
        if models_dir.exists():
            shutil.rmtree(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"  [red]Deleted model artefacts in {models_dir}[/red]")

    if reset_predicted:
        if predicted_dir.exists():
            shutil.rmtree(predicted_dir)
            predicted_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"  [red]Deleted all predictions in {predicted_dir}[/red]")

    console.print("\n[green bold]Reset complete.[/green bold]")

if __name__ == "__main__":
    main()