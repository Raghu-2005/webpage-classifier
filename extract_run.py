"""

What it does:
  1. Reads checkpoint.json to find URLs with scrape_status=success
     and extract_status=pending (or failed with --retry-failed flag)
  2. For each URL, reads raw.html + page.json from output folder
  3. Runs the full feature extractor
  4. Saves features.json to the same output folder
  5. Updates checkpoint.json and mapping.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import (
    get_logger, load_config, load_checkpoint, load_mapping,
    update_checkpoint, update_mapping, now_iso, console
)
from extraction.feature_extractor import extract_features, get_feature_columns

logger = get_logger("extract_run")
cfg = load_config()


@click.command()
@click.option("--retry-failed", is_flag=True, help="Also retry previously failed extractions")
def main(retry_failed: bool):
    """Extract features from scraped HTML files."""
    output_root = Path(cfg["paths"]["output_dir"])
    console.rule("[bold blue]Webpage Classifier — Extract Step")

    checkpoint = load_checkpoint()
    mapping = load_mapping()

    #  Find URLs to extract 
    to_extract = []
    for url, meta in checkpoint.items():
        if meta.get("scrape_status") != "success":
            continue  # Only process successfully scraped pages
        extract_status = meta.get("extract_status", "pending")
        if extract_status == "success":
            continue  # Already extracted
        if extract_status == "failed" and not retry_failed:
            continue  # Skip failed unless flag given

        folder = meta.get("folder")
        label = meta.get("label")
        if not folder or not label:
            continue

        save_dir = output_root / label / folder
        html_path = save_dir / "raw.html"
        json_path = save_dir / "page.json"

        if not html_path.exists():
            logger.warning(f"raw.html missing for {url} — marking as failed")
            update_checkpoint(url, {"extract_status": "failed", "extract_reason": "missing_html"})
            continue

        to_extract.append({
            "url": url,
            "label": label,
            "folder": folder,
            "save_dir": save_dir,
            "html_path": html_path,
            "json_path": json_path,
        })

    if not to_extract:
        console.print("[green]✓ No URLs pending extraction.")
        return

    console.print(f"\n[bold]Extracting features for [cyan]{len(to_extract)}[/cyan] URLs...\n")

    success_count = failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(to_extract))

        for item in to_extract:
            url = item["url"]
            progress.update(task, description=f"[cyan]{url[:60]}...")

            try:
                # Read HTML
                html = item["html_path"].read_text(encoding="utf-8", errors="replace")

                # Read page.json if available
                page_json = None
                if item["json_path"].exists():
                    page_json = json.loads(item["json_path"].read_text(encoding="utf-8"))

                # Extract features
                features = extract_features(html=html, url=url, page_json=page_json)

                # Add label for training use
                features["label"] = item["label"]

                # Save features.json
                features_path = item["save_dir"] / "features.json"
                features_path.write_text(
                    json.dumps(features, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                # Update checkpoint + mapping
                update_checkpoint(url, {
                    "extract_status": "success",
                    "extract_completed_at": now_iso(),
                    "feature_count": len(get_feature_columns(features)),
                })
                update_mapping(item["folder"], {
                    "extract_status": "success",
                    "extract_completed_at": now_iso(),
                    "feature_count": len(get_feature_columns(features)),
                    "files": ["raw.html", "page.json", "features.json"],
                })

                success_count += 1
                logger.debug(f"✓ Features extracted: {url}")

            except Exception as e:
                reason = str(e)[:200]
                logger.error(f"✗ Extraction failed: {url} → {reason}")
                update_checkpoint(url, {
                    "extract_status": "failed",
                    "extract_reason": reason,
                    "extract_completed_at": now_iso(),
                })
                update_mapping(item["folder"], {
                    "extract_status": "failed",
                    "extract_reason": reason,
                })
                failed_count += 1

            progress.advance(task)

    #  Summary 
    table = Table(title="Extraction Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Count", style="cyan")
    table.add_row("Total processed", str(len(to_extract)))
    table.add_row("✓ Success", str(success_count))
    table.add_row("✗ Failed", str(failed_count))
    console.print(table)

    console.print("\n[green]Extraction complete. Run [bold]python train_run.py[/bold] next.")


if __name__ == "__main__":
    main()