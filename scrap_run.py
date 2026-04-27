"""
scrape_run.py
─────────────
CLI command: python scrape_run.py --input data/training_urls.csv

What it does:
  1. Reads the training CSV (columns: url, label)
  2. Checks the checkpoint — skips already-scraped (success/failed/skipped) URLs
  3. Only processes NEW urls
  4. Saves raw.html + page.json to output/<label>/<folder>/
  5. Updates checkpoint.json and mapping.json after each URL
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import (
    get_logger, load_config, load_checkpoint, update_checkpoint,
    update_mapping, url_to_folder_name, now_iso, console
)
from scraper.scraper import run_scraper

logger = get_logger("scrape_run")
cfg = load_config()


def _load_urls_from_csv(csv_path: Path) -> list[dict]:
    """Load and validate the training CSV."""
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    rows = []
    valid_labels = {"list", "detail", "others"}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            url = row.get("url", "").strip()
            label = row.get("label", "").strip().lower()
            if not url:
                logger.warning(f"Row {i}: empty URL — skipped")
                continue
            if label not in valid_labels:
                logger.warning(f"Row {i}: invalid label '{label}' for {url} — skipped")
                continue
            rows.append({"url": url, "label": label})

    logger.info(f"Loaded {len(rows)} valid rows from {csv_path}")
    return rows


def _filter_new_urls(all_urls: list[dict], checkpoint: dict) -> list[dict]:
    """
    Return only URLs not yet scraped.
    Skip: success, failed, skipped statuses.
    """
    new_urls = []
    for item in all_urls:
        url = item["url"]
        if url in checkpoint:
            status = checkpoint[url].get("scrape_status", "")
            if status in {"success", "failed", "skipped"}:
                continue  # Already processed — do not retry
        new_urls.append(item)

    skipped = len(all_urls) - len(new_urls)
    logger.info(f"Checkpoint: {skipped} URLs already processed — {len(new_urls)} new URLs to scrape")
    return new_urls


@click.command()
@click.option("--input", "input_csv", default=None, help="Path to training CSV (url, label)")
@click.option("--dry-run", is_flag=True, help="Print what would be scraped without doing it")
def main(input_csv: str, dry_run: bool):
    """Scrape training URLs and save raw HTML + page JSON."""
    cfg_paths = cfg["paths"]
    csv_path = Path(input_csv) if input_csv else Path(cfg_paths["training_csv"])
    output_root = Path(cfg_paths["output_dir"])

    console.rule("[bold blue]Webpage Classifier — Scrape Step")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output root: {output_root}")

    # ── Load ──────────────────────────────────────────────────────────────────
    all_urls = _load_urls_from_csv(csv_path)
    checkpoint = load_checkpoint()
    new_urls = _filter_new_urls(all_urls, checkpoint)

    if not new_urls:
        console.print("[green]✓ All URLs already processed. Nothing to do.")
        return

    if dry_run:
        console.print(f"[yellow]Dry run: would scrape {len(new_urls)} URLs:")
        for item in new_urls[:20]:
            console.print(f"  {item['label']} | {item['url']}")
        if len(new_urls) > 20:
            console.print(f"  ... and {len(new_urls) - 20} more")
        return

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = []
    for item in new_urls:
        folder = url_to_folder_name(item["url"])
        save_dir = output_root / item["label"] / folder
        tasks.append({
            "url": item["url"],
            "label": item["label"],
            "folder": folder,
            "save_dir": save_dir,
        })

    # Pre-register in checkpoint + mapping
    for t in tasks:
        update_checkpoint(t["url"], {
            "scrape_status": "pending",
            "extract_status": "pending",
            "folder": t["folder"],
            "label": t["label"],
        })
        update_mapping(t["folder"], {
            "url": t["url"],
            "label": t["label"],
            "scrape_status": "pending",
            "extract_status": "pending",
            "created_at": now_iso(),
        })

    console.print(f"\n[bold]Scraping [cyan]{len(tasks)}[/cyan] URLs with concurrency=[cyan]{cfg['scraper']['concurrency']}[/cyan]...\n")

    # ── Scrape ────────────────────────────────────────────────────────────────
    results = asyncio.run(run_scraper(tasks, output_root))

    # ── Update checkpoint / mapping ───────────────────────────────────────────
    success_count = failed_count = 0
    for t in tasks:
        url = t["url"]
        res = results.get(url, {"status": "failed", "reason": "no_result"})
        status = res["status"]
        reason = res.get("reason", "")

        if status == "success":
            success_count += 1
            files = []
            sd = t["save_dir"]
            if (sd / "raw.html").exists():
                files.append("raw.html")
            if (sd / "page.json").exists():
                files.append("page.json")
        else:
            failed_count += 1
            files = []

        update_checkpoint(url, {
            "scrape_status": status,
            "scrape_reason": reason,
            "scrape_completed_at": now_iso(),
        })
        update_mapping(t["folder"], {
            "scrape_status": status,
            "scrape_reason": reason,
            "files": files,
            "scrape_completed_at": now_iso(),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    table = Table(title="Scraping Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Count", style="cyan")
    table.add_row("Total URLs", str(len(tasks)))
    table.add_row("✓ Success", str(success_count))
    table.add_row("✗ Failed", str(failed_count))
    console.print(table)

    if failed_count > 0:
        console.print(
            f"\n[yellow]Note: {failed_count} failed URLs are recorded in checkpoint.json "
            "and will be skipped on the next run. They will NOT be used for training."
        )

    console.print("\n[green]Scraping complete. Run [bold]python extract_run.py[/bold] next.")


if __name__ == "__main__":
    main()