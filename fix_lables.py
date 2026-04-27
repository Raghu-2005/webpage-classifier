"""
fix_labels.py
─────────────
Run this ONCE after fixing your training_urls.csv.

What it does:
  1. Reads the corrected training_urls.csv
  2. Compares each URL's current label in checkpoint vs the new label in CSV
  3. For URLs where the label CHANGED:
       - Moves the output folder from old label dir to new label dir
       - Updates checkpoint.json with the new label
       - Updates mapping.json with the new label
       - Resets extract_status to 'pending' so features.json is regenerated
         with the correct label
  4. Prints a full summary of what was moved

After running this, just run:
  python extract_run.py
  python train_run.py

No re-scraping needed at all.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import (
    console, get_logger, load_checkpoint, load_config,
    load_mapping, now_iso, save_checkpoint, save_mapping
)

logger = get_logger("fix_labels")
cfg    = load_config()


def main():
    output_root  = Path(cfg["paths"]["output_dir"])
    csv_path     = Path(cfg["paths"]["training_csv"])
    checkpoint   = load_checkpoint()
    mapping      = load_mapping()

    if not csv_path.exists():
        console.print(f"[red]CSV not found: {csv_path}")
        sys.exit(1)

    # ── Read the corrected CSV ─────────────────────────────────────────────────
    valid_labels = {"list", "detail", "others"}
    csv_labels: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url   = row.get("url", "").strip()
            label = row.get("label", "").strip().lower()
            if url and label in valid_labels:
                csv_labels[url] = label

    console.print(f"\nLoaded [cyan]{len(csv_labels)}[/cyan] URLs from {csv_path}")
    console.rule("[bold blue]Label Fix — Checking for mismatches")

    moved      = 0
    skipped    = 0
    not_found  = 0
    changes: list[tuple] = []

    for url, new_label in csv_labels.items():
        meta = checkpoint.get(url)
        if not meta:
            not_found += 1
            continue  # URL not yet scraped — will be handled by scrape_run.py

        old_label = meta.get("label", "")
        folder    = meta.get("folder", "")

        if not folder:
            skipped += 1
            continue

        if old_label == new_label:
            skipped += 1
            continue  # Label unchanged — nothing to do

        # Label changed — need to move the folder
        old_dir = output_root / old_label / folder
        new_dir = output_root / new_label / folder

        if not old_dir.exists():
            console.print(
                f"[yellow]⚠ Folder missing for {url[:60]} "
                f"(expected at {old_dir}) — skipping move"
            )
            # Still fix the label in checkpoint/mapping even if folder is missing
            meta["label"] = new_label
            meta["extract_status"] = "pending"
            meta.pop("extract_reason", None)
            meta.pop("extract_completed_at", None)
            meta.pop("feature_count", None)
            meta["label_fixed_at"] = now_iso()

            if folder in mapping:
                mapping[folder]["label"] = new_label
                mapping[folder]["extract_status"] = "pending"
                mapping[folder]["label_fixed_at"] = now_iso()
            skipped += 1
            continue

        # Move folder
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(old_dir), str(new_dir))
            console.print(
                f"  [green]✓ MOVED[/green] "
                f"[dim]{old_label}[/dim] → [bold]{new_label}[/bold] | "
                f"{url[:55]}"
            )
        except Exception as e:
            console.print(f"  [red]✗ FAILED to move {url}: {e}")
            skipped += 1
            continue

        # Delete stale features.json from the moved folder
        # (it has the wrong label embedded in it)
        stale_features = new_dir / "features.json"
        if stale_features.exists():
            stale_features.unlink()
            console.print(f"    [dim]Deleted stale features.json (will be re-extracted)[/dim]")

        # Update checkpoint
        meta["label"]              = new_label
        meta["extract_status"]     = "pending"   # re-extract with correct label
        meta.pop("extract_reason", None)
        meta.pop("extract_completed_at", None)
        meta.pop("feature_count", None)
        meta["label_fixed_at"]     = now_iso()

        # Update mapping
        if folder in mapping:
            mapping[folder]["label"]          = new_label
            mapping[folder]["extract_status"] = "pending"
            mapping[folder]["label_fixed_at"] = now_iso()
            # Remove features.json from files list
            mapping[folder]["files"] = [
                f for f in mapping[folder].get("files", [])
                if f != "features.json"
            ]

        changes.append((url, old_label, new_label, folder))
        moved += 1

    # ── Save updated checkpoint + mapping ─────────────────────────────────────
    save_checkpoint(checkpoint)
    save_mapping(mapping)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("[bold blue]Summary")
    console.print(f"\n  [green]✓ Folders moved & labels fixed: {moved}")
    console.print(f"  [dim]  Label unchanged (no action needed): {skipped}")
    console.print(f"  [dim]  Not yet scraped (will scrape later): {not_found}\n")

    if changes:
        console.print("[bold]Changed labels:[/bold]")
        for url, old, new, folder in changes:
            console.print(
                f"  [dim]{old:8}[/dim] → [bold]{new:8}[/bold] | "
                f"{url[:55]}"
            )

    console.print()
    console.print("[bold green]Done! Now run:[/bold green]")
    console.print("  [cyan]python extract_run.py[/cyan]   ← re-extracts only the fixed URLs")
    console.print("  [cyan]python train_run.py[/cyan]     ← trains on correct labels")
    console.print()


if __name__ == "__main__":
    main()