"""


Commands:
  python inspect.py url     <URL>          → show features for a specific URL
  python inspect.py folder  <folder>       → show by folder name
  python inspect.py compare <URL1> <URL2>  → side-by-side feature comparison
  python inspect.py top-features           → show top discriminating features
  python inspect.py list-urls              → list all URLs in checkpoint
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import (
    console, get_logger, load_checkpoint, load_config,
    load_mapping, url_to_folder_name
)

logger = get_logger("inspect")
cfg = load_config()


def _load_features_for_url(url: str) -> tuple[dict | None, Path | None]:
    """Load features.json for a given URL from output or predicted dirs."""
    checkpoint = load_checkpoint()
    meta = checkpoint.get(url)

    if meta:
        label = meta.get("label", "")
        folder = meta.get("folder", "")
        if label and folder:
            p = Path(cfg["paths"]["output_dir"]) / label / folder / "features.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8")), p

    # Try predicted dir
    folder = url_to_folder_name(url)
    p = Path(cfg["paths"]["predicted_dir"]) / folder / "features.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), p

    return None, None


def _load_features_for_folder(folder: str) -> tuple[dict | None, Path | None]:
    """Load features.json for a given folder name."""
    mapping = load_mapping()
    meta = mapping.get(folder)
    if meta:
        label = meta.get("label", "")
        p = Path(cfg["paths"]["output_dir"]) / label / folder / "features.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p

    # Try predicted
    p = Path(cfg["paths"]["predicted_dir"]) / folder / "features.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), p

    return None, None


def _print_features(features: dict, title: str = "Features") -> None:
    from rich.table import Table

    # Separate metadata from model features
    meta_keys = {k: v for k, v in features.items() if k.startswith("_")}
    model_keys = {k: v for k, v in features.items() if not k.startswith("_") and k != "label"}

    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  URL:   [cyan]{meta_keys.get('_url', '?')}[/cyan]")
    console.print(f"  Title: [dim]{str(meta_keys.get('_title', '?'))[:80]}[/dim]")
    console.print(f"  Label: [green]{features.get('label', '?')}[/green]")
    console.print(f"  Body text length: {meta_keys.get('_body_text_length', '?')} chars")
    console.print()

    # Group features by pillar prefix
    pillars = {
        "STRUCTURAL":  [k for k in model_keys if k.startswith(("tag_", "li_", "dom_", "has_data_table", "grid_", "article_section"))],
        "LINK":        [k for k in model_keys if "link" in k or "anchor" in k or k == "nav_links" or k == "external_links" or k == "internal_links"],
        "CONTENT":     [k for k in model_keys if any(k.startswith(p) for p in ("h1_", "h2_", "h3_", "h4_", "word_", "char_", "para", "avg_para", "rich_", "log_word", "log_rich", "deep_content", "has_main", "has_article_el", "text_conc", "single_h1", "heading_hier", "content_avail"))],
        "MEDIA":       [k for k in model_keys if any(k.startswith(p) for p in ("image_", "log_image", "alt_", "gallery_", "has_image_gallery", "has_video", "figure_", "figcaption_", "images_with"))],
        "LISTING":     [k for k in model_keys if any(s in k for s in ("pagination", "filter_ui", "result_count", "repeated", "breadcrumb", "compare_check"))],
        "DETAIL":      [k for k in model_keys if any(s in k for s in ("price", "buy_cta", "author", "publish_date", "rating", "toc", "internal_anchor", "spec_section", "share_button", "comments", "related"))],
        "INTERACTION": [k for k in model_keys if any(s in k for s in ("form", "button", "select", "range_input", "tab", "login", "search_form"))],
        "METADATA":    [k for k in model_keys if any(s in k for s in ("og_type", "twitter", "schema", "microdata", "canonical", "robots", "lang"))],
        "SEMANTIC":    [k for k in model_keys if any(s in k for s in ("cue_word", "numeric_token", "title_"))],
        "URL":         [k for k in model_keys if k.startswith("url_")],
        "COMPOSITE":   [k for k in model_keys if k.startswith("composite_")],
    }

    for pillar, keys in pillars.items():
        if not keys:
            continue
        table = Table(title=f"[bold]{pillar}[/bold]", show_header=True, expand=False, min_width=60)
        table.add_column("Feature", style="cyan", min_width=38)
        table.add_column("Value", style="white")

        for k in sorted(keys):
            v = model_keys.get(k, "—")
            if isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            # Highlight non-zero values
            if v not in (0, 0.0, False, "0", "0.0000"):
                table.add_row(f"[bold]{k}[/bold]", f"[green]{v_str}[/green]")
            else:
                table.add_row(k, f"[dim]{v_str}[/dim]")

        console.print(table)


def _compare_features(feat1: dict, feat2: dict, url1: str, url2: str) -> None:
    from rich.table import Table

    all_keys = sorted(set(
        k for k in list(feat1.keys()) + list(feat2.keys())
        if not k.startswith("_") and k != "label"
    ))

    table = Table(title="Feature Comparison", show_header=True, expand=True)
    table.add_column("Feature", style="cyan", min_width=38)
    table.add_column(f"A: {url1[:40]}", style="white")
    table.add_column(f"B: {url2[:40]}", style="white")
    table.add_column("Diff", style="yellow")

    for k in all_keys:
        v1 = feat1.get(k, 0)
        v2 = feat2.get(k, 0)

        if isinstance(v1, float) or isinstance(v2, float):
            v1_s = f"{v1:.4f}"
            v2_s = f"{v2:.4f}"
            try:
                diff = f"{float(v2) - float(v1):+.4f}"
            except Exception:
                diff = "—"
        else:
            v1_s = str(v1)
            v2_s = str(v2)
            try:
                diff = f"{int(v2) - int(v1):+d}"
            except Exception:
                diff = "—"

        # Highlight rows where values differ
        if v1 != v2:
            table.add_row(f"[bold]{k}[/bold]", v1_s, v2_s, diff)

    console.print(table)
    console.print(f"\n  [dim](Only showing features that differ between A and B)[/dim]")


@click.group()
def cli():
    """Inspect pipeline output — features, folders, and comparisons."""
    pass


@cli.command()
@click.argument("url")
def url(url: str):
    """Show extracted features for a URL."""
    features, path = _load_features_for_url(url.strip())
    if features is None:
        console.print(f"[red]No features found for URL: {url}")
        console.print("Make sure the URL has been scraped and extracted first.")
        return
    console.print(f"  [dim]Source: {path}[/dim]")
    _print_features(features, title=f"Features for {url[:60]}")


@cli.command()
@click.argument("folder_name")
def folder(folder_name: str):
    """Show extracted features for a folder name."""
    features, path = _load_features_for_folder(folder_name.strip())
    if features is None:
        console.print(f"[red]No features found for folder: {folder_name}")
        return
    console.print(f"  [dim]Source: {path}[/dim]")
    _print_features(features, title=f"Features in folder: {folder_name}")


@cli.command()
@click.argument("url1")
@click.argument("url2")
def compare(url1: str, url2: str):
    """Compare features of two URLs side by side (only shows differences)."""
    feat1, _ = _load_features_for_url(url1.strip())
    feat2, _ = _load_features_for_url(url2.strip())

    if feat1 is None:
        console.print(f"[red]No features found for: {url1}")
        return
    if feat2 is None:
        console.print(f"[red]No features found for: {url2}")
        return

    console.rule(f"[bold cyan]Comparing Features")
    console.print(f"  A: [cyan]{url1}[/cyan] (label: [green]{feat1.get('label', '?')}[/green])")
    console.print(f"  B: [cyan]{url2}[/cyan] (label: [green]{feat2.get('label', '?')}[/green])\n")
    _compare_features(feat1, feat2, url1, url2)


@cli.command("top-features")
def top_features():
    """Show SHAP feature importance from the trained model."""
    imp_path = Path(cfg["paths"]["models_dir"]) / "feature_importance.json"
    if not imp_path.exists():
        console.print("[yellow]No model trained yet. Run: python train_run.py")
        return

    from rich.table import Table
    importance = json.loads(imp_path.read_text())

    table = Table(title="Top Features by SHAP Importance", show_header=True, expand=False)
    table.add_column("Rank", style="dim")
    table.add_column("Feature", style="cyan", min_width=40)
    table.add_column("SHAP Score", style="green")
    table.add_column("Bar", style="blue")

    for rank, item in enumerate(importance[:40], 1):
        bar = "█" * max(1, int(item["importance"] * 200))
        table.add_row(
            str(rank),
            item["feature"],
            f"{item['importance']:.6f}",
            bar[:40],
        )
    console.print(table)


@cli.command("list-urls")
@click.option("--label", default=None, help="Filter by label: list, detail, others")
@click.option("--status", default="success", help="Filter by scrape status (default: success)")
def list_urls(label: str | None, status: str):
    """List all URLs in the checkpoint."""
    from rich.table import Table
    checkpoint = load_checkpoint()

    rows = []
    for url, meta in checkpoint.items():
        if label and meta.get("label") != label:
            continue
        if status and meta.get("scrape_status") != status:
            continue
        rows.append((url, meta))

    if not rows:
        console.print(f"[yellow]No URLs found with status={status}" + (f", label={label}" if label else ""))
        return

    table = Table(
        title=f"URLs (status={status}" + (f", label={label}" if label else "") + f") — {len(rows)} total",
        show_header=True, expand=True
    )
    table.add_column("URL", style="cyan", max_width=65)
    table.add_column("Label", style="bold")
    table.add_column("Scrape", style="green")
    table.add_column("Extract", style="green")
    table.add_column("Folder", style="dim")

    for url, meta in sorted(rows, key=lambda x: x[1].get("label", "")):
        s = meta.get("scrape_status", "?")
        e = meta.get("extract_status", "?")
        s_color = "green" if s == "success" else "red" if s == "failed" else "yellow"
        e_color = "green" if e == "success" else "red" if e == "failed" else "yellow"
        table.add_row(
            url[:65],
            meta.get("label", "?"),
            f"[{s_color}]{s}[/{s_color}]",
            f"[{e_color}]{e}[/{e_color}]",
            meta.get("folder", "?")[:20],
        )
    console.print(table)


if __name__ == "__main__":
    cli()