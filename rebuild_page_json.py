"""

Usage:
    python rebuild_page_json.py              # rebuild all output/ folders
    python rebuild_page_json.py --predicted  # also rebuild predicted/ folders
    python rebuild_page_json.py --dry-run    # show what would be rebuilt

What it produces (page.json v2):
    url, title, domain, path, query, scraped_at, html_length, status
    meta         → all <meta> name/property/itemprop tags
    open_graph   → all og:* values
    canonical    → canonical URL
    schema_org   → {raw: [...], types_found: [...]}
    counts       → links, images, forms, h1-h4, body_text_chars, body_text_words
    headings     → {h1: [...], h2: [...], h3: [...], h4: [...]}
    pagination   → {has_next, has_prev, rel_links: [...]}
    forms        → [{action, method, inputs: [...]}]
    links        → [{text, href, is_internal, rel}, ...]  ← ALL links
    images       → [{src, alt, width, height, loading}, ...]  ← ALL images
    body_text    → full visible text (first 50k chars)

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import click
from bs4 import BeautifulSoup
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import console, get_logger, load_checkpoint, load_config, now_iso

logger = get_logger("rebuild_page_json")
cfg    = load_config()


#  Core parser 

def build_rich_page_json(html: str, url: str) -> Dict:
    """
    Parse raw HTML with BeautifulSoup and extract everything into a
    structured dict matching the new page.json v2 format.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    parsed = urlparse(url)
    domain = parsed.netloc

    # Title 
    title_tag = soup.find("title")
    title     = title_tag.get_text(strip=True) if title_tag else ""

    #  Meta tags 
    meta: Dict[str, str] = {}
    for m in soup.find_all("meta"):
        name    = m.get("name") or m.get("property") or m.get("itemprop")
        content = m.get("content")
        if name and content:
            meta[name] = content

    # Open Graph 
    open_graph: Dict[str, str] = {}
    for m in soup.find_all("meta", property=True):
        prop = m.get("property", "")
        if prop.startswith("og:"):
            open_graph[prop] = m.get("content", "")

    # Canonical 
    canon_tag = soup.find("link", rel="canonical")
    canonical = canon_tag.get("href") if canon_tag else None

    # Schema.org JSON-LD 
    schema_raw: List = []
    schema_types: List[str] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            schema_raw.append(data)
        except Exception:
            pass

    def _collect_types(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            if isinstance(t, list):
                schema_types.extend(t)
            elif t:
                schema_types.append(str(t))
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _collect_types(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_types(item)

    _collect_types(schema_raw)
    schema_org = {"raw": schema_raw, "types_found": schema_types}

    #  Headings 
    headings: Dict[str, List[str]] = {}
    for tag in ["h1", "h2", "h3", "h4"]:
        headings[tag] = [
            h.get_text(strip=True)
            for h in soup.find_all(tag)
            if h.get_text(strip=True)
        ]

    #  Pagination 
    rel_links: List[Dict] = []
    has_next = has_prev = False
    for link in soup.find_all("link", rel=True):
        rel_val = link.get("rel", [])
        if isinstance(rel_val, str):
            rel_val = [rel_val]
        rel_lower = [r.lower() for r in rel_val]
        href = link.get("href", "")
        for r in rel_lower:
            if r == "next":
                has_next = True
                rel_links.append({"rel": "next", "href": href})
            elif r in ("prev", "previous"):
                has_prev = True
                rel_links.append({"rel": "prev", "href": href})

    pagination = {"has_next": has_next, "has_prev": has_prev, "rel_links": rel_links}

    #  All links 
    links: List[Dict] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href == "#" or href.startswith("javascript:"):
            continue

        # Determine internal/external
        is_internal = True
        if href.startswith("http://") or href.startswith("https://"):
            try:
                link_domain = urlparse(href).netloc
                is_internal = (link_domain == domain)
            except Exception:
                is_internal = False

        text = a.get_text(separator=" ", strip=True)
        rel  = a.get("rel", "")
        if isinstance(rel, list):
            rel = " ".join(rel)

        links.append({
            "text":        text[:200],
            "href":        href[:500],
            "is_internal": is_internal,
            "rel":         rel,
        })

    # All images 
    images: List[Dict] = []
    for img in soup.find_all("img"):
        src = (
            img.get("src") or
            img.get("data-src") or
            img.get("data-lazy-src") or
            ""
        ).strip()
        if not src:
            continue
        images.append({
            "src":     src[:500],
            "alt":     img.get("alt", "").strip()[:300],
            "width":   img.get("width", ""),
            "height":  img.get("height", ""),
            "loading": img.get("loading", ""),
        })

    # Forms 
    forms: List[Dict] = []
    for form in soup.find_all("form"):
        inputs = []
        for inp in form.find_all(["input", "select", "textarea"]):
            inputs.append({
                "type":        inp.get("type", inp.name),
                "name":        inp.get("name", ""),
                "placeholder": inp.get("placeholder", ""),
                "required":    inp.get("required") is not None,
            })
        forms.append({
            "action": form.get("action", ""),
            "method": form.get("method", "get").lower(),
            "inputs": inputs,
        })

    #  Body text (visible text only) 
    # Remove script/style/noscript from a clone for clean text
    body_soup = BeautifulSoup(html, "lxml")
    for tag in body_soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    body_text = body_soup.get_text(separator=" ", strip=True)
    # Normalise whitespace
    body_text = " ".join(body_text.split())[:50000]

    # Counts 
    counts = {
        "links":            len(links),
        "images":           len(images),
        "forms":            len(forms),
        "scripts":          len(soup.find_all("script")),
        "h1":               len(headings["h1"]),
        "h2":               len(headings["h2"]),
        "h3":               len(headings["h3"]),
        "h4":               len(headings["h4"]),
        "body_text_chars":  len(body_text),
        "body_text_words":  len(body_text.split()),
    }

    return {
        "url":         url,
        "title":       title,
        "domain":      domain,
        "path":        parsed.path,
        "query":       parsed.query,
        "scraped_at":  now_iso(),
        "html_length": len(html),
        "status":      "success",
        "rebuilt_from_html": True,   # flag: built by rebuild script, not live browser
        # Metadata
        "meta":        meta,
        "open_graph":  open_graph,
        "canonical":   canonical,
        "schema_org":  schema_org,
        # Structure
        "counts":      counts,
        "headings":    headings,
        "pagination":  pagination,
        "forms":       forms,
        # Content — the parts you wanted to verify
        "links":       links,
        "images":      images,
        "body_text":   body_text,
    }


#  Folder walker

def _collect_folders(output_root: Path) -> List[Dict]:
    """
    Walk output/<label>/<folder>/ and predicted/<folder>/ to find
    all folders that have raw.html but need page.json rebuilt.
    """
    checkpoint = load_checkpoint()

    # Build url lookup from checkpoint: folder → url
    folder_to_url: Dict[str, str] = {}
    for url, meta in checkpoint.items():
        folder = meta.get("folder")
        if folder:
            folder_to_url[folder] = url

    # Also check mapping.json for predicted folders
    from pipeline.utils import load_mapping
    mapping = load_mapping()
    for folder, meta in mapping.items():
        if folder not in folder_to_url and meta.get("url"):
            folder_to_url[folder] = meta["url"]

    folders = []

    # output/<label>/<folder>/
    for label_dir in ["list", "detail", "others"]:
        label_path = output_root / label_dir
        if not label_path.exists():
            continue
        for folder_path in label_path.iterdir():
            if not folder_path.is_dir():
                continue
            html_path = folder_path / "raw.html"
            if not html_path.exists():
                continue
            folder_name = folder_path.name
            url = folder_to_url.get(folder_name, "")
            folders.append({
                "folder_path": folder_path,
                "html_path":   html_path,
                "url":         url,
                "label":       label_dir,
            })

    return folders


def _collect_predicted_folders(predicted_root: Path) -> List[Dict]:
    """Walk predicted/<folder>/ for predicted pages."""
    from pipeline.utils import load_mapping
    mapping = load_mapping()

    # Also check predicted mapping
    pred_mapping_path = predicted_root / "mapping.json"
    pred_mapping = {}
    if pred_mapping_path.exists():
        pred_mapping = json.loads(pred_mapping_path.read_text())

    folders = []
    for folder_path in predicted_root.iterdir():
        if not folder_path.is_dir():
            continue
        html_path = folder_path / "raw.html"
        if not html_path.exists():
            continue
        folder_name = folder_path.name
        url = pred_mapping.get(folder_name, {}).get("url", "")
        folders.append({
            "folder_path": folder_path,
            "html_path":   html_path,
            "url":         url,
            "label":       "predicted",
        })

    return folders


# CLI 

@click.command()
@click.option("--predicted", is_flag=True, help="Also rebuild predicted/ folders")
@click.option("--dry-run",   is_flag=True, help="Show what would be rebuilt without doing it")
@click.option("--force",     is_flag=True, help="Rebuild even if page.json already exists")
def main(predicted: bool, dry_run: bool, force: bool):
    """
    Rebuild rich page.json from existing raw.html files.
    No re-scraping needed — reads HTML from disk.
    """
    console.rule("[bold blue]Rebuild page.json — v2 Rich Format")

    output_root    = Path(cfg["paths"]["output_dir"])
    predicted_root = Path(cfg["paths"]["predicted_dir"])

    # Collect all folders to process
    folders = _collect_folders(output_root)
    if predicted:
        folders += _collect_predicted_folders(predicted_root)

    if not folders:
        console.print("[yellow]No raw.html files found. Run scrape_run.py first.[/yellow]")
        return

    # Filter: skip already rebuilt unless --force
    if not force:
        to_rebuild = []
        skipped    = 0
        for item in folders:
            pj = item["folder_path"] / "page.json"
            if pj.exists():
                try:
                    existing = json.loads(pj.read_text())
                    # Check if already v2 (has links array)
                    if isinstance(existing.get("links"), list):
                        skipped += 1
                        continue
                except Exception:
                    pass
            to_rebuild.append(item)
        if skipped:
            console.print(f"[dim]Skipping {skipped} folders already on v2 format (use --force to redo)[/dim]")
    else:
        to_rebuild = folders

    console.print(f"Found [cyan]{len(to_rebuild)}[/cyan] folders to rebuild\n")

    if dry_run:
        for item in to_rebuild[:30]:
            console.print(f"  Would rebuild: [dim]{item['folder_path']}[/dim]")
            if item.get("url"):
                console.print(f"    URL: {item['url']}")
        if len(to_rebuild) > 30:
            console.print(f"  ... and {len(to_rebuild) - 30} more")
        return

    success_count = failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rebuilding...", total=len(to_rebuild))

        for item in to_rebuild:
            folder_path = item["folder_path"]
            html_path   = item["html_path"]
            url         = item.get("url", "")
            progress.update(task, description=f"[cyan]{folder_path.name[:40]}...")

            try:
                html = html_path.read_text(encoding="utf-8", errors="replace")

                # If we don't have the URL (rare), try to extract from existing page.json
                if not url:
                    pj_path = folder_path / "page.json"
                    if pj_path.exists():
                        try:
                            existing = json.loads(pj_path.read_text())
                            url = existing.get("url", "")
                        except Exception:
                            pass

                if not url:
                    logger.warning(f"No URL found for {folder_path.name} — skipping")
                    failed_count += 1
                    progress.advance(task)
                    continue

                page_json = build_rich_page_json(html=html, url=url)
                out_path  = folder_path / "page.json"
                out_path.write_text(
                    json.dumps(page_json, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                success_count += 1

            except Exception as e:
                logger.error(f"Failed for {folder_path.name}: {e}")
                failed_count += 1

            progress.advance(task)

    console.print(f"\n[green]✓ Rebuilt: {success_count}[/green]")
    if failed_count:
        console.print(f"[red]✗ Failed:  {failed_count}[/red]")

    console.print(f"\n[bold]page.json now includes:[/bold]")
    console.print("  • All links with text, href, is_internal, rel")
    console.print("  • All images with src, alt, width, height")
    console.print("  • All headings h1–h4 as lists")
    console.print("  • Full body text (up to 50k chars)")
    console.print("  • Schema.org JSON-LD parsed + types extracted")
    console.print("  • Open Graph values")
    console.print("  • Pagination rel=next/prev links")
    console.print("  • All forms with their inputs")
    console.print("\n[dim]Feature extraction is unchanged — this is for debugging only.[/dim]")


if __name__ == "__main__":
    main()