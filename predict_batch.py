"""
predict_batch.py — Batch prediction with pre-classifier pipeline

CHANGELOG:
  - Added scrape_failed early-return matching predict_run.py.
    When HTML is empty/scraping failed, the URL is recorded as
    scrape_failed in mapping.json instead of a fake confident prediction.
"""
from __future__ import annotations
import asyncio, csv, json, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import click, joblib, numpy as np, pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import console, get_logger, load_config, now_iso, url_to_folder_name
from scraper.scraper import run_scraper
from extraction.feature_extractor import extract_features, _url_features
from extraction.pre_classifier import pre_classify

logger = get_logger("predict_batch")
cfg    = load_config()
PREDICTED_DIR          = Path(cfg["paths"]["predicted_dir"])
MODELS_DIR             = Path(cfg["paths"]["models_dir"])
PREDICTED_MAPPING_PATH = PREDICTED_DIR / "mapping.json"


def _load_predicted_mapping():
    return json.loads(PREDICTED_MAPPING_PATH.read_text()) if PREDICTED_MAPPING_PATH.exists() else {}


def _save_predicted_mapping(data):
    PREDICTED_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTED_MAPPING_PATH.write_text(json.dumps(data, indent=2))


def _load_model_artefacts():
    mp = MODELS_DIR / "classifier.joblib"
    if not mp.exists():
        logger.error("Model not found. Run: python train_run.py")
        sys.exit(1)
    return (joblib.load(mp),
            joblib.load(MODELS_DIR / "label_encoder.joblib"),
            json.loads((MODELS_DIR / "feature_columns.json").read_text()))


def _url_pattern_fallback(url):
    f = _url_features(url)
    if f.get("url_has_search_param") or f.get("url_has_page_param"): return "list", 0.62
    if f.get("url_has_filter_param"):   return "list",   0.60
    if f.get("url_has_numeric_id") and not f.get("url_is_root"): return "detail", 0.60
    if f.get("url_has_long_slug"):      return "detail", 0.58
    if f.get("url_is_root"):            return "others", 0.65
    return "others", 0.50


def _predict_one(html, url, save_dir, model, le, feature_cols):
    """
    Full pipeline for one URL: pre-classifier → ML → URL fallback.
    Returns None if scraping failed (empty HTML) — caller handles that case.
    """
    # Pre-classifier
    pre = pre_classify(html=html, url=url, http_status=None)
    if pre:
        if html:
            pj = None
            pjp = save_dir / "page.json"
            if pjp.exists():
                try:
                    pj = json.loads(pjp.read_text())
                except Exception:
                    pass
            feats = extract_features(html=html, url=url, page_json=pj)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "features.json").write_text(
                json.dumps(feats, indent=2, ensure_ascii=False)
            )
        return {
            "predicted_label": pre["label"],
            "confidence": pre["confidence"],
            "method": pre["method"],
            "reason": pre.get("reason", ""),
            "probabilities": {},
            "fallback_used": False,
            "content_available": bool(html),
            "is_error_page": pre.get("is_error_page", False),
        }

    # ML model
    pj = None
    pjp = save_dir / "page.json"
    if pjp.exists():
        try:
            pj = json.loads(pjp.read_text())
        except Exception:
            pass
    features = extract_features(html=html, url=url, page_json=pj)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "features.json").write_text(
        json.dumps(features, indent=2, ensure_ascii=False)
    )

    content_available = features.get("content_available", 0)
    row   = {col: features.get(col, 0) for col in feature_cols}
    X     = pd.DataFrame([row]).fillna(0)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    lbl   = le.inverse_transform([idx])[0]
    conf  = float(proba[idx])
    proba_dict = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}
    fallback = False
    if conf < 0.50 and not content_available:
        lbl, conf = _url_pattern_fallback(url)
        fallback  = True
    return {
        "predicted_label": lbl,
        "confidence": round(conf, 4),
        "probabilities": proba_dict,
        "method": "ml_model" + ("_url_fallback" if fallback else ""),
        "fallback_used": fallback,
        "content_available": bool(content_available),
    }


@click.command()
@click.option("--input", "input_csv", required=True)
@click.option("--force", is_flag=True)
@click.option("--concurrency", default=None, type=int)
def main(input_csv, force, concurrency):
    """Batch predict webpage types from a CSV of URLs."""
    csv_path = Path(input_csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    console.rule("[bold blue]Webpage Classifier — Batch Predict")
    urls = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("url") or row.get("URL") or "").strip()
            if u:
                urls.append(u)
    if not urls:
        logger.error("No URLs found. Column must be 'url'.")
        sys.exit(1)
    console.print(f"Loaded [cyan]{len(urls)}[/cyan] URLs")

    pred_mapping = _load_predicted_mapping()
    if not force:
        done, new = [], []
        for u in urls:
            fn = url_to_folder_name(u)
            status = pred_mapping.get(fn, {}).get("status", "")
            # Skip already-predicted AND already-failed (don't retry scrape failures
            # unless --force is passed — consistent with training checkpoint behaviour)
            if status in ("success", "scrape_failed"):
                done.append(u)
            else:
                new.append(u)
        if done:
            console.print(f"[dim]Skipping {len(done)} already-processed URLs[/dim]")
        urls = new
    if not urls:
        console.print("[green]✓ All URLs already processed.")
        return

    model, le, feature_cols = _load_model_artefacts()
    if concurrency:
        cfg["scraper"]["concurrency"] = concurrency

    tasks = [
        {
            "url": u,
            "label": "predict",
            "folder": url_to_folder_name(u),
            "save_dir": PREDICTED_DIR / url_to_folder_name(u),
        }
        for u in urls
    ]
    console.print(f"\nScraping [cyan]{len(tasks)}[/cyan] URLs...\n")
    scrape_results = asyncio.run(run_scraper(tasks, PREDICTED_DIR))

    results = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console
    ) as progress:
        bar = progress.add_task("Predicting...", total=len(tasks))

        for item in tasks:
            url      = item["url"]
            folder   = item["folder"]
            save_dir = item["save_dir"]
            progress.update(bar, description=f"[cyan]{url[:55]}...")

            sr        = scrape_results.get(url, {})
            scrape_ok = sr.get("status") == "success"
            html      = ""
            if scrape_ok:
                hp = save_dir / "raw.html"
                if hp.exists():
                    html = hp.read_text(encoding="utf-8", errors="replace")

            # ── Scrape failed: record and skip ────────────────────────────────
            if not scrape_ok or not html.strip():
                fail_reason = sr.get("reason", "unknown")
                logger.warning(f"Scrape failed ({fail_reason}): {url}")
                pred_output = {
                    "url": url, "folder": folder, "predicted_at": now_iso(),
                    "scrape_status": "failed",
                    "predicted_label": "others", "confidence": 0.0,
                    "method": "scrape_failed",
                    "reason": f"scraping_failed: {fail_reason}",
                    "probabilities": {}, "fallback_used": False,
                    "content_available": False, "is_scrape_failed": True,
                }
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / "prediction.json").write_text(json.dumps(pred_output, indent=2))
                pred_mapping[folder] = {
                    "url": url, "folder": folder, "status": "scrape_failed",
                    "predicted_label": "others", "confidence": 0.0,
                    "method": "scrape_failed", "fallback_used": False,
                    "predicted_at": now_iso(), "files": ["prediction.json"],
                }
                results.append({
                    "url": url, "predicted_label": "scrape_failed",
                    "confidence": 0.0, "method": "scrape_failed",
                    "fallback_used": False, "scrape_status": "failed",
                    "status": "scrape_failed",
                })
                progress.advance(bar)
                continue

            # ── Normal prediction pipeline ────────────────────────────────────
            try:
                pred = _predict_one(html, url, save_dir, model, le, feature_cols)
                pred_output = {
                    "url": url, "folder": folder, "predicted_at": now_iso(),
                    "scrape_status": "success", **pred,
                }
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / "prediction.json").write_text(json.dumps(pred_output, indent=2))
                pred_mapping[folder] = {
                    "url": url, "folder": folder, "status": "success",
                    "predicted_label": pred["predicted_label"],
                    "confidence": pred["confidence"],
                    "method": pred.get("method", "?"),
                    "fallback_used": pred.get("fallback_used", False),
                    "predicted_at": now_iso(),
                    "files": ["raw.html", "features.json", "prediction.json"],
                }
                results.append({
                    "url": url,
                    "predicted_label": pred["predicted_label"],
                    "confidence": pred["confidence"],
                    "method": pred.get("method", "?"),
                    "fallback_used": pred.get("fallback_used", False),
                    "scrape_status": "success",
                    "status": "success",
                })
            except Exception as e:
                err = str(e)[:120]
                logger.error(f"Prediction failed for {url}: {err}")
                results.append({
                    "url": url, "predicted_label": "error", "confidence": 0.0,
                    "method": "error", "fallback_used": False,
                    "scrape_status": "success", "status": "failed", "error": err,
                })
            progress.advance(bar)

    _save_predicted_mapping(pred_mapping)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PREDICTED_DIR / f"batch_results_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["url", "predicted_label", "confidence", "method",
                        "fallback_used", "scrape_status", "status"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)

    label_dist  = Counter(r["predicted_label"] for r in results if r["status"] == "success")
    method_dist = Counter(r.get("method", "?") for r in results if r["status"] == "success")
    succs        = sum(1 for r in results if r["status"] == "success")
    scrape_fails = sum(1 for r in results if r["status"] == "scrape_failed")
    pred_fails   = sum(1 for r in results if r["status"] == "failed")

    table = Table(title="Batch Prediction Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Total URLs", str(len(results)))
    table.add_row("✓ Predicted", str(succs))
    table.add_row("⚠ Scrape failed", str(scrape_fails))
    table.add_row("✗ Predict error", str(pred_fails))
    for lbl, cnt in sorted(label_dist.items()):
        table.add_row(f"  → {lbl}", str(cnt))
    table.add_row("", "")
    table.add_row("[bold]Methods used[/bold]", "")
    for m, cnt in sorted(method_dist.items()):
        table.add_row(f"  {m}", str(cnt))
    console.print(table)
    console.print(f"\n[green]✓ Results → {out}")


if __name__ == "__main__":
    main()