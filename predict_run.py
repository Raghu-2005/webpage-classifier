"""

Two modes:

  1. Interactive loop (default):
       python predict_run.py
     → prompts "Enter URL (or 'exit'):" repeatedly until you type exit/quit/q

  2. Single URL (original behaviour):
       python predict_run.py --url https://example.com
       python predict_run.py --url https://example.com --force
       python predict_run.py --url https://example.com --debug

Classification pipeline (priority order):
  1. Scrape failed  → return scrape_failed immediately, no prediction
  2. Error page     → "others" immediately
  3. Schema.org     → direct result, skip ML
  4. Open Graph     → direct result, skip ML (unless listing URL)
  5. Twitter Card   → direct result, skip ML
  6. Strong DOM     → direct result, skip ML
  7. ML model       → XGBoost on all features
  8. URL fallback   → ONLY if ML < 50% AND no content
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path
import click, joblib, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import console, get_logger, load_config, now_iso, url_to_folder_name
from scraper.scraper import run_scraper
from extraction.feature_extractor import extract_features, _url_features
from extraction.pre_classifier import pre_classify

logger = get_logger("predict_run")
cfg    = load_config()
PREDICTED_MAPPING_PATH = Path(cfg["paths"]["predicted_dir"]) / "mapping.json"


# ─── Helpers (unchanged) ──────────────────────────────────────────────────────

def _load_predicted_mapping():
    return json.loads(PREDICTED_MAPPING_PATH.read_text()) if PREDICTED_MAPPING_PATH.exists() else {}


def _save_predicted_mapping(data):
    PREDICTED_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTED_MAPPING_PATH.write_text(json.dumps(data, indent=2))


def _load_model_artefacts(models_dir):
    mp = models_dir / "classifier.joblib"
    if not mp.exists():
        logger.error("Model not found. Run: python train_run.py first.")
        sys.exit(1)
    return (
        joblib.load(mp),
        joblib.load(models_dir / "label_encoder.joblib"),
        json.loads((models_dir / "feature_columns.json").read_text()),
    )


def _url_pattern_fallback(url):
    f = _url_features(url)
    if f.get("url_has_search_param") or f.get("url_has_page_param"): return "list",   0.62
    if f.get("url_has_filter_param"):                                  return "list",   0.60
    if f.get("url_has_numeric_id") and not f.get("url_is_root"):       return "detail", 0.60
    if f.get("url_has_long_slug"):                                      return "detail", 0.58
    if f.get("url_is_root"):                                            return "others", 0.65
    return "others", 0.50


def _run_ml(features, model, le, feature_cols, url):
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
        "confidence":      round(conf, 4),
        "probabilities":   proba_dict,
        "method":          "ml_model" + ("_url_fallback" if fallback else ""),
        "fallback_used":   fallback,
        "content_available": bool(content_available),
    }


def _save_features(html, url, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    pj_path   = save_dir / "page.json"
    page_json = json.loads(pj_path.read_text()) if pj_path.exists() else None
    features  = extract_features(html=html, url=url, page_json=page_json)
    (save_dir / "features.json").write_text(json.dumps(features, indent=2, ensure_ascii=False))
    return features


# ─── Core prediction logic (shared by both modes) ────────────────────────────

def _predict_url(url: str, force: bool, debug: bool,
                 model, le, feature_cols,
                 predicted_root: Path, models_dir: Path) -> None:
    """
    Run the full prediction pipeline for one URL and print results.
    Extracted so both the single-URL mode and the interactive loop can call it.
    """
    url          = url.strip()
    folder       = url_to_folder_name(url)
    save_dir     = predicted_root / folder
    pred_mapping = _load_predicted_mapping()

    # Already predicted cache check
    if folder in pred_mapping and not force:
        ex = pred_mapping[folder]
        if ex.get("status") == "success":
            console.print(
                f"\n[yellow]Already predicted → [bold]{ex['predicted_label'].upper()}[/bold] "
                f"(conf: {ex['confidence']:.1%}, method: {ex.get('method','?')})[/yellow]"
            )
            console.print(f"[dim]  Use --force flag to re-run.[/dim]\n")
            return

    # Step 1: Scrape
    console.print(f"\n[bold cyan]URL:[/bold cyan] {url}")
    console.print("[dim]Scraping...[/dim]", end="\r")
    scrape_results = asyncio.run(run_scraper(
        [{"url": url, "label": "predict", "folder": folder, "save_dir": save_dir}],
        predicted_root,
    ))
    sr        = scrape_results.get(url, {})
    scrape_ok = sr.get("status") == "success"
    html      = ""

    if scrape_ok:
        hp   = save_dir / "raw.html"
        html = hp.read_text(encoding="utf-8", errors="replace") if hp.exists() else ""

    # Scrape failed
    if not scrape_ok or not html.strip():
        fail_reason = sr.get("reason", "unknown")
        prediction  = {
            "predicted_label":  "others",
            "confidence":        0.0,
            "method":            "scrape_failed",
            "reason":            f"scraping_failed: {fail_reason}",
            "probabilities":     {},
            "fallback_used":     False,
            "content_available": False,
            "is_scrape_failed":  True,
        }
        pred_output = {"url": url, "folder": folder, "predicted_at": now_iso(),
                       "scrape_status": "failed", **prediction}
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "prediction.json").write_text(json.dumps(pred_output, indent=2))
        pred_mapping[folder] = {
            "url": url, "folder": folder, "status": "scrape_failed",
            "predicted_label": "others", "confidence": 0.0,
            "method": "scrape_failed", "fallback_used": False,
            "predicted_at": now_iso(), "files": ["prediction.json"],
        }
        _save_predicted_mapping(pred_mapping)
        _print_result(prediction, save_dir, scrape_failed=True, fail_reason=fail_reason)
        return

    # Step 2: Pre-classifier
    pre = pre_classify(html=html, url=url, http_status=None)
    if pre:
        prediction = {
            "predicted_label":   pre["label"],
            "confidence":        pre["confidence"],
            "method":            pre["method"],
            "reason":            pre.get("reason", ""),
            "probabilities":     {},
            "fallback_used":     False,
            "content_available": bool(html),
            "is_error_page":     pre.get("is_error_page", False),
        }
        if debug:
            console.print(f"  [green]Pre-classifier: {pre['method']} → "
                          f"{pre['label']} ({pre['confidence']:.0%})[/green]")
        if html:
            _save_features(html, url, save_dir)
    else:
        # Step 3: ML model
        features   = _save_features(html, url, save_dir)
        prediction = _run_ml(features, model, le, feature_cols, url)
        if debug:
            console.print(f"  [blue]ML: {prediction['predicted_label']} "
                          f"({prediction['confidence']:.0%})[/blue]")

    # Save
    pred_output = {"url": url, "folder": folder, "predicted_at": now_iso(),
                   "scrape_status": "success", **prediction}
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "prediction.json").write_text(json.dumps(pred_output, indent=2))
    pred_mapping[folder] = {
        "url": url, "folder": folder, "status": "success",
        "predicted_label": prediction["predicted_label"],
        "confidence":      prediction["confidence"],
        "method":          prediction.get("method", "unknown"),
        "fallback_used":   prediction.get("fallback_used", False),
        "predicted_at":    now_iso(),
        "files":           ["raw.html", "features.json", "prediction.json"],
    }
    _save_predicted_mapping(pred_mapping)
    _print_result(prediction, save_dir)


def _print_result(prediction: dict, save_dir: Path,
                  scrape_failed: bool = False, fail_reason: str = "") -> None:
    """Print the prediction result box."""
    lc = {"list": "blue", "detail": "green", "others": "yellow"}.get(
        prediction["predicted_label"], "white"
    )
    mc = {
        "schema_org": "green", "open_graph": "green", "twitter_card": "green",
        "strong_listing": "cyan", "domain_specific": "cyan",
        "error_page": "red", "ml_model": "blue", "ml_model_url_fallback": "yellow",
        "scrape_failed": "red",
    }.get(prediction.get("method", ""), "white")

    console.print(f"\n  {'─'*50}")
    if scrape_failed:
        console.print(f"  [red bold]⚠  Scraping failed[/red bold]")
        console.print(f"  Reason: [dim]{fail_reason}[/dim]")
        console.print(f"  [dim]Page could not be scraped (anti-bot / network error).[/dim]")
    else:
        console.print(
            f"  Prediction:  [{lc}][bold]{prediction['predicted_label'].upper()}[/bold][/{lc}]"
        )
        console.print(f"  Confidence:  {prediction['confidence']:.1%}")
        console.print(f"  Method:      [{mc}]{prediction.get('method', '?')}[/{mc}]")
        if prediction.get("is_error_page"):
            console.print("  [red]⚠  Error/blocked page[/red]")
        if prediction.get("reason"):
            console.print(f"  Reason:      [dim]{prediction['reason']}[/dim]")
        if prediction.get("fallback_used"):
            console.print("  [yellow]⚠  URL-pattern fallback used (thin content)[/yellow]")
        if prediction.get("probabilities"):
            console.print("  Probabilities:")
            for cls, prob in sorted(prediction["probabilities"].items(), key=lambda x: -x[1]):
                console.print(f"    {cls:<10} {prob:.1%}  {'█' * int(prob * 20)}")
    console.print(f"  {'─'*50}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--url",   default=None,  help="URL to classify (omit for interactive loop)")
@click.option("--force", is_flag=True,  help="Re-classify even if already predicted")
@click.option("--debug", is_flag=True,  help="Show which layer decided")
def main(url, force, debug):
    """
    Classify webpages as list / detail / others.

    \b
    Interactive mode (no --url):
      python predict_run.py
      → keeps asking for URLs until you type: exit, quit, or q

    Single URL mode:
      python predict_run.py --url https://example.com
    """
    predicted_root = Path(cfg["paths"]["predicted_dir"])
    models_dir     = Path(cfg["paths"]["models_dir"])

    # Load model once — shared across all predictions in the loop
    model, le, feature_cols = _load_model_artefacts(models_dir)

    # ── Single URL mode ───────────────────────────────────────────────────────
    if url:
        console.rule("[bold blue]Webpage Classifier — Predict")
        _predict_url(url, force, debug, model, le, feature_cols, predicted_root, models_dir)
        return

    # ── Interactive loop mode ─────────────────────────────────────────────────
    console.rule("[bold blue]Webpage Classifier — Interactive Mode")
    console.print("  Type a URL and press Enter to classify it.")
    console.print("  Type [bold]exit[/bold], [bold]quit[/bold], or [bold]q[/bold] to stop.\n")

    while True:
        try:
            raw = input("  Enter URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted — goodbye.[/dim]")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        # --- NEW: parse flags from input ---
        force_flag = False
        debug_flag = False

        if "--force" in raw:
            force_flag = True
            raw = raw.replace("--force", "").strip()

        if "--debug" in raw:
            debug_flag = True
            raw = raw.replace("--debug", "").strip()
        # -----------------------------------

        # Auto-add https:// if user forgot
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
            console.print(f"  [dim]→ Using: {raw}[/dim]")

        try:
            _predict_url(
                raw,
                force_flag,   # 👈 use dynamic flag
                debug_flag,   # 👈 use dynamic flag
                model,
                le,
                feature_cols,
                predicted_root,
                models_dir
            )
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            logger.exception(f"Prediction error for {raw}")

if __name__ == "__main__":
    main()