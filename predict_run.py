"""
predict_run.py
──────────────
Two modes:

  1. Interactive loop (default):
       python predict_run.py
     → prompts "Enter URL:" repeatedly until you type exit/quit/q
     → supports inline flags: https://example.com --force --debug

  2. Single URL:
       python predict_run.py --url https://example.com
       python predict_run.py --url https://example.com --force
       python predict_run.py --url https://example.com --debug

Classification pipeline (v2 — dual-prediction with tie-breaker):

  LAYER 1 — Absolute rules (pre-classifier only, ML skipped):
    • Error / blocked page   → "others" (conf 0.97) — page doesn't exist
    • Schema.org JSON-LD     → direct class (conf 0.96) — developer declared

  LAYER 2 — Soft rules + ML validation (BOTH run, tie-breaker decides):
    • RSS feed               → pre says "list"  (conf 0.93) — ML also runs
    • Open Graph type        → pre says class    (conf 0.93) — ML also runs
    • Twitter card=player    → pre says "detail" (conf 0.93) — ML also runs
    • Strong listing (3/5)   → pre says "list"   (conf 0.90) — ML also runs
    • Domain-specific rules  → pre says class    (conf 0.86–0.90) — ML also runs
    • Others signals         → pre says "others" (conf 0.90) — ML also runs

  LAYER 3 — ML only (pre-classifier returned None):
    • XGBoost on 200+ features → predicted class + probabilities

  LAYER 4 — URL fallback (only if ML conf < 50% AND no content):
    • URL path/query pattern → weak class hint (conf 0.50–0.65)

TIE-BREAKER LOGIC (applies in Layer 2):
  • If pre and ML agree → use pre-classifier result (already confident)
  • If pre says X (conf >= 0.93) AND ML says Y (prob >= 0.70) AND Y != X:
      → ML wins — the hard rule is wrong for this page
      → method tagged as "ml_override_pre"
  • If pre says X (conf 0.86–0.92) AND ML says Y (prob >= 0.60) AND Y != X:
      → ML wins — soft rule overridden by ML evidence
      → method tagged as "ml_override_soft_rule"
  • If pre says X AND ML is uncertain (max prob < 0.60):
      → pre-classifier wins — ML doesn't have enough evidence to override
  • If both are low confidence:
      → blend: pick the class that both agree on, else prefer ML

CONFIDENCE THRESHOLDS:
  ABSOLUTE_CONF = 0.95    → Schema.org, error pages — ML never runs
  STRONG_CONF   = 0.93    → RSS, OG, Twitter — ML runs but needs 0.70 to override
  SOFT_CONF     = 0.90    → Strong listing, others signals — ML needs 0.60 to override
  ML_STRONG     = 0.70    → ML confidence to override a strong pre-classifier rule
  ML_SOFT       = 0.60    → ML confidence to override a soft pre-classifier rule
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import console, get_logger, load_config, now_iso, url_to_folder_name
from scraper.scraper import run_scraper
from extraction.feature_extractor import extract_features, _url_features
from extraction.pre_classifier import pre_classify

logger    = get_logger("predict_run")
cfg       = load_config()
PREDICTED_MAPPING_PATH = Path(cfg["paths"]["predicted_dir"]) / "mapping.json"

# ── Confidence thresholds ─────────────────────────────────────────────────────
ABSOLUTE_CONF = 0.95   # Schema.org / error — ML never runs
ML_STRONG     = 0.70   # ML prob needed to override a strong (0.93) pre rule
ML_SOFT       = 0.60   # ML prob needed to override a soft (0.86–0.92) pre rule


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_predicted_mapping():
    return json.loads(PREDICTED_MAPPING_PATH.read_text()) \
        if PREDICTED_MAPPING_PATH.exists() else {}


def _save_predicted_mapping(data):
    PREDICTED_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTED_MAPPING_PATH.write_text(json.dumps(data, indent=2))


def _load_model_artefacts(models_dir: Path):
    mp = models_dir / "classifier.joblib"
    if not mp.exists():
        logger.error("Model not found. Run: python train_run.py first.")
        sys.exit(1)
    return (
        joblib.load(mp),
        joblib.load(models_dir / "label_encoder.joblib"),
        json.loads((models_dir / "feature_columns.json").read_text()),
    )


def _url_pattern_fallback(url: str):
    f = _url_features(url)
    if f.get("url_has_search_param") or f.get("url_has_page_param"): return "list",   0.62
    if f.get("url_has_filter_param"):                                  return "list",   0.60
    if f.get("url_has_numeric_id") and not f.get("url_is_root"):       return "detail", 0.60
    if f.get("url_has_long_slug"):                                      return "detail", 0.58
    if f.get("url_is_root"):                                            return "others", 0.65
    return "others", 0.50


def _run_ml(features: dict, model, le, feature_cols: list, url: str) -> dict:
    """Run the XGBoost model and return full prediction dict."""
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
        "label":             lbl,
        "confidence":        round(conf, 4),
        "probabilities":     proba_dict,
        "method":            "ml_model" + ("_url_fallback" if fallback else ""),
        "fallback_used":     fallback,
        "content_available": bool(content_available),
    }


def _save_features(html: str, url: str, save_dir: Path) -> dict:
    """Extract features from HTML and save to features.json."""
    save_dir.mkdir(parents=True, exist_ok=True)
    pj_path   = save_dir / "page.json"
    page_json = json.loads(pj_path.read_text()) if pj_path.exists() else None
    features  = extract_features(html=html, url=url, page_json=page_json)
    (save_dir / "features.json").write_text(
        json.dumps(features, indent=2, ensure_ascii=False)
    )
    return features


def _apply_tiebreaker(pre: dict, ml: dict, debug: bool = False) -> dict:
    """
    Core tie-breaker logic.

    ABSOLUTE rules (Schema.org, error pages — conf >= 0.95):
      ML is not run at all — this function is never called for these.

    STRONG rules (RSS, OG, Twitter — conf 0.90–0.94):
      ML needs prob >= 0.70 to override. Below that, pre-classifier wins.

    SOFT rules (domain-specific, others signals — conf 0.86–0.89):
      ML needs prob >= 0.60 to override. Below that, pre-classifier wins.

    When ML overrides:
      The final result uses ML's label and confidence, but records what the
      pre-classifier said so you can audit the decision.

    Returns the final prediction dict.
    """
    pre_label  = pre["label"]
    pre_conf   = pre["confidence"]
    pre_method = pre["method"]

    ml_label = ml["label"]
    ml_conf  = ml["confidence"]
    ml_probs = ml.get("probabilities", {})

    # Same label — no conflict, keep pre-classifier (already high confidence)
    if pre_label == ml_label:
        result = {
            "predicted_label":   pre_label,
            "confidence":        pre_conf,
            "method":            pre_method,
            "reason":            pre.get("reason", ""),
            "probabilities":     ml_probs,         # include ML probs for transparency
            "fallback_used":     False,
            "content_available": ml.get("content_available", True),
            "pre_label":         pre_label,
            "pre_confidence":    pre_conf,
            "ml_label":          ml_label,
            "ml_confidence":     ml_conf,
            "tiebreaker":        "agreement",
        }
        if debug:
            console.print(f"  [green]✓ Agreement: pre={pre_label}({pre_conf:.0%}) "
                          f"ml={ml_label}({ml_conf:.0%})[/green]")
        return result

    # Conflict — apply threshold rules
    threshold = ML_STRONG if pre_conf >= 0.90 else ML_SOFT

    if ml_conf >= threshold:
        # ML overrides the pre-classifier rule
        override_method = (
            "ml_override_pre"       if pre_conf >= 0.90
            else "ml_override_soft_rule"
        )
        result = {
            "predicted_label":   ml_label,
            "confidence":        ml_conf,
            "method":            override_method,
            "reason":            (f"ML({ml_label}@{ml_conf:.0%}) overrode "
                                  f"pre({pre_label}@{pre_conf:.0%} via {pre_method})"),
            "probabilities":     ml_probs,
            "fallback_used":     ml.get("fallback_used", False),
            "content_available": ml.get("content_available", True),
            "pre_label":         pre_label,
            "pre_confidence":    pre_conf,
            "ml_label":          ml_label,
            "ml_confidence":     ml_conf,
            "tiebreaker":        "ml_wins",
        }
        if debug:
            console.print(
                f"  [yellow]⚡ ML overrides pre-classifier: "
                f"pre={pre_label}({pre_conf:.0%}) → ml={ml_label}({ml_conf:.0%})[/yellow]"
            )
        logger.info(
            f"ML override: pre={pre_label}({pre_conf:.2f}) → "
            f"ml={ml_label}({ml_conf:.2f}) via {pre_method}"
        )
    else:
        # Pre-classifier wins — ML not confident enough to override
        result = {
            "predicted_label":   pre_label,
            "confidence":        pre_conf,
            "method":            pre_method,
            "reason":            (f"pre({pre_label}@{pre_conf:.0%}) held; "
                                  f"ML({ml_label}@{ml_conf:.0%}) below threshold({threshold:.0%})"),
            "probabilities":     ml_probs,
            "fallback_used":     False,
            "content_available": ml.get("content_available", True),
            "pre_label":         pre_label,
            "pre_confidence":    pre_conf,
            "ml_label":          ml_label,
            "ml_confidence":     ml_conf,
            "tiebreaker":        "pre_wins",
        }
        if debug:
            console.print(
                f"  [cyan]pre-classifier holds: "
                f"pre={pre_label}({pre_conf:.0%}), "
                f"ml={ml_label}({ml_conf:.0%}) < threshold({threshold:.0%})[/cyan]"
            )

    return result


# ─── Core prediction logic ────────────────────────────────────────────────────

def _predict_url(
    url: str,
    force: bool,
    debug: bool,
    model,
    le,
    feature_cols: list,
    predicted_root: Path,
    models_dir: Path,
) -> None:
    """
    Full dual-prediction pipeline for one URL.
    Handles caching, scraping, pre-classification, ML, and tie-breaking.
    """
    url      = url.strip()
    folder   = url_to_folder_name(url)
    save_dir = predicted_root / folder
    pred_mapping = _load_predicted_mapping()

    # Cache check
    if folder in pred_mapping and not force:
        ex = pred_mapping[folder]
        if ex.get("status") == "success":
            console.print(
                f"\n[yellow]Already predicted → "
                f"[bold]{ex['predicted_label'].upper()}[/bold] "
                f"(conf: {ex['confidence']:.1%}, method: {ex.get('method','?')})[/yellow]"
            )
            console.print(f"[dim]  Use --force flag to re-run.[/dim]\n")
            return

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
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

    if not scrape_ok or not html.strip():
        fail_reason = sr.get("reason", "unknown")
        prediction  = {
            "predicted_label":   "others",
            "confidence":        0.0,
            "method":            "scrape_failed",
            "reason":            f"scraping_failed: {fail_reason}",
            "probabilities":     {},
            "fallback_used":     False,
            "content_available": False,
            "is_scrape_failed":  True,
        }
        _save_prediction(url, folder, save_dir, pred_mapping,
                         prediction, scrape_ok=False)
        _print_result(prediction, save_dir, scrape_failed=True,
                      fail_reason=fail_reason)
        return

    # ── Step 2: Pre-classifier ────────────────────────────────────────────────
    console.print("[dim]Classifying...[/dim]", end="\r")
    pre = pre_classify(html=html, url=url, http_status=None)

    # ── Step 3: Features (always extracted for ML + audit) ────────────────────
    features = _save_features(html, url, save_dir)

    # ── Step 4: Decision logic ────────────────────────────────────────────────
    if pre is None:
        # Pre-classifier found nothing → ML is the only decision maker
        ml_result  = _run_ml(features, model, le, feature_cols, url)
        prediction = {
            "predicted_label":   ml_result["label"],
            "confidence":        ml_result["confidence"],
            "method":            ml_result["method"],
            "reason":            "",
            "probabilities":     ml_result["probabilities"],
            "fallback_used":     ml_result["fallback_used"],
            "content_available": ml_result["content_available"],
            "pre_label":         None,
            "pre_confidence":    None,
            "ml_label":          ml_result["label"],
            "ml_confidence":     ml_result["confidence"],
            "tiebreaker":        "ml_only",
        }
        if debug:
            console.print(
                f"  [blue]ML only: {ml_result['label']} "
                f"({ml_result['confidence']:.0%})[/blue]"
            )

    elif pre["confidence"] >= ABSOLUTE_CONF:
        # ABSOLUTE rule (Schema.org, error pages) — ML is never run for these.
        # These signals are developer-declared ground truth. Overriding them
        # with ML would reduce accuracy.
        prediction = {
            "predicted_label":   pre["label"],
            "confidence":        pre["confidence"],
            "method":            pre["method"],
            "reason":            pre.get("reason", ""),
            "probabilities":     {},
            "fallback_used":     False,
            "content_available": bool(html),
            "is_error_page":     pre.get("is_error_page", False),
            "pre_label":         pre["label"],
            "pre_confidence":    pre["confidence"],
            "ml_label":          None,
            "ml_confidence":     None,
            "tiebreaker":        "absolute_rule",
        }
        if debug:
            console.print(
                f"  [green]Absolute rule: {pre['method']} → "
                f"{pre['label']} ({pre['confidence']:.0%})[/green]"
            )

    else:
        # SOFT rule (RSS, OG, Twitter, strong listing, domain-specific, others)
        # Run ML and apply tie-breaker — soft rules can be wrong.
        ml_result  = _run_ml(features, model, le, feature_cols, url)
        prediction = _apply_tiebreaker(pre, ml_result, debug=debug)
        prediction["is_error_page"] = pre.get("is_error_page", False)

    # ── Step 5: Save and display ──────────────────────────────────────────────
    _save_prediction(url, folder, save_dir, pred_mapping, prediction, scrape_ok=True)
    _print_result(prediction, save_dir)


def _save_prediction(
    url: str,
    folder: str,
    save_dir: Path,
    pred_mapping: dict,
    prediction: dict,
    scrape_ok: bool,
) -> None:
    """Persist prediction.json and update the mapping file."""
    pred_output = {
        "url":           url,
        "folder":        folder,
        "predicted_at":  now_iso(),
        "scrape_status": "success" if scrape_ok else "failed",
        **prediction,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "prediction.json").write_text(json.dumps(pred_output, indent=2))

    files = ["prediction.json"]
    if scrape_ok:
        files = ["raw.html", "features.json", "prediction.json"]

    pred_mapping[folder] = {
        "url":             url,
        "folder":          folder,
        "status":          "success" if scrape_ok else "scrape_failed",
        "predicted_label": prediction["predicted_label"],
        "confidence":      prediction["confidence"],
        "method":          prediction.get("method", "unknown"),
        "tiebreaker":      prediction.get("tiebreaker", ""),
        "fallback_used":   prediction.get("fallback_used", False),
        "predicted_at":    now_iso(),
        "files":           files,
    }
    _save_predicted_mapping(pred_mapping)


def _print_result(
    prediction: dict,
    save_dir: Path,
    scrape_failed: bool = False,
    fail_reason: str = "",
) -> None:
    """Render the prediction result box in the terminal."""
    lc = {"list": "blue", "detail": "green", "others": "yellow"}.get(
        prediction["predicted_label"], "white"
    )
    mc = {
        "schema_org":            "green",
        "open_graph":            "green",
        "twitter_card":          "green",
        "strong_listing":        "cyan",
        "domain_specific":       "cyan",
        "error_page":            "red",
        "ml_model":              "blue",
        "ml_model_url_fallback": "yellow",
        "ml_override_pre":       "magenta",
        "ml_override_soft_rule": "magenta",
        "scrape_failed":         "red",
    }.get(prediction.get("method", ""), "white")

    tiebreaker_labels = {
        "agreement":          "✓ Pre + ML agree",
        "ml_wins":            "⚡ ML overrode pre-classifier",
        "pre_wins":           "◆ Pre-classifier held (ML uncertain)",
        "ml_only":            "◇ ML only (no pre-classifier rule fired)",
        "absolute_rule":      "★ Absolute rule (Schema.org / error page)",
    }

    console.print(f"\n  {'─'*54}")
    if scrape_failed:
        console.print(f"  [red bold]⚠  Scraping failed[/red bold]")
        console.print(f"  Reason:   [dim]{fail_reason}[/dim]")
        console.print(f"  [dim]Page could not be scraped (anti-bot / network error).[/dim]")
    else:
        console.print(
            f"  Prediction:  [{lc}][bold]{prediction['predicted_label'].upper()}[/bold][/{lc}]"
        )
        console.print(f"  Confidence:  {prediction['confidence']:.1%}")
        console.print(f"  Method:      [{mc}]{prediction.get('method', '?')}[/{mc}]")

        # Tiebreaker summary
        tb = prediction.get("tiebreaker", "")
        if tb and tb != "absolute_rule":
            tb_label = tiebreaker_labels.get(tb, tb)
            tb_color = {"agreement": "green", "ml_wins": "magenta",
                        "pre_wins": "cyan", "ml_only": "blue"}.get(tb, "white")
            console.print(f"  Decision:    [{tb_color}]{tb_label}[/{tb_color}]")

        # Show both predictions when they differed
        pre_lbl  = prediction.get("pre_label")
        ml_lbl   = prediction.get("ml_label")
        pre_conf = prediction.get("pre_confidence")
        ml_conf  = prediction.get("ml_confidence")

        if pre_lbl and ml_lbl and pre_lbl != ml_lbl:
            console.print(
                f"  Pre-rule:    [dim]{pre_lbl} ({pre_conf:.0%})[/dim]  "
                f"ML: [dim]{ml_lbl} ({ml_conf:.0%})[/dim]"
            )
        elif pre_lbl and ml_lbl:
            console.print(
                f"  Pre+ML:      [dim]both said {pre_lbl} "
                f"(pre {pre_conf:.0%} / ml {ml_conf:.0%})[/dim]"
            )

        if prediction.get("is_error_page"):
            console.print("  [red]⚠  Error/blocked page[/red]")
        if prediction.get("reason"):
            console.print(f"  Reason:      [dim]{prediction['reason'][:90]}[/dim]")
        if prediction.get("fallback_used"):
            console.print("  [yellow]⚠  URL-pattern fallback used (thin content)[/yellow]")
        if prediction.get("probabilities"):
            console.print("  Probabilities:")
            for cls, prob in sorted(
                prediction["probabilities"].items(), key=lambda x: -x[1]
            ):
                console.print(f"    {cls:<10} {prob:.1%}  {'█' * int(prob * 20)}")

    console.print(f"  {'─'*54}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--url",   default=None, help="URL to classify (omit for interactive loop)")
@click.option("--force", is_flag=True, help="Re-classify even if already predicted")
@click.option("--debug", is_flag=True, help="Show which layer decided the prediction")
def main(url: str, force: bool, debug: bool):
    """
    Classify webpages as list / detail / others using dual-prediction pipeline.

    \b
    Interactive mode (no --url flag):
      python predict_run.py
      → keeps asking for URLs until you type: exit, quit, or q
      → inline flags supported: https://example.com --force --debug

    Single URL mode:
      python predict_run.py --url https://example.com [--force] [--debug]
    """
    predicted_root = Path(cfg["paths"]["predicted_dir"])
    models_dir     = Path(cfg["paths"]["models_dir"])

    # Load model once at startup — reused for every prediction in the loop
    model, le, feature_cols = _load_model_artefacts(models_dir)

    # ── Single URL mode ───────────────────────────────────────────────────────
    if url:
        console.rule("[bold blue]Webpage Classifier — Predict")
        _predict_url(url, force, debug, model, le, feature_cols,
                     predicted_root, models_dir)
        return

    # ── Interactive loop mode ─────────────────────────────────────────────────
    console.rule("[bold blue]Webpage Classifier — Interactive Mode")
    console.print("  Type a URL and press Enter to classify it.")
    console.print("  Inline flags supported: [dim]https://example.com --force --debug[/dim]")
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

        # Parse inline flags from input
        force_flag = "--force" in raw
        debug_flag = "--debug" in raw
        raw = raw.replace("--force", "").replace("--debug", "").strip()

        # Auto-add https:// if missing
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
            console.print(f"  [dim]→ Using: {raw}[/dim]")

        try:
            _predict_url(
                raw, force_flag, debug_flag,
                model, le, feature_cols,
                predicted_root, models_dir,
            )
        except Exception as exc:
            console.print(f"  [red]Error: {exc}[/red]")
            logger.exception(f"Prediction error for {raw}")


if __name__ == "__main__":
    main()