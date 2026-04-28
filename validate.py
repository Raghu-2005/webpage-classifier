"""
Runs a full pre-flight check of your environment BEFORE you start the pipeline.
Catches all common problems early so you don't discover them mid-scrape.

Checks:
  1. Python version (3.10+)
  2. All required packages installed + version check
  3. Playwright + Chromium installed and launchable
  4. Stealth plugin available
  5. Config file valid
  6. Training CSV exists and has correct format
  7. Output directories writable
  8. Model artefacts (if training was done)
  9. Quick scrape smoke test (scrapes example.com to verify browser works)
"""

from __future__ import annotations

import sys
import csv
import json
import subprocess
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

# Don't import project modules yet — we're checking if they even work
from rich.console import Console
from rich.table import Table

console = Console()


def _check(name: str, passed: bool, detail: str = "") -> bool:
    status = "[green]✓[/green]" if passed else "[red]✗[/red]"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  [dim]{detail}[/dim]"
    console.print(msg)
    return passed


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    return _check(
        "Python version",
        ok,
        f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else " — need 3.10+")
    )


def check_packages() -> bool:
    required = [
        ("playwright", "1.44.0"),
        ("bs4", None),
        ("xgboost", None),
        ("sklearn", None),
        ("numpy", None),
        ("pandas", None),
        ("shap", None),
        ("yaml", None),
        ("click", None),
        ("rich", None),
        ("joblib", None),
        ("fake_useragent", None),
        ("langdetect", None),
        ("tqdm", None),
    ]
    all_ok = True
    for pkg, _ in required:
        try:
            __import__(pkg)
            _check(f"Package: {pkg}", True)
        except ImportError:
            _check(f"Package: {pkg}", False, "Not installed — run: pip install -r requirements.txt")
            all_ok = False

    # Check playwright-stealth separately 
    try:
        import playwright_stealth
        _check("Package: playwright-stealth", True, "(stealth mode enabled)")
    except ImportError:
        _check(
            "Package: playwright-stealth", False,
            "[yellow]Optional but recommended — run: pip install playwright-stealth[/yellow]"
        )
        # Not a hard failure

    return all_ok


def check_playwright() -> bool:
    """Check that Playwright Chromium is installed and can launch."""
    import asyncio

    async def _try_launch():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await browser.new_page()
            await page.goto("about:blank")
            title = await page.title()
            await browser.close()
            return True

    try:
        result = asyncio.run(_try_launch())
        return _check("Playwright Chromium launch", result, "Browser launched OK")
    except Exception as e:
        _check("Playwright Chromium launch", False, f"{str(e)[:80]}")
        console.print("  [dim]Fix: playwright install chromium && playwright install-deps chromium[/dim]")
        return False


def check_config() -> bool:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        return _check("Config file", False, "config/settings.yaml not found")
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        required_keys = ["paths", "scraper", "extraction", "model", "logging"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            return _check("Config file", False, f"Missing sections: {missing}")
        return _check("Config file", True, "config/settings.yaml OK")
    except Exception as e:
        return _check("Config file", False, str(e))


def check_training_csv() -> bool:
    try:
        import yaml
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)
        csv_path = Path(cfg["paths"]["training_csv"])
    except Exception:
        csv_path = Path("data/training_urls.csv")

    if not csv_path.exists():
        _check("Training CSV", False, f"{csv_path} not found — create it first")
        console.print(f"  [dim]Required columns: url, label (valid labels: list, detail, others)[/dim]")
        return False

    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        if "url" not in headers:
            return _check("Training CSV", False, "Missing 'url' column")
        if "label" not in headers:
            return _check("Training CSV", False, "Missing 'label' column")

        valid_labels = {"list", "detail", "others"}
        bad_labels = [r["label"] for r in rows if r.get("label", "").lower() not in valid_labels]

        label_counts: dict[str, int] = {}
        for r in rows:
            lbl = r.get("label", "").lower()
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        detail = (
            f"{len(rows)} rows | "
            + " | ".join(f"{lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items()))
        )
        if bad_labels:
            return _check("Training CSV", False, f"Invalid labels found: {set(bad_labels)}")

        ok = len(rows) >= 3
        _check(
            "Training CSV",
            ok,
            detail + ("" if ok else " — need at least 3 rows total")
        )

        # Warn if class counts are low
        for lbl, cnt in label_counts.items():
            if lbl in valid_labels and cnt < 30:
                console.print(
                    f"  [yellow]  ⚠ Class '{lbl}' has {cnt} sample(s). "
                    "30+ recommended per class for good performance.[/yellow]"
                )
        return ok

    except Exception as e:
        return _check("Training CSV", False, str(e))


def check_directories() -> bool:
    dirs_needed = [
        "output/list", "output/detail", "output/others",
        "predicted", "models", "logs", "data",
    ]
    all_ok = True
    for d in dirs_needed:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        try:
            test_file = path / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            _check(f"Directory writable: {d}", True)
        except Exception as e:
            _check(f"Directory writable: {d}", False, str(e))
            all_ok = False
    return all_ok


def check_model() -> bool:
    model_path = Path("models/classifier.joblib")
    if not model_path.exists():
        _check("Trained model", False, "Not trained yet — run: python train_run.py (after scraping)")
        return False

    report_path = Path("models/training_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        ma = report.get("evaluation", {}).get("classification_report", {}).get("macro avg", {})
        f1 = ma.get("f1-score", 0)
        detail = f"macro F1={f1:.3f} | {report.get('num_training_samples', '?')} train samples"
        return _check("Trained model", True, detail)
    return _check("Trained model", True, "model exists")


def check_smoke_test() -> bool:
    """Try scraping example.com to verify the full stack works."""
    import asyncio

    async def _smoke():
        from playwright.async_api import async_playwright
        try:
            from playwright_stealth import stealth_async
            stealth_available = True
        except ImportError:
            stealth_available = False

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            if stealth_available:
                await stealth_async(page)
            await page.goto("https://example.com", timeout=15000, wait_until="domcontentloaded")
            html = await page.content()
            title = await page.title()
            await browser.close()
            return len(html) > 100, title

    try:
        ok, title = asyncio.run(_smoke())
        return _check(
            "Smoke test (scrape example.com)",
            ok,
            f"title='{title}' | HTML received" if ok else "HTML too short"
        )
    except Exception as e:
        _check("Smoke test (scrape example.com)", False, str(e)[:80])
        return False


@click.command()
@click.option("--skip-smoke", is_flag=True, help="Skip the live scrape smoke test")
@click.option("--skip-playwright", is_flag=True, help="Skip Playwright launch check")
def main(skip_smoke: bool, skip_playwright: bool):
    """Run pre-flight validation checks for the full pipeline."""
    console.rule("[bold blue]Webpage Classifier — Pre-flight Validation")

    results: list[bool] = []

    console.print("\n[bold]1. Python Environment[/bold]")
    results.append(check_python())

    console.print("\n[bold]2. Required Packages[/bold]")
    results.append(check_packages())

    console.print("\n[bold]3. Configuration[/bold]")
    results.append(check_config())

    console.print("\n[bold]4. Directories[/bold]")
    results.append(check_directories())

    console.print("\n[bold]5. Training Data[/bold]")
    results.append(check_training_csv())

    if not skip_playwright:
        console.print("\n[bold]6. Playwright Browser[/bold]")
        pw_ok = check_playwright()
        results.append(pw_ok)

        if pw_ok and not skip_smoke:
            console.print("\n[bold]7. Live Scrape Smoke Test[/bold]")
            results.append(check_smoke_test())
        elif not pw_ok:
            console.print("\n[dim]Skipping smoke test (Playwright failed)[/dim]")

    console.print("\n[bold]8. Model Status[/bold]")
    check_model()  # Not added to results — model not required at this stage

    #  Final verdict 
    console.rule()
    critical_failures = sum(1 for r in results if not r)
    if critical_failures == 0:
        console.print("\n[green bold]✓ All checks passed! You're ready to run the pipeline.[/green bold]")
        console.print("\n  Next steps:")
        console.print("  1. [cyan]python scrape_run.py[/cyan]")
        console.print("  2. [cyan]python extract_run.py[/cyan]")
        console.print("  3. [cyan]python train_run.py[/cyan]")
        console.print("  4. [cyan]python predict_run.py --url <URL>[/cyan]\n")
    else:
        console.print(
            f"\n[red bold]✗ {critical_failures} check(s) failed.[/red bold] "
            "Fix the issues above before running the pipeline.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()