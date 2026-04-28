
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PWTimeout,
    async_playwright,
)

try:
    from playwright_stealth import stealth_async
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False

from pipeline.utils import get_logger, load_config, now_iso, update_checkpoint, update_mapping, url_to_folder_name

logger = get_logger("scraper")
cfg = load_config()
S = cfg["scraper"]

#  Block-detection patterns
BLOCK_SIGNALS = [
    "cf-browser-verification",
    "challenge-form",
    "cf_clearance",
    "ray id",
    "just a moment",
    "ddos-guard",
    "please verify you are a human",
    "access denied",
    "403 forbidden",
    "captcha",
    "enable javascript",
    "checking your browser",
    "security check",
    "bot detection",
    "are you a robot",
]


def _is_blocked(html: str, title: str) -> bool:
    combined = (html[:3000] + title).lower()
    return any(sig in combined for sig in BLOCK_SIGNALS)


def _random_delay(min_s: float = None, max_s: float = None) -> float:
    lo = min_s or S["random_delay_min_s"]
    hi = max_s or S["random_delay_max_s"]
    return random.uniform(lo, hi)


async def _human_behaviour(page: Page) -> None:
    """Simulate human-like actions to evade bot detection."""
    try:
        vw = S["viewport"]["width"]
        vh = S["viewport"]["height"]

        for _ in range(random.randint(2, 5)):
            x = random.randint(100, vw - 100)
            y = random.randint(100, vh - 100)
            await page.mouse.move(x, y, steps=random.randint(5, 15))
            await asyncio.sleep(random.uniform(0.05, 0.2))

        total_scroll = random.randint(300, 800)
        steps = random.randint(3, 7)
        for i in range(steps):
            delta = total_scroll // steps
            await page.evaluate(f"window.scrollBy(0, {delta})")
            await asyncio.sleep(random.uniform(0.1, 0.4))

        await page.evaluate(f"window.scrollBy(0, -{random.randint(100, 300)})")
        await asyncio.sleep(random.uniform(0.1, 0.3))
    except Exception:
        pass


async def _setup_context(browser: Browser) -> BrowserContext:
    """Create a fresh, hardened browser context."""
    ua = random.choice(S["user_agents"])
    vw = S["viewport"]["width"] + random.randint(-50, 50)
    vh = S["viewport"]["height"] + random.randint(-30, 30)

    context = await browser.new_context(
        user_agent=ua,
        viewport={"width": vw, "height": vh},
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
        accept_downloads=False,
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        },
    )

    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'permissions', {
            get: () => ({ query: () => Promise.resolve({ state: 'granted' }) })
        });
    """)
    return context


async def _inject_css_inline(page: Page) -> str:
    """Return full HTML with all external stylesheets inlined."""
    try:
        html_with_styles: str = await page.evaluate("""
            () => {
                const clone = document.cloneNode(true);
                let cssText = '';
                try {
                    for (const sheet of document.styleSheets) {
                        try {
                            for (const rule of sheet.cssRules) {
                                cssText += rule.cssText + '\\n';
                            }
                        } catch(e) {}
                    }
                } catch(e) {}
                const style = document.createElement('style');
                style.textContent = cssText;
                const head = clone.querySelector('head');
                if (head) head.appendChild(style);
                return clone.documentElement ? clone.documentElement.outerHTML : document.documentElement.outerHTML;
            }
        """)
        return f"<!DOCTYPE html>\n{html_with_styles}"
    except Exception:
        return await page.content()


async def _build_page_json(page: Page, url: str, title: str, html: str) -> Dict:
    """
    Build a complete structured JSON snapshot of the page.
    Captures everything needed to verify what was scraped without opening raw.html.
    """
    parsed = urlparse(url)
    domain = parsed.netloc

    #  Meta tags 
    try:
        meta = await page.evaluate("""
            () => {
                const metas = {};
                document.querySelectorAll('meta').forEach(m => {
                    const name = m.getAttribute('name') || m.getAttribute('property') || m.getAttribute('itemprop');
                    const content = m.getAttribute('content');
                    if (name && content) metas[name] = content;
                });
                return metas;
            }
        """)
    except Exception:
        meta = {}

    #  Open Graph specifically 
    try:
        open_graph = await page.evaluate("""
            () => {
                const og = {};
                document.querySelectorAll('meta[property^="og:"]').forEach(m => {
                    const key = m.getAttribute('property');
                    const val = m.getAttribute('content');
                    if (key && val) og[key] = val;
                });
                return og;
            }
        """)
    except Exception:
        open_graph = {}

    # Headings 
    try:
        headings = await page.evaluate("""
            () => {
                const result = {};
                ['h1','h2','h3','h4'].forEach(tag => {
                    result[tag] = Array.from(document.querySelectorAll(tag))
                        .map(el => el.innerText.trim())
                        .filter(Boolean);
                });
                return result;
            }
        """)
    except Exception:
        headings = {"h1": [], "h2": [], "h3": [], "h4": []}

    # All links 
    try:
        links = await page.evaluate(f"""
            () => {{
                const domain = '{domain}';
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => {{
                        const href = a.href || '';
                        let is_internal = false;
                        try {{
                            const u = new URL(href);
                            is_internal = u.hostname === domain || u.hostname === '';
                        }} catch(e) {{
                            is_internal = !href.startsWith('http');
                        }}
                        return {{
                            text: (a.innerText || a.textContent || '').trim().slice(0, 200),
                            href: href.slice(0, 500),
                            is_internal: is_internal,
                            rel: a.getAttribute('rel') || ''
                        }};
                    }})
                    .filter(l => l.href && l.href !== '#' && !l.href.startsWith('javascript:'));
            }}
        """)
    except Exception:
        links = []

    # All images 
    try:
        images = await page.evaluate("""
            () => Array.from(document.querySelectorAll('img'))
                .map(img => ({
                    src:     (img.src || img.getAttribute('data-src') || '').slice(0, 500),
                    alt:     (img.alt || '').trim().slice(0, 300),
                    width:   img.naturalWidth || img.width || 0,
                    height:  img.naturalHeight || img.height || 0,
                    loading: img.getAttribute('loading') || ''
                }))
                .filter(img => img.src)
        """)
    except Exception:
        images = []

    # Body text (visible, cleaned) 
    try:
        body_text = await page.evaluate("""
            () => {
                const el = document.body;
                if (!el) return '';
                // Remove script/style/noscript before getting text
                const clone = el.cloneNode(true);
                clone.querySelectorAll('script, style, noscript, svg').forEach(e => e.remove());
                return clone.innerText
                    .replace(/\\s+/g, ' ')
                    .trim()
                    .slice(0, 50000);
            }
        """)
    except Exception:
        body_text = ""

    # Schema.org JSON-LD 
    try:
        schema_raw = await page.evaluate("""
            () => Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                .map(s => { try { return JSON.parse(s.textContent); } catch(e) { return null; } })
                .filter(Boolean)
        """)
        # Extract just the types for a quick summary
        schema_types = []
        def _collect_types(obj):
            if isinstance(obj, dict):
                t = obj.get("@type")
                if isinstance(t, list):
                    schema_types.extend(t)
                elif t:
                    schema_types.append(t)
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        _collect_types(v)
            elif isinstance(obj, list):
                for item in obj:
                    _collect_types(item)
        _collect_types(schema_raw)
        schema_org = {"raw": schema_raw, "types_found": schema_types}
    except Exception:
        schema_org = {"raw": [], "types_found": []}

    # Canonical URL 
    try:
        canonical = await page.evaluate("""
            () => {
                const el = document.querySelector('link[rel="canonical"]');
                return el ? el.href : null;
            }
        """)
    except Exception:
        canonical = None

    # Pagination signals 
    try:
        pagination = await page.evaluate("""
            () => {
                const relLinks = Array.from(document.querySelectorAll('link[rel]'))
                    .filter(l => {
                        const rel = (l.getAttribute('rel') || '').toLowerCase();
                        return rel === 'next' || rel === 'prev' || rel === 'previous';
                    })
                    .map(l => ({ rel: l.getAttribute('rel'), href: l.href }));
                return {
                    has_next: relLinks.some(l => l.rel === 'next'),
                    has_prev: relLinks.some(l => l.rel === 'prev' || l.rel === 'previous'),
                    rel_links: relLinks
                };
            }
        """)
    except Exception:
        pagination = {"has_next": False, "has_prev": False, "rel_links": []}

    #  Forms 
    try:
        forms = await page.evaluate("""
            () => Array.from(document.querySelectorAll('form')).map(form => ({
                action: form.action || '',
                method: form.method || 'get',
                inputs: Array.from(form.querySelectorAll('input, select, textarea')).map(inp => ({
                    type:        inp.type || inp.tagName.toLowerCase(),
                    name:        inp.name || '',
                    placeholder: inp.placeholder || '',
                    required:    inp.required || false
                }))
            }))
        """)
    except Exception:
        forms = []

    # Counts summary 
    counts = {
        "links":   len(links),
        "images":  len(images),
        "forms":   len(forms),
        "scripts": len(await page.query_selector_all("script")) if True else 0,
        "h1": len(headings.get("h1", [])),
        "h2": len(headings.get("h2", [])),
        "h3": len(headings.get("h3", [])),
        "h4": len(headings.get("h4", [])),
        "body_text_chars": len(body_text),
        "body_text_words": len(body_text.split()),
    }

    try:
        counts["scripts"] = await page.evaluate("() => document.querySelectorAll('script').length")
    except Exception:
        pass

    return {
        "url":         url,
        "title":       title,
        "domain":      domain,
        "path":        parsed.path,
        "query":       parsed.query,
        "scraped_at":  now_iso(),
        "html_length": len(html),
        "status":      "success",
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
        # Content
        "links":       links,
        "images":      images,
        "body_text":   body_text,
    }


async def scrape_url(
    url: str,
    save_dir: Path,
    semaphore: asyncio.Semaphore,
    browser: Browser,
) -> Tuple[bool, str]:
    """
    Scrape a single URL.
    Returns (success: bool, reason: str).
    Saves:
      - raw.html        : full page HTML with inlined CSS
      - page.json       : complete structured snapshot (v2)
    """
    async with semaphore:
        max_retries = S["max_retries"]
        retry_delay = S["retry_delay_s"]
        last_error = "unknown"

        for attempt in range(1, max_retries + 1):
            context: Optional[BrowserContext] = None
            page: Optional[Page] = None
            try:
                logger.info(f"[attempt {attempt}/{max_retries}] Scraping: {url}")

                await asyncio.sleep(_random_delay())

                context = await _setup_context(browser)
                page = await context.new_page()

                if _STEALTH_AVAILABLE and S.get("stealth", True):
                    await stealth_async(page)

                await page.route(
                    re.compile(r"\.(woff2?|ttf|eot|mp4|mp3|avi|webm|flv)(\?.*)?$", re.I),
                    lambda route: route.abort()
                )

                response = await page.goto(
                    url,
                    timeout=S["timeout_ms"],
                    wait_until="domcontentloaded",
                )

                try:
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=S["wait_after_load_ms"] * 2,
                    )
                except PWTimeout:
                    pass

                await asyncio.sleep(S["wait_after_load_ms"] / 1000)

                await _human_behaviour(page)

                title = await page.title()
                raw_html = await _inject_css_inline(page)

                if _is_blocked(raw_html, title):
                    logger.warning(f"Block detected on attempt {attempt}: {url}")
                    last_error = "blocked"
                    await context.close()
                    if attempt < max_retries:
                        wait = retry_delay * attempt * random.uniform(1.0, 2.0)
                        logger.info(f"Waiting {wait:.1f}s before retry...")
                        await asyncio.sleep(wait)
                    continue

                if response and response.status >= 400:
                    last_error = f"http_{response.status}"
                    logger.warning(f"HTTP {response.status} on attempt {attempt}: {url}")
                    await context.close()
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay * attempt)
                    continue

                save_dir.mkdir(parents=True, exist_ok=True)

                html_path = save_dir / "raw.html"
                html_path.write_text(raw_html, encoding="utf-8")

                page_json = await _build_page_json(page, url, title, raw_html)
                json_path = save_dir / "page.json"
                json_path.write_text(
                    json.dumps(page_json, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                await context.close()
                logger.info(f"✓ Scraped successfully: {url}")
                return True, "success"

            except PWTimeout:
                last_error = "timeout"
                logger.warning(f"Timeout on attempt {attempt}: {url}")
            except Exception as e:
                last_error = str(e)[:120]
                logger.error(f"Error on attempt {attempt}: {url} → {last_error}")
            finally:
                try:
                    if page and not page.is_closed():
                        await page.close()
                    if context:
                        await context.close()
                except Exception:
                    pass

            if attempt < max_retries:
                wait = retry_delay * attempt
                await asyncio.sleep(wait)

        logger.error(f"✗ All {max_retries} attempts failed: {url} | last_error={last_error}")
        return False, last_error


async def run_scraper(
    urls_with_meta: list[Dict],
    output_root: Path,
) -> Dict[str, Dict]:
    """
    Scrape all URLs in urls_with_meta.
    Each item: {"url": ..., "label": ..., "folder": ..., "save_dir": Path}
    Returns results dict keyed by url.
    """
    concurrency = S.get("concurrency", 3)
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[str, Dict] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=S["headless"],
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1366,768",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-web-security",
            ],
        )

        tasks = [
            scrape_url(
                url=item["url"],
                save_dir=item["save_dir"],
                semaphore=semaphore,
                browser=browser,
            )
            for item in urls_with_meta
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, result in zip(urls_with_meta, raw_results):
            url = item["url"]
            if isinstance(result, Exception):
                results[url] = {"status": "failed", "reason": str(result)[:120]}
            else:
                success, reason = result
                results[url] = {
                    "status": "success" if success else "failed",
                    "reason": reason,
                }

        await browser.close()

    return results