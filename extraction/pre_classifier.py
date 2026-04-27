"""
extraction/pre_classifier.py
─────────────────────────────
Deterministic pre-classifier that runs BEFORE the ML model.

Priority order (v8 — final):
  1. Error / blocked page detection   → "others" + error flag      (conf 0.97)
  2. Schema.org / JSON-LD             → direct classification       (conf 0.96)
  3. RSS / Atom feed link             → list (skipped for listing URLs) (conf 0.93)
  4. Open Graph type                  → direct (skipped for listing URLs) (conf 0.93)
  5. Twitter Card = player            → detail                      (conf 0.93)
  6. Strong listing (3-of-5 signals)  → list                        (conf 0.90)
  7. Domain-specific rules            → job boards, social browse   (conf 0.86–0.90)
  8. Strong 'others' signals          → others (with guards)        (conf 0.90)
  ── Everything else → ML model ──

RULES REMOVED (now ML features only — do not add back):
  - rel=next/prev hard rule (REMOVED v8):
      Documentation sites (Python docs, MDN, Django docs, Wikipedia) use
      rel=next for chapter/article navigation. Each chapter IS a detail page.
      Blog posts use rel=next for previous/next post. Product pages use it
      for prev/next product in category. rel=next is ambiguous — it does NOT
      reliably mean "this page is a paginated listing". The ML model sees
      rel_has_next and rel_has_prev as features in pillar 12 and can weigh
      them properly alongside all other signals.
  - content_context hard rule (REMOVED v7):
      Caused Kaggle /datasets and similar pages to be called "others".
      Those signals (doc_signals, forum_signals) are still ML features.
  - strong_detail hard rule (REMOVED v4):
      False positives on Spotify, Apple, python.org, shopify.com, glassdoor,
      reddit, arxiv list pages. Now ML features.

GUARDS (protect against false positives on weak rules):
  - _is_listing_url(): RSS and OG rules skip when URL clearly signals listing.
    Prevents search pages with RSS feeds or template OG tags from being
    wrongly forced into list/detail before ML runs.
  - Detail-page guard on others rule: if page has price/buy-CTA/rating/specs,
    skip the others rule — it is a product/detail page, not a marketing page.
  - Marketing requirement on others rule: requires pricing/plans/enterprise
    vocabulary so true product pages with footers don't get called "others".

v8 CHANGELOG:
  - REMOVED rel=next/prev as hard rule. Python docs, Wikipedia articles,
    blog posts all have rel=next but are detail pages, not listing pages.
    Kept as ML features rel_has_next, rel_has_prev in feature_extractor.py.
  - All other fixes from v6/v7 retained (listing URL escape, detail guard,
    marketing requirement, raised others threshold).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from pipeline.utils import get_logger

logger = get_logger("pre_classifier")

CONF_SCHEMA  = 0.96
CONF_OG      = 0.93
CONF_TWITTER = 0.93
CONF_STRONG  = 0.90

ERROR_SIGNALS = [
    r"\b404\b.*\bnot found\b",
    r"\bpage not found\b",
    r"\b404 error\b",
    r"\bnot found\b.*\b404\b",
    r"\bthis page (doesn'?t|does not) exist\b",
    r"\bpage (has been )?removed\b",
    r"\bno longer available\b",
    r"\bfile not found\b",
    r"\b500\b.*\binternal server error\b",
    r"\bservice unavailable\b",
    r"\baccess denied\b",
    r"\bforbidden\b",
    r"\b403\b.*\bforbidden\b",
    r"\bcaptcha\b",
    r"\bjust a moment\b",
    r"\bchecking your browser\b",
    r"\bddos.guard\b",
    r"\bcf-browser-verification\b",
    r"\bare you a robot\b",
    r"\bverify you are human\b",
]

# Only unambiguous Schema.org types.
# Removed: organization, person, profilepage, localbusiness, restaurant,
# hotel, lodgingbusiness, imageobject, breadcrumblist — too ambiguous.
SCHEMA_TO_CLASS: Dict[str, str] = {
    "product":             "detail",
    "productgroup":        "detail",
    "offer":               "detail",
    "newsarticle":         "detail",
    "blogposting":         "detail",
    "technicalarticle":    "detail",
    "scholarlyarticle":    "detail",
    "recipe":              "detail",
    "jobposting":          "detail",
    "event":               "detail",
    "movieseries":         "detail",
    "movie":               "detail",
    "tvseries":            "detail",
    "tvepisode":           "detail",
    "book":                "detail",
    "musicrecording":      "detail",
    "musicalbum":          "detail",
    "podcast":             "detail",
    "podcastepisode":      "detail",
    "realestate":          "detail",
    "apartment":           "detail",
    "house":               "detail",
    "medicalcondition":    "detail",
    "drug":                "detail",
    "howto":               "detail",
    "faqpage":             "detail",
    "qapage":              "detail",
    "softwareapplication": "detail",
    "videoobject":         "detail",
    "course":              "detail",
    "vehicle":             "detail",
    "automobile":          "detail",
    "searchresultspage":   "list",
    "itemlist":            "list",
    "collectionpage":      "list",
    "productcollection":   "list",
    "offerscatalog":       "list",
    "dataset":             "list",
    "datafeed":            "list",
}

OG_TYPE_PREFIX_MAP: List[Tuple[str, str, float]] = [
    ("video.",     "detail", CONF_OG),
    ("music.",     "detail", CONF_OG),
    ("book",       "detail", CONF_OG),
    ("product",    "detail", CONF_OG),
    ("profile",    "detail", CONF_OG),
    ("restaurant", "detail", CONF_OG),
    ("business.",  "detail", CONF_OG),
    ("itemlist",   "list",   CONF_OG),
]


# ─── URL Listing-Intent Helper ────────────────────────────────────────────────
def _is_listing_url(url: str) -> bool:
    """
    Returns True when the URL structure unambiguously signals a search,
    browse, or listing page — regardless of metadata tags.

    Used to skip RSS and OG rules when the URL already declares listing intent.
    A search page that exposes an RSS feed is still a search page — the RSS
    rule should not force it to "list" with high confidence.

    Returns False for documentation pages, blog posts, product pages — even
    if they have rel=next or an RSS feed, those are not listing pages.
    """
    parsed = urlparse(url)
    path   = parsed.path.lower()
    query  = parsed.query.lower()
    netloc = parsed.netloc.lower()

    # Explicit search/filter query parameters
    # k= is Amazon's search param; field-keywords is also Amazon
    if re.search(r"(^|&)(q|query|search|s|find|keyword|term|k|field-keywords)=", query):
        return True
    if re.search(r"(^|&)(type|category|filter|sort|page|offset|c)=", query):
        return True

    # Path contains a listing-intent segment
    if re.search(
        r"/(search|results?|listings?|categories|category|browse|"
        r"discover|explore|publications?|exhibitions?|datasets?|"
        r"topics?|tags?|genre|archive|year|list[/-])/?",
        path
    ):
        return True

    # Path ends with a listing noun (terminal segment)
    if re.search(
        r"/(jobs|careers|vacancies|openings|products|items|"
        r"hotels|recipes|articles|posts|news|papers?|reports?)/?$",
        path
    ):
        return True

    # Job board subdomains — entire site is job listings
    if re.search(
        r"(boards\.|jobs\.|careers\.)(greenhouse\.io|lever\.co|"
        r"workable\.com|smartrecruiters\.com)",
        netloc
    ):
        return True

    return False


# ─── Schema.org extraction ────────────────────────────────────────────────────
def _extract_schema_types(soup: BeautifulSoup) -> List[str]:
    types: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw  = script.string or ""
            data = json.loads(raw)

            def _collect(obj):
                if isinstance(obj, dict):
                    t = obj.get("@type", "")
                    if isinstance(t, list):
                        types.extend([x.lower().strip() for x in t])
                    elif t:
                        types.append(str(t).lower().strip())
                    for v in obj.values():
                        if isinstance(v, (dict, list)):
                            _collect(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _collect(item)
            _collect(data)
        except Exception:
            pass

    for el in soup.find_all(itemtype=True):
        raw   = el.get("itemtype", "")
        parts = raw.lower().rstrip("/").split("/")
        if parts:
            types.append(parts[-1].strip())

    return [t for t in types if t]


# ─── Error page detection ─────────────────────────────────────────────────────
def _detect_error_page(html: str, title: str, http_status: Optional[int]) -> bool:
    if http_status and http_status >= 400:
        return True
    combined = (title + " " + html[:2000]).lower()
    for pattern in ERROR_SIGNALS:
        if re.search(pattern, combined, re.I):
            return True
    return False


# ─── RSS / Atom feed ─────────────────────────────────────────────────────────
def _has_feed_link(soup: BeautifulSoup) -> bool:
    """
    RSS or Atom <link> in <head> signals content aggregator / listing page.
    SKIPPED when _is_listing_url() returns True — search pages that also
    expose RSS feeds should go to ML, not be forced to "list".
    Confidence: 0.93
    """
    for link in soup.find_all("link", rel=True):
        rel_val = link.get("rel", [])
        if isinstance(rel_val, str):
            rel_val = [rel_val]
        rel_lower = [r.lower() for r in rel_val]
        link_type = link.get("type", "").lower()
        if "alternate" in rel_lower and ("rss" in link_type or "atom" in link_type):
            return True
    return False


# ─── Strong listing pattern (3-of-5) ─────────────────────────────────────────
def _strong_listing_pattern(soup: BeautifulSoup, body_text: str) -> bool:
    """
    Requires 3 of 5 structural listing signals simultaneously.
    Correct on: Amazon, Flipkart, IMDB search, Glassdoor jobs, Booking.com,
    Google search results.
    """
    lower   = body_text.lower()
    signals = 0

    # Signal 1: pagination UI
    if (re.search(r"\b(next\s*page|previous\s*page|load\s*more|showing\s+\d+)", lower) or
            soup.find(class_=re.compile(r"\bpaginat", re.I)) or
            soup.find(attrs={"aria-label": re.compile(r"pagination", re.I)})):
        signals += 1

    # Signal 2: result count text
    if re.search(
        r"\b(\d[\d,]*)\s*(results?|listings?|items?\s+found|products?\s+found|"
        r"jobs?\s+found|matches|properties?\s+found)\b",
        lower
    ):
        signals += 1

    # Signal 3: filter/facet UI
    if (soup.find(class_=re.compile(r"\b(filter|facet|refine)\b", re.I)) or
            soup.find(id=re.compile(r"\b(filter|facet)\b", re.I))):
        signals += 1

    # Signal 4: repeated sibling containers
    from bs4 import Tag as BsTag
    repeated = 0
    for container in soup.find_all(["ul", "ol", "div", "section"])[:200]:
        children = [c for c in container.children if isinstance(c, BsTag)]
        if len(children) >= 5:
            counts: Dict[str, int] = {}
            for c in children:
                counts[c.name] = counts.get(c.name, 0) + 1
            if max(counts.values()) >= 5:
                repeated += 1
    if repeated >= 2:
        signals += 1

    # Signal 5: grid/card CSS class names
    grid_pat = re.compile(
        r"\b(product[-_]?(?:item|card|tile)|result[-_]?item|"
        r"listing[-_]?item|job[-_]?item|card[-_]?item)\b", re.I
    )
    grid_count = sum(
        1 for t in soup.find_all(class_=True)
        if any(grid_pat.search(c) for c in t.get("class", []))
    )
    if grid_count >= 4:
        signals += 1

    return signals >= 3


# ─── Domain-specific rules ────────────────────────────────────────────────────
def _domain_specific_rules(
    url: str, body_text: str, soup: BeautifulSoup
) -> Optional[Tuple[str, float, str]]:
    """
    High-confidence domain-specific patterns for platforms with known structure.
    """
    lower_url  = url.lower()
    lower_body = body_text.lower()

    # Job boards — the domain itself is sufficient to declare a listing
    job_board_domains = {
        "lever.co":              CONF_STRONG,
        "greenhouse.io":         CONF_STRONG,
        "workable.com":          CONF_STRONG,
        "smartrecruiters.com":   CONF_STRONG,
        "applytojob.com":        CONF_STRONG,
        "jobs.lever.co":         CONF_STRONG,
        "boards.greenhouse.io":  CONF_STRONG,
        "jobs.greenhouse.io":    CONF_STRONG,
        "monster.com":           CONF_STRONG,
        "indeed.com":            CONF_STRONG,
        "glassdoor.com":         CONF_STRONG,
        "linkedin.com/jobs":     CONF_STRONG,
        "ziprecruiter.com":      CONF_STRONG,
        "bamboohr.com":          CONF_STRONG,
    }
    for domain, conf in job_board_domains.items():
        if domain in lower_url:
            if "/jobs" in lower_url or "/search" in lower_url or "/careers" in lower_url:
                return ("list", conf, f"job_board_domain={domain}")
            if "boards.greenhouse.io" in lower_url or "jobs.lever.co" in lower_url:
                return ("list", conf, f"job_board_domain={domain}")

    # API / Developer documentation — specific well-known platforms only
    api_doc_tuples = [
        ("docs.python.org",       "Python docs",  "/library/"),
        ("developer.mozilla.org", "MDN Docs",     "/en-US/docs/"),
        ("docs.aws.amazon.com",   "AWS Docs",     "/docs/"),
        ("github.com/docs",       "GitHub Docs",  "/docs/"),
        ("api.github.com",        "GitHub API",   ""),
    ]
    for domain, name, path_marker in api_doc_tuples:
        if domain in lower_url:
            if not path_marker or path_marker in lower_url:
                if bool(re.search(
                    r"\b(function|method|parameter|api|endpoint|return|"
                    r"exception|class|module)\b",
                    lower_body
                )):
                    return ("detail", 0.92, f"api_doc_platform={name}")

    # Academic / research listing pages
    if "arxiv.org" in lower_url and re.search(r"/(year|list|search|find)/", lower_url):
        return ("list", 0.88, "arxiv_listings")

    # Wikipedia — individual article pages only
    if "wikipedia.org" in lower_url and "/wiki/" in lower_url \
            and "/wiki/Special:" not in lower_url:
        after_wiki = lower_url.split("/wiki/")[1]
        if ":" not in after_wiki or "/" not in after_wiki:
            return ("detail", 0.85, "wikipedia_article")

    # Social media category / browse / discover pages
    social_browse_domains = {
        "vimeo.com/categories":    "video_categories",
        "vimeo.com/category":      "video_category",
        "music.apple.com/browse":  "music_browse",
        "soundcloud.com/discover": "music_discover",
        "soundcloud.com/stream":   "music_stream",
        "dribbble.com/tags":       "design_tags",
        "behance.net":             "portfolio_listing",
        "spotify.com/genre":       "music_genre",
        "bandcamp.com/tags":       "music_tags",
    }
    for domain, type_name in social_browse_domains.items():
        if domain in lower_url:
            return ("list", 0.88, f"social_browse={type_name}")

    # Generic browse/discover/exhibition paths on any domain
    # Guarded by link count to avoid mis-firing on thin pages
    browse_path_pat = re.compile(
        r"/(categories|category|discover|exhibitions?|browse|collections?|"
        r"explore|shows?|events?|publications?|archives?)/?$",
        re.I
    )
    parsed_url = urlparse(url)
    if browse_path_pat.search(parsed_url.path):
        if len(soup.find_all("a", href=True)) >= 15:
            return ("list", 0.86, f"browse_path={parsed_url.path}")

    return None


# ─── Main pre_classify function ───────────────────────────────────────────────
def pre_classify(
    html: str,
    url: str,
    http_status: Optional[int] = None,
) -> Optional[Dict]:
    """
    Run deterministic pre-classification before the ML model.

    Returns dict(label, confidence, method, reason) or None.
    None = no unambiguous rule fired = caller runs ML model.

    Key design principle: when in doubt, return None.
    The ML model with 180+ features is more reliable than any single rule.
    Hard rules are only for cases where the signal is developer-declared
    (Schema.org) or structurally overwhelming (3-of-5 listing signals).
    """
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body_text  = soup.get_text(separator=" ", strip=True)
    title      = soup.title.string.strip() if soup.title and soup.title.string else ""
    lower_html = html[:3000].lower()
    lower      = body_text.lower()

    # Pre-compute listing URL intent — shared by RSS, OG, others_signals rules
    listing_url = _is_listing_url(url)

    # ── Step 1: Error / blocked page ──────────────────────────────────────────
    if _detect_error_page(lower_html, title, http_status):
        logger.info(f"Error page detected: {url}")
        return {
            "label": "others", "confidence": 0.97,
            "method": "error_page", "reason": "page_error_or_blocked",
            "is_error_page": True,
        }

    # ── Step 2: Schema.org JSON-LD ────────────────────────────────────────────
    # Developer-declared ground truth — most reliable signal available.
    # When a site sets @type: "Product" or @type: "ItemList", trust it.
    schema_types = _extract_schema_types(soup)
    if schema_types:
        for stype in schema_types:
            cls = SCHEMA_TO_CLASS.get(stype)
            if cls:
                logger.info(f"Schema.org '{stype}' → {cls}: {url}")
                return {
                    "label": cls, "confidence": CONF_SCHEMA,
                    "method": "schema_org",
                    "reason": f"schema_type={stype}",
                    "schema_types": schema_types,
                }

    # ── Step 3: RSS / Atom feed ───────────────────────────────────────────────
    # Skipped for listing-intent URLs — search/browse pages often expose RSS
    # feeds but are still search pages, not "list" in the aggregator sense.
    

    # ── Step 4: Open Graph type ───────────────────────────────────────────────
    # Skipped for listing-intent URLs — search/browse pages inherit og:type
    # from site templates (e.g. Dribbble search has og:type=article → wrong).
    # NOTE: rel=next/prev previously was step 3 here (REMOVED in v8).
    #   Documentation sites (Python docs, MDN, Django, Wikipedia) use rel=next
    #   for chapter/article sequence navigation — not pagination of a list.
    #   Blog posts use rel=next for previous/next article. Product pages use it
    #   for prev/next product. rel=next is ambiguous and fires incorrectly on
    #   detail pages. It is now ONLY an ML feature (rel_has_next, rel_has_prev
    #   in pillar 12 of feature_extractor.py).
    if not listing_url:
        og_tag  = soup.find("meta", property="og:type")
        og_type = (og_tag.get("content", "") if og_tag else "").lower().strip()
        if og_type and og_type not in {"website", "webpage", ""}:
            for prefix, cls, conf in OG_TYPE_PREFIX_MAP:
                if og_type.startswith(prefix) or og_type == prefix.rstrip("."):
                    logger.info(f"OG type '{og_type}' → {cls}: {url}")
                    return {
                        "label": cls, "confidence": conf,
                        "method": "open_graph", "reason": f"og:type={og_type}",
                    }

    # ── Step 5: Twitter card = player ─────────────────────────────────────────
    # Very specific — only playable media items set this. Reliable regardless of URL.
    tc_tag  = soup.find("meta", attrs={"name": "twitter:card"})
    tc_type = (tc_tag.get("content", "") if tc_tag else "").lower().strip()
    if tc_type == "player":
        logger.info(f"Twitter card=player → detail: {url}")
        return {
            "label": "detail", "confidence": CONF_TWITTER,
            "method": "twitter_card", "reason": "twitter:card=player",
        }

    # ── Step 6: Strong listing pattern (3-of-5 structural signals) ────────────
    if _strong_listing_pattern(soup, body_text):
        logger.info(f"Strong listing pattern → list: {url}")
        return {
            "label": "list", "confidence": CONF_STRONG,
            "method": "strong_listing", "reason": "multi_signal_listing_dom",
        }

    # ── Step 7: Domain-specific rules ─────────────────────────────────────────
    domain_result = _domain_specific_rules(url, body_text, soup)
    if domain_result:
        cls, conf, reason = domain_result
        logger.info(f"Domain rule '{reason}' → {cls}: {url}")
        return {
            "label": cls, "confidence": conf,
            "method": "domain_specific", "reason": reason,
        }

    # ── Step 8: Strong 'others' signals ───────────────────────────────────────
    # Detects true marketing / landing / legal / info pages.
    #
    # THREE GUARDS prevent false positives:
    #
    # Guard A — Listing URL escape:
    #   Job boards, browse pages, search pages mention "about us", "careers",
    #   "privacy" in their footer. Skip this rule for listing-intent URLs.
    #
    # Guard B — Detail page escape:
    #   Product pages, articles, and review pages ALSO have FAQ sections,
    #   "learn more" CTAs, and full footer nav (privacy, about us, contact).
    #   If ANY strong detail signal is present → this is not an "others" page.
    #   Example: Xiaomi TV on Poorvika had legal=5, info=5, faq=4 from footer
    #   but was a product page. Without this guard → OTHERS at 96%. Wrong.
    #
    # Guard C — Marketing vocabulary requirement:
    #   True OTHERS pages (Stripe, Notion, HubSpot) mention "pricing", "plans",
    #   "enterprise", "solutions" in their main content. Product pages and
    #   articles do not. This is the final separator between "SaaS landing page"
    #   and "product detail page that happens to have a standard footer".

    if not listing_url:

        # Guard B: Check for detail-page signals in main content
        has_price = bool(re.search(
            r"(\$|€|£|¥|₹|usd|eur|gbp)\s*[\d,]+|[\d,]+\s*(\$|€|£|¥|₹)|"
            r"\bprice[:\s]|₹\s*\d",
            lower
        ))
        has_buy_cta = bool(re.search(
            r"\b(add\s+to\s+cart|buy\s+now|add\s+to\s+bag|add\s+to\s+basket|"
            r"purchase\s+now|order\s+now|buy\s+it\s+now|shop\s+now)\b",
            lower
        ))
        has_rating = bool(re.search(
            r"\b(\d+(\.\d+)?)\s*(out\s+of|\/)\s*5\b|"
            r"\b\d+\s*(reviews?|ratings?|stars?)\b|"
            r"\bcustomer\s+(reviews?|ratings?)\b",
            lower
        ))
        has_specs = bool(re.search(
            r"\b(specifications?|specs?|display|processor|battery|ram|storage|"
            r"resolution|camera|weight|dimensions?|material|ingredients?|warranty|"
            r"model\s+number)\b",
            lower
        ))
        has_author_date = bool(
            (soup.find("time") or soup.find(attrs={"datetime": True})) and
            re.search(r"\b(by\s+\w+|author[:\s]|written\s+by|published)\b", lower)
        )

        # If any detail signal fires → skip others rule, let ML decide
        detail_guard = has_price or has_buy_cta or has_rating or has_specs or has_author_date
        if detail_guard:
            logger.debug(
                f"Detail-page guard skipped others rule "
                f"(price={has_price}, buy={has_buy_cta}, rating={has_rating}, "
                f"specs={has_specs}, author_date={has_author_date}): {url}"
            )
        else:
            # Compute others signals
            legal_signals = sum([
                bool(re.search(r"\bprivacy\s+policy\b", lower)),
                bool(re.search(r"\bterms\s+of\s+service\b", lower)),
                bool(re.search(r"\bterms\s+and\s+conditions\b", lower)),
                bool(re.search(r"\blegal\b", lower)),
                bool(re.search(r"\bcookie\s+policy\b", lower)),
                bool(re.search(r"\bdisclaimer\b", lower)),
                bool(re.search(r"\btrademark\b", lower)),
                bool(re.search(r"\bcompliance\b", lower)),
            ])
            # Removed "careers?", "help", "support" — appear in almost every
            # site footer, causing false "others" on listing and detail pages.
            info_signals = sum([
                bool(re.search(r"\babout\s+us\b", lower)),
                bool(re.search(r"\bcontact\s+us\b", lower)),
                bool(re.search(r"\bget\s+in\s+touch\b", lower)),
                bool(re.search(r"\bteam\b", lower)),
                bool(re.search(r"\bpress\b", lower)),
            ])
            faq_signals = sum([
                bool(re.search(r"\bfaq\b", lower)),
                bool(re.search(r"\bfrequently\s+asked\s+questions\b", lower)),
                bool(re.search(r"\bhow\s+can\s+i\b", lower)),
                bool(re.search(r"\bwhere\s+can\s+i\b", lower)),
                bool(re.search(r"\bwhat\s+is\b", lower)),
            ])
            cta_signals = sum([
                bool(re.search(r"\bget\s+started\b", lower)),
                bool(re.search(r"\bsign\s+up\b", lower)),
                bool(re.search(r"\btry\s+(for\s+)?free\b", lower)),
                bool(re.search(r"\bstart\s+(your\s+)?(free\s+)?trial\b", lower)),
                bool(re.search(r"\bbook\s+a\s+demo\b", lower)),
                bool(re.search(r"\brequest\s+a\s+demo\b", lower)),
                bool(re.search(r"\bdownload\s+(the\s+)?(app|now)\b", lower)),
                bool(re.search(r"\blearn\s+more\b", lower)),
                bool(re.search(r"\bwatch\s+video\b", lower)),
                bool(re.search(r"\bexplore\s+now\b", lower)),
            ])
            # Guard C: Marketing vocabulary (true SaaS/landing pages only)
            marketing_signals = sum([
                bool(re.search(r"\bpricing\b", lower)),
                bool(re.search(r"\bplans?\b", lower)),
                bool(re.search(r"\benterprise\b", lower)),
                bool(re.search(r"\bsolutions?\b", lower)),
                bool(re.search(r"\bplatform\b", lower)),
                bool(re.search(r"\bfor\s+(teams?|business|companies|organizations?)\b", lower)),
                bool(re.search(r"\bfree\s+trial\b", lower)),
                bool(re.search(r"\bno\s+credit\s+card\b", lower)),
            ])

            total_info = legal_signals + info_signals + faq_signals

            # Rule fires only when:
            #   • Strong info/legal/faq signals (≥3)
            #   • AND marketing vocabulary present (page is about a service/product LINE)
            #   • AND multiple CTAs (converting visitors, not selling one specific item)
            if total_info >= 3 and cta_signals >= 2 and marketing_signals >= 1:
                logger.info(f"Strong 'others' signals → others: {url}")
                return {
                    "label":      "others",
                    "confidence": 0.90,
                    "method":     "others_signals",
                    "reason": (
                        f"marketing_page: info={total_info}, "
                        f"cta={cta_signals}, marketing={marketing_signals}"
                    ),
                }

    # ── No rule matched → ML model ────────────────────────────────────────────
    # This is the correct outcome for ambiguous pages.
    # The ML model with 180+ features handles the rest.
    logger.debug(f"No rule matched → ML model: {url}")
    return None