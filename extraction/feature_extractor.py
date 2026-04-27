"""
extraction/feature_extractor.py
────────────────────────────────
Production-grade feature extractor for webpage classification.

Design philosophy:
  - NO single feature dominates — we extract structural, content, semantic,
    URL-pattern, and ratio-based features so the model can triangulate.
  - Features are grouped into pillars. Each pillar captures a different
    "signal" for list / detail / others.
  - Dead features (constant across classes) are excluded.
  - Fallback to URL-pattern features when HTML is too thin.

Feature Pillars
───────────────
1.  STRUCTURAL   – DOM shape (tag counts, nesting depth, grid/table patterns)
2.  LINK         – Link density, internal vs external, anchor text diversity
3.  CONTENT      – Text length, heading hierarchy, paragraph count, content blocks
4.  MEDIA        – Image count/density, video presence, gallery patterns
5.  LISTING      – Repeated card/item patterns, pagination, filter UI signals
6.  DETAIL       – Single-entity signals (breadcrumb depth, price/date/author, TOC)
7.  INTERACTION  – Forms, buttons, CTAs, search bars
8.  METADATA     – OG type, schema.org type, canonical, lang
9.  SEMANTIC NLP – Keyword density for list/detail cue words (no heavy models)
10. URL PATTERN  – Path tokens, depth (capped at 3 ← was 5), numeric IDs, slugs
11. SINGLE ENTITY– Entity name repetition, structure-content alignment
12. LANDING PAGE – Homepage/marketing signals, converted hard rules, RSS/rel=next
13. MODERN LIST  – Browse/discover, academic listings, job boards, no-pag lists

v6 CHANGELOG (URL bias reduction + composite boost):
  - url_path_depth cap lowered 5→3. SHAP rank 1 at 0.667 meant the model
    was cheating on URL shape. Cap at 3 forces DOM/content learning.
  - url_has_query_params weight neutralised — converted to url_has_any_param
    (binary, lower weight) + individual param type features kept.
  - Added url_depth_normalized (0.0–1.0 over 3 levels) as soft replacement.
  - composite_listing_score_v2: now includes pillar-13 modern-list signals
    (high_link_no_pagination, browse_vocab_signal, academic_listing_signal,
    is_job_listing_page, article_sibling_containers, is_browse_path).
    These were previously extracted but NOT feeding composite → model ignored them.
  - composite_detail_score_v2: adds others_vocab_is_boilerplate correction.
  - Added url_bias_corrected_listing: URL says "list" but content also agrees.
  - Extraction version bumped to 6.0.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from pipeline.utils import get_logger, load_config

logger = get_logger("extractor")
cfg = load_config()
EX_CFG = cfg["extraction"]

# ─── Keyword Sets ─────────────────────────────────────────────────────────────
LIST_CUE_WORDS = {
    "results", "listings", "search", "browse", "catalog", "catalogue",
    "index", "directory", "archive", "collection", "products", "items",
    "articles", "posts", "jobs", "properties", "recipes", "hotels",
    "filter", "sort", "showing", "found", "per page", "load more",
    "next page", "previous", "pagination", "view all", "see all",
    "compare", "grid", "list view", "refine",
}

DETAIL_CUE_WORDS = {
    "description", "overview", "specification", "specifications", "details",
    "about", "summary", "biography", "profile", "review", "reviews",
    "rating", "price", "buy", "add to cart", "purchase", "order",
    "in stock", "availability", "published", "author", "posted",
    "last updated", "share", "comment", "reply", "related",
    "table of contents", "introduction", "conclusion", "references",
    "ingredients", "instructions", "how to", "step", "method",
    "salary", "experience", "requirements", "responsibilities",
}

LIST_SCHEMA_TYPES = {
    "searchresultspage", "itemlist", "collectionpage", "productcollection",
    "offerscatalog", "breadcrumblist",
}

DETAIL_SCHEMA_TYPES = {
    "product", "article", "newsarticle", "blogposting", "recipe",
    "jobposting", "event", "movie", "book", "person", "organization",
    "hotel", "restaurant", "localbusiness", "medicalcondition",
    "howto", "faqpage", "profilepage",
}


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    return round(num / den, 6) if den > 0 else default


def _safe_log(x: float) -> float:
    return round(math.log1p(max(x, 0)), 6)


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 – STRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════
def _structural_features(soup: BeautifulSoup, body_text: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}

    for tag in ["div", "section", "article", "ul", "ol", "li", "table",
                "tr", "td", "span", "p", "h1", "h2", "h3", "h4", "aside",
                "nav", "header", "footer", "main"]:
        f[f"tag_{tag}_count"] = len(soup.find_all(tag))

    li_count = f["tag_li_count"]
    total_tags = sum(f[k] for k in f if k.startswith("tag_")) or 1
    f["li_density"] = _safe_ratio(li_count, total_tags)
    f["article_section_ratio"] = _safe_ratio(
        f["tag_article_count"], f["tag_section_count"] + 1
    )

    def _max_depth(el: Tag, depth: int = 0) -> int:
        children = [c for c in el.children if isinstance(c, Tag)]
        if not children:
            return depth
        return max(_max_depth(c, depth + 1) for c in children[:20])

    try:
        body = soup.find("body")
        f["dom_max_depth"] = _max_depth(body, 0) if body else 0
    except Exception:
        f["dom_max_depth"] = 0

    f["has_data_table"] = int(
        bool(soup.find("table")) and len(soup.find_all("td")) > 6
    )

    grid_pattern = re.compile(r"\b(grid|card|tile|product[-_]?item|result[-_]?item|"
                               r"listing[-_]?item|job[-_]?item|post[-_]?item|"
                               r"item[-_]?card|product[-_]?card)\b", re.I)
    grid_matches = len([t for t in soup.find_all(class_=True)
                        if any(grid_pattern.search(c) for c in t.get("class", []))])
    f["grid_class_count"] = min(grid_matches, 200)
    f["has_grid_layout"] = int(grid_matches >= 3)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 – LINK FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _link_features(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    domain = urlparse(url).netloc

    all_links = soup.find_all("a", href=True)
    total = len(all_links)
    f["total_links"] = total
    f["log_total_links"] = _safe_log(total)

    internal, external, nav_links = 0, 0, 0
    anchor_texts: List[str] = []
    nav_el = soup.find("nav")
    nav_hrefs = set()
    if nav_el:
        nav_hrefs = {a.get("href", "") for a in nav_el.find_all("a", href=True)}

    for a in all_links:
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if text:
            anchor_texts.append(text.lower()[:80])
        parsed = urlparse(href)
        if parsed.netloc and parsed.netloc != domain:
            external += 1
        else:
            internal += 1
        if href in nav_hrefs:
            nav_links += 1

    f["internal_links"] = internal
    f["external_links"] = external
    f["nav_links"] = nav_links
    f["content_links"] = max(0, internal - nav_links)
    f["external_link_ratio"] = _safe_ratio(external, total)

    unique_anchors = len(set(anchor_texts))
    f["anchor_text_diversity"] = _safe_ratio(unique_anchors, max(total, 1))

    body_text = soup.get_text(separator=" ", strip=True)
    word_count = max(len(body_text.split()), 1)
    f["link_to_word_ratio"] = _safe_ratio(total, word_count)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 – CONTENT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _content_features(soup: BeautifulSoup) -> Dict[str, Any]:
    f: Dict[str, Any] = {}

    body_text = soup.get_text(separator=" ", strip=True)
    words = body_text.split()
    word_count = len(words)
    f["word_count"] = word_count
    f["log_word_count"] = _safe_log(word_count)
    f["char_count"] = len(body_text)

    for h in ["h1", "h2", "h3", "h4"]:
        f[f"{h}_count"] = len(soup.find_all(h))

    f["heading_hierarchy_score"] = (
        f["h1_count"] * 4 + f["h2_count"] * 3 +
        f["h3_count"] * 2 + f["h4_count"]
    )
    f["single_h1"] = int(f["h1_count"] == 1)

    paras = soup.find_all("p")
    para_texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40]
    f["rich_paragraph_count"] = len(para_texts)
    f["log_rich_paragraph_count"] = _safe_log(len(para_texts))

    avg_para_len = sum(len(t) for t in para_texts) / max(len(para_texts), 1)
    f["avg_paragraph_length"] = round(avg_para_len, 2)

    content_els = soup.find_all(["article", "main", "section"])
    deep_content = [e for e in content_els if len(e.get_text(strip=True)) > 200]
    f["deep_content_blocks"] = len(deep_content)
    f["has_main_content_element"] = int(bool(soup.find("main")))
    f["has_article_element"] = int(bool(soup.find("article")))

    all_divs = soup.find_all(["div", "article", "section", "main"])
    if all_divs:
        lens = [len(d.get_text(strip=True)) for d in all_divs]
        max_len = max(lens) if lens else 0
        f["text_concentration_ratio"] = _safe_ratio(max_len, max(len(body_text), 1))
    else:
        f["text_concentration_ratio"] = 0.0

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 – MEDIA FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _media_features(soup: BeautifulSoup) -> Dict[str, Any]:
    f: Dict[str, Any] = {}

    imgs = soup.find_all("img")
    f["image_count"] = len(imgs)
    f["log_image_count"] = _safe_log(len(imgs))

    alt_texts = [img.get("alt", "").strip() for img in imgs if img.get("alt", "").strip()]
    f["images_with_alt"] = len(alt_texts)
    f["alt_text_ratio"] = _safe_ratio(len(alt_texts), max(len(imgs), 1))

    gallery_pattern = re.compile(
        r"\b(gallery|carousel|slider|swiper|thumbnail|thumb|product[-_]?image)\b", re.I
    )
    gallery_count = len([i for i in imgs
                         if gallery_pattern.search(str(i.get("class", [])) + str(i.get("id", "")))])
    f["gallery_image_count"] = gallery_count
    f["has_image_gallery"] = int(gallery_count >= 3)
    f["has_video"] = int(bool(soup.find(["video", "iframe"])))
    f["figure_count"] = len(soup.find_all("figure"))
    f["figcaption_count"] = len(soup.find_all("figcaption"))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 – LISTING-SPECIFIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _listing_features(soup: BeautifulSoup, body_text: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    lower_text = body_text.lower()

    pagination_patterns = [
        r"page\s*\d+", r"\d+\s*of\s*\d+", r"next\s*page",
        r"previous\s*page", r"load\s*more", r"show\s*more",
        r"view\s*all", r"see\s*all", r"\bpage\b",
    ]
    f["has_pagination"] = int(
        any(re.search(p, lower_text) for p in pagination_patterns) or
        bool(soup.find(class_=re.compile(r"pag(e|ination)", re.I))) or
        bool(soup.find(attrs={"aria-label": re.compile(r"pagination", re.I)}))
    )

    # Enhanced pagination: count page number links
    page_links = soup.find_all("a", href=re.compile(r"[?&]page=\d+", re.I))
    f["page_number_links"] = len(page_links)
    f["has_page_param_links"] = int(len(page_links) >= 2)

    filter_pattern = re.compile(r"\b(filter|facet|refine|sort[-_]by|sort\s+by)\b", re.I)
    f["has_filter_ui"] = int(
        bool(soup.find(class_=filter_pattern)) or
        bool(soup.find(id=filter_pattern)) or
        bool(re.search(r"\bfilter\b|\bsort by\b|\brefine\b", lower_text))
    )

    result_count_pattern = re.compile(
        r"\b(\d[\d,]*)\s*(results?|listings?|items?|products?|jobs?|properties|found|matches)\b",
        re.I
    )
    f["has_result_count"] = int(bool(result_count_pattern.search(lower_text)))

    repeated_blocks = 0
    for container in soup.find_all(["ul", "ol", "div", "section"]):
        children = [c for c in container.children if isinstance(c, Tag)]
        if len(children) >= 4:
            tag_counts = {}
            for c in children:
                tag_counts[c.name] = tag_counts.get(c.name, 0) + 1
            dominant = max(tag_counts.values())
            if dominant >= 4:
                repeated_blocks += 1

    f["repeated_sibling_containers"] = min(repeated_blocks, 50)
    f["has_repeated_blocks"] = int(repeated_blocks >= 2)
    f["has_compare_checkboxes"] = int(
        len(soup.find_all("input", attrs={"type": "checkbox"})) >= 3
    )

    # Card/grid pattern detection
    card_elements = soup.find_all(class_=re.compile(r"\b(card|item|tile|product|listing|result)\b", re.I))
    f["card_count"] = min(len(card_elements), 100)
    f["has_card_elements"] = int(len(card_elements) >= 3)

    grid_classes = sum(1 for el in soup.find_all(class_=True)
                      if re.search(r"\b(grid|flex|col|row|container)\b", " ".join(el.get("class", [])), re.I))
    f["grid_class_count"] = min(grid_classes, 50)
    f["has_grid_layout"] = int(grid_classes >= 3)

    # Link diversity features
    all_links = soup.find_all("a", href=True)
    f["total_links"] = len(all_links)
    link_texts = [a.get_text(strip=True) for a in all_links if a.get_text(strip=True)]
    unique_link_texts = len(set(link_texts)) if link_texts else 0
    f["anchor_text_diversity"] = _safe_ratio(unique_link_texts, len(link_texts)) if link_texts else 0
    f["links_per_1k_words"] = _safe_ratio(len(all_links) * 1000, max(len(lower_text.split()), 1))

    breadcrumb = soup.find(
        [True],
        attrs={"class": re.compile(r"breadcrumb", re.I)}
    )
    if not breadcrumb:
        breadcrumb = soup.find(attrs={"aria-label": re.compile(r"breadcrumb", re.I)})
    if breadcrumb:
        crumbs = breadcrumb.find_all(["li", "span", "a"])
        f["breadcrumb_depth"] = len(crumbs)
    else:
        f["breadcrumb_depth"] = 0

    f["has_breadcrumb"] = int(f["breadcrumb_depth"] > 0)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 6 – DETAIL-SPECIFIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _detail_features(soup: BeautifulSoup, body_text: str, url: str = "") -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    lower_text = body_text.lower()

    price_pattern = re.compile(
        r"(\$|€|£|¥|₹|USD|EUR|GBP)\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*(\$|€|£|¥|₹)",
        re.I
    )
    price_matches = price_pattern.findall(lower_text)
    f["price_mention_count"] = min(len(price_matches), 20)
    f["has_price"] = int(len(price_matches) > 0)

    buy_pattern = re.compile(
        r"\b(add\s+to\s+cart|buy\s+now|add\s+to\s+bag|purchase|order\s+now|"
        r"book\s+now|apply\s+now|get\s+started|download\s+now)\b",
        re.I
    )
    f["has_buy_cta"] = int(bool(buy_pattern.search(lower_text)))

    author_pattern = re.compile(
        r"\b(by|author|written\s+by|posted\s+by|reviewed\s+by)\b", re.I
    )
    f["has_author"] = int(bool(author_pattern.search(lower_text)))

    date_pattern = re.compile(
        r"\b(published|posted|updated|last\s+modified|date)\b.*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})",
        re.I
    )
    f["has_publish_date"] = int(bool(
        date_pattern.search(lower_text) or
        soup.find("time") or
        soup.find(attrs={"datetime": True})
    ))

    rating_pattern = re.compile(
        r"\b(\d+(\.\d+)?)\s*(out of|\/)\s*\d+\b|\b(stars?|rating|reviews?)\b",
        re.I
    )
    f["has_rating"] = int(bool(rating_pattern.search(lower_text)))

    toc_pattern = re.compile(r"table\s+of\s+contents|toc|contents", re.I)
    f["has_toc"] = int(
        bool(soup.find(class_=toc_pattern)) or
        bool(soup.find(id=toc_pattern)) or
        bool(toc_pattern.search(lower_text[:500]))
    )

    anchor_links = [a for a in soup.find_all("a", href=True)
                    if a["href"].startswith("#")]
    f["internal_anchor_links"] = len(anchor_links)
    f["has_internal_anchors"] = int(len(anchor_links) >= 3)

    spec_pattern = re.compile(
        r"\b(specification|spec|features?|dimensions?|weight|material|"
        r"compatibility|requirements?|ingredient)\b",
        re.I
    )
    f["has_spec_section"] = int(bool(
        soup.find(class_=spec_pattern) or
        spec_pattern.search(lower_text[:3000])
    ))

    share_pattern = re.compile(r"\b(share|tweet|facebook|linkedin|whatsapp)\b", re.I)
    f["has_share_buttons"] = int(bool(share_pattern.search(lower_text)))

    comment_pattern = re.compile(r"\b(comment|reply|discussion|disqus|livefyre)\b", re.I)
    f["has_comments_section"] = int(bool(comment_pattern.search(lower_text)))

    related_pattern = re.compile(
        r"\b(related|similar|you\s+may\s+also|recommended|people\s+also)\b", re.I
    )
    f["has_related_section"] = int(bool(related_pattern.search(lower_text)))

    og_type_raw = (soup.find("meta", property="og:type") or {}).get("content", "").lower()
    f["og_type_is_video"] = int("video" in og_type_raw)

    tc_raw = (soup.find("meta", attrs={"name": "twitter:card"}) or {}).get("content", "").lower()
    f["has_twitter_player"] = int("player" in tc_raw)

    f["has_duration_meta"] = int(bool(
        soup.find("meta", property="video:duration") or
        soup.find("meta", property="music:duration") or
        soup.find("meta", attrs={"itemprop": "duration"}) or
        re.search(r"\b(duration|runtime)\s*[:\-]\s*\d+", lower_text[:2000])
    ))

    single_item_url_pattern = re.compile(
        r"(\?v=|&v=|/watch\b|/item/|/dp/|/product/|/p/[a-zA-Z0-9_-]+|"
        r"/post/|/article/|/news/|/story/|/job/|/event/|/recipe/|/listing/|/shorts/)",
        re.I
    )
    f["url_signals_single_item"] = int(bool(single_item_url_pattern.search(url)))

    engagement_pattern = re.compile(
        r"\b(\d[\d,.]*[KMBk]?)\s*(views?|likes?|dislikes?|subscribers?|followers?|watches?)\b",
        re.I
    )
    f["has_engagement_metrics"] = int(bool(engagement_pattern.search(lower_text)))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 7 – INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _interaction_features(soup: BeautifulSoup) -> Dict[str, Any]:
    f: Dict[str, Any] = {}

    forms = soup.find_all("form")
    f["form_count"] = len(forms)
    f["has_search_form"] = int(bool(
        soup.find("input", attrs={"type": "search"}) or
        soup.find("input", attrs={"placeholder": re.compile(r"search", re.I)})
    ))
    f["has_login_form"] = int(bool(soup.find("input", attrs={"type": "password"})))

    all_buttons = soup.find_all(["button", "input"])
    f["button_count"] = len([b for b in all_buttons if b.name == "button"])
    f["select_count"] = len(soup.find_all("select"))
    f["range_input_count"] = len(soup.find_all("input", attrs={"type": "range"}))

    tab_pattern = re.compile(r"\btab\b", re.I)
    f["has_tabs"] = int(bool(
        soup.find(role="tablist") or soup.find(class_=tab_pattern)
    ))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 8 – METADATA / SEO FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _metadata_features(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}

    og_type = (soup.find("meta", property="og:type") or {}).get("content", "").lower()
    f["og_type_is_article"] = int("article" in og_type)
    f["og_type_is_product"] = int("product" in og_type)
    f["og_type_is_website"] = int(og_type in {"website", "webpage", ""})

    tc = (soup.find("meta", attrs={"name": "twitter:card"}) or {}).get("content", "").lower()
    f["has_twitter_summary_large"] = int("summary_large" in tc)

    schema_types: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, list):
                for item in data:
                    t = item.get("@type", "")
                    if isinstance(t, list):
                        schema_types.extend([x.lower() for x in t])
                    else:
                        schema_types.append(str(t).lower())
            elif isinstance(data, dict):
                t = data.get("@type", "")
                if isinstance(t, list):
                    schema_types.extend([x.lower() for x in t])
                else:
                    schema_types.append(str(t).lower())
        except Exception:
            pass

    f["has_schema_list_type"] = int(bool(set(schema_types) & LIST_SCHEMA_TYPES))
    f["has_schema_detail_type"] = int(bool(set(schema_types) & DETAIL_SCHEMA_TYPES))
    f["schema_type_count"] = len(schema_types)
    f["has_microdata"] = int(len(soup.find_all(itemtype=True)) > 0)
    f["has_canonical"] = int(bool(soup.find("link", rel="canonical")))

    robots = (soup.find("meta", attrs={"name": "robots"}) or {}).get("content", "").lower()
    f["robots_noindex"] = int("noindex" in robots)

    html_tag = soup.find("html")
    lang = (html_tag.get("lang", "") if html_tag else "").lower()
    f["page_lang_is_english"] = int(lang.startswith("en"))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 9 – SEMANTIC / NLP FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _semantic_features(body_text: str, title: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    sample = (title + " " + body_text[: EX_CFG["max_text_sample"]]).lower()
    words = sample.split()
    total_words = max(len(words), 1)

    list_hits = sum(1 for w in LIST_CUE_WORDS if w in sample)
    detail_hits = sum(1 for w in DETAIL_CUE_WORDS if w in sample)

    f["list_cue_word_hits"] = list_hits
    f["detail_cue_word_hits"] = detail_hits
    f["list_vs_detail_cue_ratio"] = _safe_ratio(list_hits, detail_hits + 1)
    f["detail_vs_list_cue_ratio"] = _safe_ratio(detail_hits, list_hits + 1)

    numeric_tokens = sum(1 for w in words if re.match(r"^\$?[\d,]+\.?\d*$", w))
    f["numeric_token_density"] = _safe_ratio(numeric_tokens, total_words)

    f["title_word_count"] = len(title.split())
    f["title_has_list_cue"] = int(any(w in title.lower() for w in LIST_CUE_WORDS))
    f["title_has_detail_cue"] = int(any(w in title.lower() for w in DETAIL_CUE_WORDS))
    f["title_has_question"] = int("?" in title)
    f["title_has_count_in_parens"] = int(bool(re.search(r"\(\d+\)", title)))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 10 – URL PATTERN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _url_features(url: str) -> Dict[str, Any]:
    """
    v6: url_path_depth cap lowered 5→3.
    SHAP showed depth at rank 1 (0.667) — model was learning URL shape
    instead of page content. Cap at 3 forces content/DOM features to matter.

    url_has_query_params kept as binary but renamed to url_has_any_param
    to reduce its conceptual weight. The individual param-type features
    (url_has_search_param, url_has_filter_param) carry the real signal.

    url_depth_normalized added: 0.0–1.0 over 3 levels, smooth hint only.
    """
    f: Dict[str, Any] = {}
    parsed = urlparse(url)
    path   = parsed.path.lower()
    query  = parsed.query.lower()
    path_parts = [p for p in path.split("/") if p]

    # ── CHANGED: cap at 3 (was 5) to reduce SHAP dominance ──
    raw_depth = len(path_parts)
    f["url_path_depth"]           = min(raw_depth, 3)
    f["url_depth_normalized"]     = round(min(raw_depth, 3) / 3.0, 4)  # 0.0–1.0

    # Keep binary query param signal but reduce its weight by keeping it simple
    f["url_has_any_param"]        = int(bool(query))   # renamed from url_has_query_params
    f["url_has_query_params"]     = int(bool(query))   # kept for backward compat in composite
    f["url_query_param_count"]    = len(query.split("&")) if query else 0
    f["url_has_numeric_id"]       = int(bool(re.search(r"/\d{3,}", path)))

    slug_parts = [p for p in path_parts if re.match(r"[a-z]+-[a-z]+-[a-z]", p)]
    f["url_has_long_slug"] = int(len(slug_parts) >= 1)

    list_url_tokens = {
        "search", "results", "catalog", "category", "categories",
        "products", "listings", "browse", "index", "archive",
        "tag", "tags", "shop", "store", "jobs", "properties",
        "hotels", "recipes", "articles", "news", "blog",
    }
    detail_url_tokens = {
        "product", "item", "article", "post", "property",
        "job", "hotel", "recipe", "profile", "detail",
        "view", "show", "read", "watch", "event",
    }
    path_token_set = set(path_parts)
    f["url_list_token_match"]   = int(bool(path_token_set & list_url_tokens))
    f["url_detail_token_match"] = int(bool(path_token_set & detail_url_tokens))

    last_part = path_parts[-1] if path_parts else ""
    f["url_has_extension"] = int("." in last_part)
    f["url_is_root"]       = int(path in {"", "/"})

    search_params = {"q", "query", "search", "keyword", "s", "find"}
    filter_params = {"filter", "sort", "order", "page", "offset", "limit", "from"}
    params = set(k.split("=")[0] for k in query.split("&")) if query else set()
    f["url_has_search_param"] = int(bool(params & search_params))
    f["url_has_filter_param"] = int(bool(params & filter_params))
    f["url_has_page_param"]   = int(
        "page" in params or bool(re.search(r"[?&]page=\d+", url.lower()))
    )

    # Domain-specific features
    domain = parsed.netloc.lower()
    job_board_domains = {"lever.co", "greenhouse.io", "workable.com", "smartrecruiters.com",
                        "applytojob.com", "bamboohr.com", "jobs.lever.co", "boards.greenhouse.io"}
    api_doc_domains = {"docs.python.org", "developer.mozilla.org", "docs.aws.amazon.com",
                      "github.com/docs", "official-docs"}
    f["is_job_board_domain"] = int(bool(any(d in domain for d in job_board_domains)))
    f["is_api_doc_domain"]   = int(bool(any(d in domain for d in api_doc_domains)))
    f["has_jobs_in_path"]    = int("jobs" in path or "careers" in path)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
def _composite_features(all_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    v6: composite_listing_score now includes pillar-13 modern-list signals.
    Previously those features existed but never fed into the composite, so
    the model was ignoring them. Now they contribute directly.
    """
    f: Dict[str, Any] = {}
    g = all_features

    # ── Listing score (v2: includes modern-list signals from pillar 13) ────────
    f["composite_listing_score"] = round(
        g.get("has_pagination", 0) * 3
        + g.get("has_filter_ui", 0) * 2
        + g.get("has_result_count", 0) * 2
        + g.get("has_repeated_blocks", 0) * 1.5
        + g.get("grid_class_count", 0) * 0.04
        + g.get("url_list_token_match", 0)
        + g.get("has_schema_list_type", 0) * 3
        + g.get("list_cue_word_hits", 0) * 0.3
        + g.get("url_has_filter_param", 0)
        + g.get("url_has_page_param", 0)
        + g.get("title_has_list_cue", 0)
        + g.get("anchor_text_diversity", 0) * 1.5
        # ── NEW: modern-list signals (pillar 13 → composite) ──────────────────
        + g.get("high_link_no_pagination", 0) * 2      # modern no-pag list
        + g.get("browse_vocab_signal", 0) * 1.5        # browse/discover pages
        + g.get("academic_listing_signal", 0) * 2      # arxiv, pubmed, gov
        + g.get("is_job_listing_page", 0) * 2          # job board listing
        + g.get("article_sibling_containers", 0) * 0.3  # semantic article grids
        + g.get("is_browse_path", 0) * 1.5             # /categories /discover
        + g.get("has_many_years", 0) * 0.5,            # year-heavy listing
        4
    )

    # ── Detail score (unchanged except boilerplate correction) ────────────────
    f["composite_detail_score"] = round(
        g.get("has_buy_cta", 0) * 3
        + g.get("has_price", 0) * 2
        + g.get("has_author", 0) * 2
        + g.get("has_publish_date", 0) * 2
        + g.get("has_toc", 0) * 2
        + g.get("single_h1", 0) * 2
        + g.get("has_spec_section", 0) * 2
        + g.get("has_rating", 0)
        + g.get("url_detail_token_match", 0)
        + g.get("has_schema_detail_type", 0) * 3
        + g.get("detail_cue_word_hits", 0) * 0.3
        + g.get("url_has_numeric_id", 0)
        + g.get("url_has_long_slug", 0)
        + g.get("has_share_buttons", 0)
        + g.get("has_comments_section", 0)
        + g.get("og_type_is_video", 0) * 4
        + g.get("has_twitter_player", 0) * 3
        + g.get("has_duration_meta", 0) * 2
        + g.get("url_signals_single_item", 0) * 2
        + g.get("has_engagement_metrics", 0)
        + (2 if g.get("avg_paragraph_length", 0) > 200 else
           1 if g.get("avg_paragraph_length", 0) > 100 else 0),
        4
    )

    ls = f["composite_listing_score"]
    ds = f["composite_detail_score"]
    f["composite_others_score"] = round(
        max(0, 5 - abs(ls - ds)) * 0.5
        + g.get("og_type_is_website", 0)
        + g.get("has_login_form", 0)
        + g.get("url_is_root", 0) * 2,
        4
    )

    # ── NEW: URL-bias-corrected listing signal ─────────────────────────────────
    # URL says list AND content also agrees → high confidence listing.
    # Reduces the risk of URL alone driving a list prediction.
    url_says_list = int(
        g.get("url_list_token_match", 0) == 1 or
        g.get("url_has_search_param", 0) == 1 or
        g.get("url_has_page_param", 0) == 1
    )
    content_says_list = int(
        g.get("has_repeated_blocks", 0) == 1 or
        g.get("has_result_count", 0) == 1 or
        g.get("anchor_text_diversity", 0) > 0.6 or
        g.get("high_link_no_pagination", 0) == 1
    )
    f["url_bias_corrected_listing"] = int(url_says_list and content_says_list)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 11 — SINGLE-ENTITY DETECTION + CONTENT-STRUCTURE ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
def _single_entity_features(
    soup: BeautifulSoup,
    body_text: str,
    url: str,
    all_feats: Dict[str, Any],
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    lower = body_text.lower()
    words = lower.split()
    total_words = max(len(words), 1)
    parsed = urlparse(url)
    path   = parsed.path.lower()

    h1_count   = all_feats.get("h1_count", 0)
    word_count = all_feats.get("word_count", 0)
    rep_blocks = all_feats.get("repeated_sibling_containers", 0)
    avg_para   = all_feats.get("avg_paragraph_length", 0)
    rich_paras = all_feats.get("rich_paragraph_count", 0)

    f["single_entity_strong"] = int(
        h1_count == 1 and word_count > 150 and avg_para > 80 and
        rich_paras >= 2 and rep_blocks < 8
    )
    f["single_entity_weak"] = int(
        h1_count == 1 and word_count > 80 and rep_blocks < 12
    )

    h1_tags = soup.find_all("h1")
    entity_repetition = 0
    if h1_tags:
        h1_text = h1_tags[0].get_text(strip=True).lower()
        h1_words = h1_text.split()
        if len(h1_words) >= 1 and len(h1_text) > 3:
            if len(h1_words) >= 2:
                entity_repetition = lower.count(h1_text)
            else:
                entity_repetition = sum(1 for w in words if w == h1_text)
    f["entity_name_repetition"] = min(entity_repetition, 20)
    f["entity_is_repeated"] = int(entity_repetition >= 3)

    struct_list = (
        all_feats.get("has_repeated_blocks", 0)
        + all_feats.get("has_pagination", 0)
        + all_feats.get("has_filter_ui", 0)
        + int(all_feats.get("grid_class_count", 0) >= 4)
    )
    content_list = (
        all_feats.get("has_result_count", 0)
        + int(all_feats.get("anchor_text_diversity", 0) > 0.75)
        + int(all_feats.get("link_to_word_ratio", 0) > 0.20)
        + all_feats.get("title_has_list_cue", 0)
        + int(all_feats.get("list_cue_word_hits", 0) >= 3)
    )
    f["listing_structure_without_content"] = int(struct_list >= 2 and content_list == 0)
    f["struct_list_score"]  = round(struct_list, 4)
    f["content_list_score"] = round(content_list, 4)

    struct_detail = (
        int(all_feats.get("dom_max_depth", 0) > 8)
        + all_feats.get("has_article_element", 0)
        + all_feats.get("has_data_table", 0)
        + int(all_feats.get("deep_content_blocks", 0) >= 2)
    )
    content_detail = (
        int(all_feats.get("avg_paragraph_length", 0) > 150)
        + int(all_feats.get("rich_paragraph_count", 0) >= 4)
        + all_feats.get("has_author", 0)
        + all_feats.get("has_publish_date", 0)
        + all_feats.get("has_buy_cta", 0)
        + int(all_feats.get("detail_cue_word_hits", 0) >= 3)
    )
    f["detail_alignment_score"]  = round(min(struct_detail, content_detail), 4)
    f["listing_alignment_score"] = round(min(struct_list, content_list), 4)
    f["alignment_conflict"] = int(
        (struct_list >= 2 and content_detail >= 2) or
        (struct_detail >= 2 and content_list >= 2)
    )
    f["numeric_id_with_single_h1"] = int(
        all_feats.get("url_has_numeric_id", 0) == 1 and h1_count == 1
    )
    f["directory_detail_signals"] = sum([
        int(bool(re.search(r"/\d{7,}/", path))),
        int(bool(re.search(r"/[a-z0-9-]{3,}/[a-z0-9-]{3,}$", path))),
        int(all_feats.get("url_has_numeric_id", 0) == 1 and h1_count == 1),
        int(bool(re.search(r"\b(address|phone|email|website|opening\s+hours|contact)\b", lower))),
        int(bool(re.search(r"\b(about\s+us|our\s+story|founded|established|est\.)\b", lower))),
    ])

    li_texts = [li.get_text(strip=True).lower()[:40] for li in soup.find_all("li")
                if li.get_text(strip=True)]
    if li_texts and len(li_texts) > 4:
        unique_li = len(set(li_texts))
        f["li_text_diversity"] = round(unique_li / len(li_texts), 4)
        f["low_variation_li"]  = int(unique_li / len(li_texts) < 0.4)
    else:
        f["li_text_diversity"] = 0.0
        f["low_variation_li"]  = 0

    schema_types_raw: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            def _get_types(obj: Any) -> None:
                if isinstance(obj, dict):
                    t = obj.get("@type", "")
                    if isinstance(t, list):
                        schema_types_raw.extend([x.lower() for x in t])
                    elif t:
                        schema_types_raw.append(str(t).lower())
                    for v in obj.values():
                        if isinstance(v, (dict, list)):
                            _get_types(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _get_types(item)
            _get_types(data)
        except Exception:
            pass

    pre_hint = 0
    for t in schema_types_raw:
        if t in {"searchresultspage","itemlist","collectionpage","productcollection",
                 "offerscatalog","breadcrumblist","dataset","datafeed"}:
            pre_hint = 1; break
        if t in {"product","productgroup","article","newsarticle","blogposting",
                 "recipe","jobposting","event","movie","book","person","organization",
                 "localbusiness","hotel","restaurant","howto","faqpage","profilepage",
                 "videoobject","podcastepisode","course","vehicle","softwareapplication"}:
            pre_hint = 2; break
    f["pre_hint_schema"]     = pre_hint
    f["pre_hint_has_schema"] = int(pre_hint > 0)

    og = (soup.find("meta", property="og:type") or {}).get("content", "").lower()
    f["og_numeric"] = (2 if any(og.startswith(p) for p in
                                ["article","video.","music.","product","profile","book"])
                       else (1 if og in {"website","webpage",""} else 0))

    f["event_listing_signals"] = sum([
        int(bool(re.search(r"\b(events?\s+in|upcoming\s+events?|all\s+events?|browse\s+events?)\b", lower))),
        int(bool(re.search(r"\b(explore|find|discover)\b.*\bevents?\b", lower))),
        int(bool(re.search(r"\b\d+\s+events?\b", lower))),
    ])
    f["single_event_signals"] = sum([
        int(bool(re.search(r"\b(buy\s+tickets?|book\s+tickets?|register|rsvp)\b", lower))),
        int(bool(re.search(r"\b(venue|location|doors?\s+open|start\s+time)\b", lower))),
        int(bool(re.search(r"\b(lineup|performer|artist|speaker)\b", lower))),
    ])
    f["job_posting_signals"] = sum([
        int(bool(re.search(r"\b(apply\s+now|submit\s+application|apply\s+for\s+this)\b", lower))),
        int(bool(re.search(r"\b(salary|compensation|remuneration)\b", lower))),
        int(bool(re.search(r"\b(qualifications?|responsibilities|requirements)\b", lower))),
        int(bool(re.search(r"\b(deadline|closing\s+date|application\s+deadline)\b", lower))),
        int(bool(re.search(r"/(job|jobs|vacancy|vacancies|career|careers)/", path))),
    ])
    f["article_in_product_site"] = int(
        all_feats.get("has_author", 0) == 1 and
        all_feats.get("rich_paragraph_count", 0) >= 3 and
        all_feats.get("avg_paragraph_length", 0) > 100 and
        all_feats.get("form_count", 0) >= 1
    )
    f["forum_thread_signals"] = sum([
        int(bool(re.search(r"\b(reply|replies|post\s+reply|original\s+post)\b", lower))),
        int(bool(re.search(r"/(thread|topic|discussion)/[a-z0-9-]", path))),
        int(bool(re.search(r"\b(joined|member\s+since|posts?[:\s]\d+)\b", lower))),
    ])

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 12 — LANDING PAGE + SEO INTENT + CONVERTED RULE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
def _pillar12_features(
    soup: BeautifulSoup,
    body_text: str,
    url: str,
    all_feats: Dict[str, Any],
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    lower = body_text.lower()
    parsed = urlparse(url)
    path = parsed.path.strip("/").lower()

    # ── 1. Landing page / homepage score ──────────────────────────────────────
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
    f["cta_signal_count"] = cta_signals

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
    f["legal_signal_count"] = legal_signals

    info_signals = sum([
        bool(re.search(r"\babout\s+us\b", lower)),
        bool(re.search(r"\bcontact\s+us\b", lower)),
        bool(re.search(r"\bget\s+in\s+touch\b", lower)),
        bool(re.search(r"\bhelp\b", lower)),
        bool(re.search(r"\bsupport\b", lower)),
        bool(re.search(r"\bteam\b", lower)),
        bool(re.search(r"\bpress\b", lower)),
        bool(re.search(r"\bcareers?\b", lower)),
    ])
    f["info_signal_count"] = info_signals

    faq_signals = sum([
        bool(re.search(r"\bfaq\b", lower)),
        bool(re.search(r"\bfrequently\s+asked\s+questions\b", lower)),
        bool(re.search(r"\bhow\s+can\s+i\b", lower)),
        bool(re.search(r"\bwhere\s+can\s+i\b", lower)),
        bool(re.search(r"\bwhat\s+is\b", lower)),
    ])
    f["faq_signal_count"] = faq_signals

    hero_pat = re.compile(r"\b(hero|banner|jumbotron|masthead|splash)\b", re.I)
    has_hero = int(bool(soup.find(class_=hero_pat) or soup.find(id=hero_pat)))
    f["has_hero_section"] = has_hero

    has_pricing = int(bool(re.search(
        r"\b(pricing|plans?|per\s+month|per\s+year|billed\s+annually)\b", lower
    )))
    f["has_pricing_section"] = has_pricing

    has_testimonials = int(bool(re.search(
        r"\b(testimonial|review|trusted\s+by|customers?|client|case\s+study)\b", lower
    )))
    f["has_testimonials"] = has_testimonials

    is_root_path = (
        path in {"", "home", "index", "index.html", "index.htm", "en", "us", "en-us"}
        or bool(re.match(r'^[a-z]{2}(-[a-z]{2})?/?$', path))
    )
    f["is_root_url"] = int(is_root_path)

    f["landing_page_score"] = round(
        cta_signals * 0.5
        + has_hero * 1.5
        + has_pricing * 1.0
        + has_testimonials * 0.5
        + int(is_root_path) * 2.0,
        4
    )

    # ── 2. Navigation density ratio ───────────────────────────────────────────
    total_links = max(all_feats.get("total_links", 1), 1)
    nav_links   = all_feats.get("nav_links", 0)
    f["navigation_density_ratio"] = _safe_ratio(nav_links, total_links)

    # ── 3. Content section count ──────────────────────────────────────────────
    main_el = soup.find("main") or soup.find("body")
    if main_el:
        top_sections = [
            c for c in main_el.children
            if isinstance(c, Tag) and c.name in {"section", "div", "article"}
            and len(c.get_text(strip=True)) > 50
        ]
        f["content_section_count"] = min(len(top_sections), 20)
    else:
        f["content_section_count"] = 0

    # ── 4. Developer-declared pagination / feed as ML features ────────────────
    has_rel_next = 0
    has_rel_prev = 0
    has_rss      = 0
    for link in soup.find_all("link", rel=True):
        rel_val = link.get("rel", [])
        if isinstance(rel_val, str):
            rel_val = [rel_val]
        rel_lower  = [r.lower() for r in rel_val]
        link_type  = link.get("type", "").lower()
        if "next" in rel_lower:
            has_rel_next = 1
        if "prev" in rel_lower or "previous" in rel_lower:
            has_rel_prev = 1
        if "alternate" in rel_lower and ("rss" in link_type or "atom" in link_type):
            has_rss = 1
    f["rel_has_next"]  = has_rel_next
    f["rel_has_prev"]  = has_rel_prev
    f["has_rss_feed"]  = has_rss

    # ── 5. Converted hard rules → ML features ─────────────────────────────────
    has_price_signal = bool(re.search(
        r"(\$|€|£|¥|₹)\s*[\d,]+|[\d,]+\s*(\$|€|£|¥|₹)", lower
    ))
    has_atc = bool(re.search(
        r"\b(add\s+to\s+cart|buy\s+now|add\s+to\s+bag|order\s+now)\b", lower
    ))
    f["purchase_funnel_signal"] = int(has_price_signal and has_atc)

    has_author_signal = bool(re.search(
        r"\b(by\s+[A-Z][a-z]+|author[:\s]|written\s+by)\b", body_text
    ))
    has_date_signal = bool(
        soup.find("time") or
        soup.find(attrs={"datetime": True}) or
        re.search(r"\b(published|posted|updated)\b.*\d{4}", lower)
    )
    f["article_byline_signal"] = int(has_author_signal and has_date_signal)

    has_job_intent = bool(re.search(
        r"\b(apply\s+now|submit\s+application|apply\s+for\s+this\s+job)\b", lower
    ))
    has_job_vocab = bool(re.search(
        r"\b(salary|compensation|qualifications?|responsibilities|requirements)\b", lower
    ))
    f["job_posting_signal_combined"] = int(has_job_intent and has_job_vocab)

    # ── 6. Title intent signals ───────────────────────────────────────────────
    title_tag = soup.find("title")
    title_lower = title_tag.string.lower() if title_tag and title_tag.string else ""

    listing_title_words = {
        "results", "search", "browse", "catalog", "category", "directory",
        "listings", "products", "jobs", "hotels", "properties", "recipes",
        "articles", "all", "find", "explore", "shop", "store",
    }
    detail_title_words = {
        "review", "guide", "how to", "what is", "about", "profile",
        "price", "specification", "overview", "tutorial", "introduction",
    }
    f["title_signals_listing"] = int(any(w in title_lower for w in listing_title_words))
    f["title_signals_detail"]  = int(any(w in title_lower for w in detail_title_words))

    # ── 7. Others-specific path and content signals ───────────────────────────
    f["others_path_signal"] = int(bool(re.search(
        r"/(about|contact|faq|help|support|privacy|terms|legal|"
        r"careers|team|press|sitemap|login|signup|register|"
        r"account|settings|dashboard)/",
        "/" + path + "/"
    )))
    f["others_content_signal"] = int(bool(re.search(
        r"\b(our\s+mission|our\s+vision|our\s+team|about\s+us|contact\s+us|"
        r"privacy\s+policy|terms\s+of\s+service|cookie\s+policy|"
        r"frequently\s+asked|get\s+in\s+touch)\b",
        lower
    )))

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 13 — MODERN LIST PATTERNS + BROWSE/DISCOVER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def _pillar13_features(
    soup: BeautifulSoup,
    body_text: str,
    url: str,
    all_feats: Dict[str, Any],
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    lower = body_text.lower()
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    domain = parsed.netloc.lower()

    # ── 1. Browse / discover / curated listing page ───────────────────────────
    browse_path = int(bool(re.search(
        r"/(categories|category|discover|browse|explore|"
        r"exhibitions?|collections?|shows?|events?|archive|"
        r"publications?|highlights?|featured)/?$",
        path
    )))
    f["is_browse_path"] = browse_path

    f["browse_vocab_signal"] = int(bool(re.search(
        r"\b(discover|browse|explore|categories|all\s+(shows?|events?|"
        r"albums?|artists?|collections?|exhibitions?|publications?))\b",
        lower
    )))

    # ── 2. Academic / government publication listing ───────────────────────────
    year_mentions = len(re.findall(r"\b(19|20)\d{2}\b", lower))
    f["year_mention_count"] = min(year_mentions, 50)
    f["has_many_years"] = int(year_mentions >= 5)

    f["publication_list_signal"] = int(bool(re.search(
        r"\b(publications?|proceedings?|papers?|reports?|preprints?|"
        r"manuscripts?|abstracts?|bibliography|references)\b",
        lower
    )))

    f["is_academic_domain"] = int(bool(re.search(
        r"\.(edu|ac\.uk|gov|org)$", domain
    )))

    f["academic_listing_signal"] = int(
        (f["is_academic_domain"] or "arxiv" in domain or "pubmed" in domain) and
        f["publication_list_signal"] == 1 and
        year_mentions >= 3
    )

    # ── 3. Search results without classic ?q= param ───────────────────────────
    f["has_search_in_path"] = int(bool(re.search(r"/(search|results?|find)/", path)))
    f["has_type_query_param"] = int("type=" in query or "category=" in query)

    # ── 4. Job listing page (not job detail) ──────────────────────────────────
    job_listing_vocab = sum([
        bool(re.search(r"\b(full[- ]time|part[- ]time|remote|hybrid|on[- ]site)\b", lower)),
        bool(re.search(r"\b(\d+[\d,]*\s*(jobs?|positions?|openings?|roles?|vacancies))\b", lower)),
        bool(re.search(r"\b(filter\s+by|sort\s+by|job\s+type|experience\s+level)\b", lower)),
        bool(re.search(r"\b(new\s+jobs?|latest\s+jobs?|featured\s+jobs?)\b", lower)),
    ])
    f["job_listing_vocab"] = job_listing_vocab
    f["is_job_listing_page"] = int(job_listing_vocab >= 2)

    # ── 5. Link-to-item ratio for no-pagination lists ─────────────────────────
    content_links = all_feats.get("content_links", 0)
    has_pagination = all_feats.get("has_pagination", 0)
    anchor_diversity = all_feats.get("anchor_text_diversity", 0.0)
    f["high_link_no_pagination"] = int(
        content_links >= 20 and
        has_pagination == 0 and
        anchor_diversity > 0.55
    )

    # ── 6. Grid of homogeneous cards without product CSS classes ──────────────
    article_siblings = 0
    for container in soup.find_all(["ul", "div", "section"])[:100]:
        articles = container.find_all("article", recursive=False)
        if len(articles) >= 4:
            article_siblings += 1
    f["article_sibling_containers"] = min(article_siblings, 20)
    f["has_article_grid"] = int(article_siblings >= 1)

    # ── 7. Others-signal noise correction ────────────────────────────────────
    footer_el = soup.find("footer")
    nav_el = soup.find("nav")
    footer_nav_text = ""
    if footer_el:
        footer_nav_text += footer_el.get_text(separator=" ", strip=True).lower()
    if nav_el:
        footer_nav_text += " " + nav_el.get_text(separator=" ", strip=True).lower()

    others_vocab_in_footer = sum([
        bool(re.search(r"\babout\s+us\b", footer_nav_text)),
        bool(re.search(r"\bcontact\b", footer_nav_text)),
        bool(re.search(r"\bprivacy\b", footer_nav_text)),
        bool(re.search(r"\bterms\b", footer_nav_text)),
        bool(re.search(r"\bhelp\b", footer_nav_text)),
        bool(re.search(r"\bsupport\b", footer_nav_text)),
        bool(re.search(r"\bcareers?\b", footer_nav_text)),
    ])
    others_vocab_in_body = sum([
        bool(re.search(r"\babout\s+us\b", lower)),
        bool(re.search(r"\bcontact\b", lower)),
        bool(re.search(r"\bprivacy\b", lower)),
        bool(re.search(r"\bterms\b", lower)),
        bool(re.search(r"\bhelp\b", lower)),
        bool(re.search(r"\bsupport\b", lower)),
        bool(re.search(r"\bcareers?\b", lower)),
    ])
    f["others_vocab_footer_ratio"] = _safe_ratio(others_vocab_in_footer, max(others_vocab_in_body, 1))
    f["others_vocab_is_boilerplate"] = int(
        others_vocab_in_footer >= 3 and
        others_vocab_in_footer >= others_vocab_in_body * 0.7
    )

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(
    html: str,
    url: str,
    page_json: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Extract all features from raw HTML + URL.
    Returns a flat dict of feature_name → value (int/float).
    """
    features: Dict[str, Any] = {}

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body_text = soup.get_text(separator=" ", strip=True)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if page_json:
        title = title or page_json.get("title", "")

    features["content_available"] = int(len(body_text) >= EX_CFG["min_text_length"])

    features.update(_structural_features(soup, body_text))
    features.update(_link_features(soup, url))
    features.update(_content_features(soup))
    features.update(_media_features(soup))
    features.update(_listing_features(soup, body_text))
    features.update(_detail_features(soup, body_text, url))
    features.update(_interaction_features(soup))
    features.update(_metadata_features(soup, url))
    features.update(_semantic_features(body_text, title))
    features.update(_url_features(url))
    features.update(_composite_features(features))
    features.update(_single_entity_features(soup, body_text, url, features))
    features.update(_pillar12_features(soup, body_text, url, features))
    features.update(_pillar13_features(soup, body_text, url, features))

    features["_url"]                = url
    features["_title"]              = title
    features["_body_text_length"]   = len(body_text)
    features["_extraction_version"] = "6.0"

    return features


def get_feature_columns(features: Dict[str, Any]) -> List[str]:
    """Return only the numeric feature columns (exclude _metadata keys)."""
    return [k for k in features.keys() if not k.startswith("_")]