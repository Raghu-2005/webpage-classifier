"""
pipeline/utils.py
─────────────────
Shared utilities used across the entire pipeline:
  - Config loading
  - Consistent logger setup (file + console via Rich)
  - URL → safe folder-name hashing
  - Checkpoint read/write
  - Mapping read/write
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.logging import RichHandler

# ─── Singleton console ────────────────────────────────────────────────────────
console = Console()

# ─── Config ───────────────────────────────────────────────────────────────────
_config: Optional[Dict[str, Any]] = None
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load (and cache) YAML config."""
    global _config
    if _config is None:
        cfg_path = path or _CONFIG_PATH
        with open(cfg_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


# ─── Logger ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both console (Rich) and a log file."""
    cfg = load_config()
    log_level = getattr(logging, cfg["logging"]["level"], logging.INFO)
    log_file = Path(cfg["logging"]["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(log_level)

    # Rich console handler
    rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    rich_handler.setLevel(log_level)
    logger.addHandler(rich_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


# ─── URL Hashing ──────────────────────────────────────────────────────────────
def url_to_folder_name(url: str) -> str:
    """
    Convert a URL to a deterministic, filesystem-safe folder name.
    Format: <domain_slug>__<sha256_8chars>
    Example: en_wikipedia_org__a3f2b1c9
    """
    url = url.strip().rstrip("/")
    # extract domain part for readability
    domain = re.sub(r"https?://", "", url).split("/")[0]
    domain_slug = re.sub(r"[^a-zA-Z0-9]", "_", domain)[:40]
    sha = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{domain_slug}__{sha}"


# ─── Checkpoint ───────────────────────────────────────────────────────────────
def load_checkpoint(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Returns dict keyed by url with value = status dict.
    {
      "https://example.com": {
        "scrape_status": "success" | "failed" | "skipped",
        "extract_status": "success" | "failed" | "pending",
        "folder": "example_com__abc12345",
        "label": "list",
        "timestamp": "2024-..."
      }
    }
    """
    cfg = load_config()
    cp_path = path or Path(cfg["paths"]["checkpoint"])
    if cp_path.exists():
        with open(cp_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    cfg = load_config()
    cp_path = path or Path(cfg["paths"]["checkpoint"])
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump(data, f, indent=2)


def update_checkpoint(
    url: str,
    updates: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Atomically update a single URL entry in the checkpoint."""
    data = load_checkpoint(path)
    if url not in data:
        data[url] = {}
    data[url].update(updates)
    data[url]["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_checkpoint(data, path)


# ─── Mapping ──────────────────────────────────────────────────────────────────
def load_mapping(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Global mapping: folder_name → metadata.
    {
      "example_com__abc12345": {
        "url": "https://example.com",
        "label": "list",
        "scrape_status": "success",
        "extract_status": "success",
        "files": ["raw.html", "page.json", "features.json"],
        "created_at": "..."
      }
    }
    """
    cfg = load_config()
    m_path = path or Path(cfg["paths"]["mapping"])
    if m_path.exists():
        with open(m_path, "r") as f:
            return json.load(f)
    return {}


def save_mapping(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    cfg = load_config()
    m_path = path or Path(cfg["paths"]["mapping"])
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w") as f:
        json.dump(data, f, indent=2)


def update_mapping(
    folder_name: str,
    updates: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    data = load_mapping(path)
    if folder_name not in data:
        data[folder_name] = {}
    data[folder_name].update(updates)
    data[folder_name]["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_mapping(data, path)


# ─── Timestamp ────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()