# Webpage Classifier

A supervised machine learning pipeline that classifies any webpage URL into one of three types: `detail`, `list`, or `others`. Built for scraping teams to automate page-type identification before writing extraction logic.

```
URL → Scrape → Extract Features → Pre-classify → ML Model → Prediction
```

---

## Performance

| Metric | Value |
|--------|-------|
| Training samples | 1,499 |
| Features | 200 |
| Algorithm | XGBoost (GridSearchCV tuned) |
| ML model accuracy | 94.67% |
| Full pipeline accuracy | 92.00% |
| Macro F1 | 0.950 |
| Test set size | 300 |

**Per-class results (ML model):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| detail | 0.990 | 0.980 | 0.985 |
| list | 0.961 | 0.908 | 0.934 |
| others | 0.899 | 0.967 | 0.932 |

---

## Classes

| Class | Description |
|-------|-------------|
| `detail` | A page about one specific thing — product, article, job posting, Wikipedia entry |
| `list` | A collection or index page — search results, category listings, job boards |
| `others` | Everything else — homepages, about pages, login pages, dashboards |

---

## Project Structure

```
webpage-classifier/
├── config/
│   └── settings.yaml              # All configuration: paths, scraper, model params
│
├── data/
│   ├── training_urls.csv          # Labeled URL dataset (url, label columns)
│   ├── others_augmented.csv       # Additional others-class samples
│   ├── checkpoint.json            # Auto-generated scrape/extract progress
│   └── mapping.json               # URL-to-folder index
│
├── extraction/
│   ├── feature_extractor.py       # 200-feature extraction across 13 pillars
│   └── pre_classifier.py          # Deterministic rule-based pre-classifier
│
├── scraper/
│   └── scraper.py                 # Playwright + stealth scraper
│
├── pipeline/
│   └── utils.py                   # Shared utilities: config, logging, checkpoint
│
├── models/
│   ├── classifier.joblib          # Trained XGBoost model
│   ├── label_encoder.joblib       # Label encoder
│   ├── feature_columns.json       # Ordered list of 200 feature names
│   ├── feature_importance.json    # SHAP importance scores
│   └── training_report.json       # Full training metrics
│
├── output/
│   ├── detail/                    # Scraped data for detail class
│   ├── list/                      # Scraped data for list class
│   └── others/                    # Scraped data for others class
│
├── predicted/                     # Prediction outputs
│
├── scrap_run.py                   # Step 1: scrape training URLs
├── extract_run.py                 # Step 2: extract features from HTML
├── train_run.py                   # Step 3: train the model
├── predict_run.py                 # Step 4: classify a URL
├── predict_batch.py               # Batch classify from CSV
├── eval_pipeline.py               # Evaluate ML-only vs full pipeline accuracy
├── status.py                      # Pipeline progress dashboard
├── inspect_utils.py               # Feature inspection and debugging
├── validate.py                    # Pre-flight environment check
├── reset.py                       # Selective pipeline reset
└── rebuild_page_json.py           # Rebuild page.json from existing raw.html
```

---

## Setup

**Requirements:** Python 3.9+, Git

```bash
git clone https://github.com/Raghu-2005/webpage-classifier.git
cd webpage-classifier

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

playwright install chromium
playwright install-deps chromium

python validate.py                # verify everything is working
```

---

## Usage

The pipeline runs in four steps. Each step is checkpoint-aware — already processed URLs are never reprocessed.

### Step 1 — Scrape

```bash
python scrap_run.py
```

Reads `data/training_urls.csv`, visits each URL with a headless Chromium browser, saves `raw.html` and `page.json` to `output/<label>/<folder>/`. Failed URLs are logged and skipped automatically.

```bash
python scrap_run.py --dry-run      # preview what would be scraped
python scrap_run.py --input my.csv # use a different CSV
python status.py                   # check progress at any time
```

### Step 2 — Extract Features

```bash
python extract_run.py
```

Parses saved HTML and computes 200 features per page. Saves `features.json` alongside each page. Only processes URLs with a successful scrape.

```bash
python extract_run.py --retry-failed   # also retry previously failed extractions
```

### Step 3 — Train

```bash
python train_run.py
```

Loads all feature files, runs stratified 80/20 split, performs GridSearchCV with 5-fold cross-validation, trains the final XGBoost model, computes SHAP importance, and saves all artefacts to `models/`.

```bash
python train_run.py --no-gridsearch    # skip GridSearch, faster
python train_run.py --show-results     # print saved results without retraining
```

### Step 4 — Predict

```bash
python predict_run.py --url https://example.com/product/123
```

Scrapes the URL, extracts features, runs the pre-classifier, then the ML model if no rule fires. Saves result to `predicted/<folder>/prediction.json`.

```bash
python predict_run.py              # interactive loop — enter URLs one by one
python predict_run.py --url <URL> --force   # re-classify an already predicted URL
python predict_run.py --url <URL> --debug   # show which layer made the decision
```

**Batch prediction:**

```bash
python predict_batch.py --input urls.csv
```

Input CSV needs a `url` column. Results saved to `predicted/batch_results_<timestamp>.csv`.

---

## Prediction Pipeline

Each URL goes through a priority-ordered stack before reaching the ML model. The first rule that fires wins.

| Priority | Layer | Method | Confidence |
|----------|-------|--------|------------|
| 1 | Error / blocked page | HTML pattern match | 0.97 |
| 2 | Schema.org JSON-LD | Developer-declared type | 0.96 |
| 3 | rel=next / rel=prev | Pagination declared in HTML head | 0.95 |
| 4 | RSS / Atom feed link | Content aggregator signal | 0.93 |
| 5 | Open Graph type | og:type = article / product / video | 0.93 |
| 6 | Twitter Card player | Single media item | 0.93 |
| 7 | Strong listing DOM | 3-of-5 structural signals | 0.90 |
| 7b | Domain-specific rules | Job boards, Wikipedia, social browse | 0.85–0.92 |
| 8 | Others signals | Marketing/landing page pattern | 0.90 |
| — | XGBoost model | All 200 features | Variable |
| — | URL fallback | Path/param patterns only | 0.50–0.65 |

The URL fallback only activates when model confidence is below 50% and the page had no readable content (blocked or empty).

---

## Feature Engineering

Features are extracted from parsed HTML using BeautifulSoup with lxml. All `<script>`, `<style>`, and `<noscript>` tags are stripped before text analysis.

### Pillar Overview

| Pillar | Count | What it captures |
|--------|-------|-----------------|
| Structural | 26 | DOM tag counts, nesting depth, grid/card class patterns |
| Link | 9 | Total links, internal/external split, anchor text diversity |
| Content | 16 | Word count, paragraph length, heading hierarchy |
| Media | 9 | Image count, alt text ratio, gallery detection, video presence |
| Listing-specific | 13 | Pagination, filter UI, result count text, repeated card blocks |
| Detail-specific | 20 | Price, buy CTA, author, publish date, rating, TOC, spec section |
| Interaction | 7 | Forms, buttons, dropdowns, login detection |
| Metadata/SEO | 11 | OG type, Schema.org type, canonical, robots |
| Semantic/NLP | 11 | Cue word density, numeric token density, title signals |
| URL Pattern | 13 | Path depth, slug patterns, numeric IDs, query parameters |
| Single-Entity | 14 | Entity name repetition, structure-content alignment |
| Landing Page | 17 | CTA signals, hero section, pricing section, others path/content |
| Modern List | 11 | Browse/discover pages, academic listings, no-pagination lists |
| Composite | 4 | Derived listing, detail, others, and URL-corrected scores |

### Top 20 Features by SHAP Importance

| Rank | Feature | SHAP Score |
|------|---------|------------|
| 1 | url_path_depth | 0.4859 |
| 2 | avg_paragraph_length | 0.3353 |
| 3 | tag_h3_count | 0.2268 |
| 4 | external_link_ratio | 0.2261 |
| 5 | title_word_count | 0.2122 |
| 6 | url_has_any_param | 0.1936 |
| 7 | others_path_signal | 0.1895 |
| 8 | tag_tr_count | 0.1720 |
| 9 | numeric_token_density | 0.1591 |
| 10 | academic_listing_signal | 0.1450 |
| 11 | content_links | 0.1382 |
| 12 | internal_anchor_links | 0.1314 |
| 13 | landing_page_score | 0.1285 |
| 14 | composite_listing_score | 0.1269 |
| 15 | anchor_text_diversity | 0.1256 |
| 16 | composite_others_score | 0.1195 |
| 17 | url_depth_normalized | 0.1155 |
| 18 | entity_name_repetition | 0.1037 |
| 19 | tag_table_count | 0.1027 |
| 20 | tag_h2_count | 0.0969 |

---

## Configuration

All settings are in `config/settings.yaml`. Key options:

```yaml
scraper:
  wait_after_load_ms: 4000     # increase to 5000-6000 for heavy JS/SPA sites
  concurrency: 2               # parallel browser contexts
  max_retries: 3

model:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

---

## Utility Scripts

```bash
# Pipeline management
python status.py                    # full dashboard: scrape/extract/model/predict status
python status.py --failed           # list all failed URLs with reasons
python validate.py                  # pre-flight check before running pipeline

# Debugging
python inspect_utils.py url <URL>           # show all 200 features for a URL
python inspect_utils.py compare <URL1> <URL2>  # side-by-side feature diff
python inspect_utils.py top-features        # SHAP importance ranking

# Evaluation
python eval_pipeline.py                     # compare ML-only vs full pipeline
python eval_pipeline.py --verbose           # show all misclassified URLs
python eval_pipeline.py --export out.csv    # full per-URL result export

# Maintenance
python reset.py --extract --yes             # re-extract all (keeps HTML)
python reset.py --failed --yes              # retry failed URLs
python reset.py --model --yes              # delete model, keep data
python rebuild_page_json.py                 # rebuild page.json from existing HTML
```

---

## Adding Training Data

```bash
# Add new URLs to data/training_urls.csv, then:
python scrap_run.py        # only new URLs are scraped
python extract_run.py      # only new URLs are extracted
python train_run.py        # retrains on everything
```

After changing feature extraction logic, re-extract everything:

```bash
python reset.py --extract --yes
python extract_run.py
python train_run.py
```

---

## Output Files

**prediction.json** (saved per URL in `predicted/<folder>/`):

```json
{
  "url": "https://example.com/product/123",
  "predicted_label": "detail",
  "confidence": 0.9721,
  "probabilities": { "detail": 0.9721, "list": 0.0201, "others": 0.0078 },
  "method": "ml_model",
  "fallback_used": false,
  "content_available": true,
  "scrape_status": "success",
  "predicted_at": "2026-04-27T..."
}
```

**features.json** — flat dictionary of all 200 feature values per page.

**training_report.json** — full metrics: classification report, confusion matrix, best hyperparameters, top 30 SHAP features.

---

## Known Limitations

- Cloudflare Enterprise sites will fail scraping even with stealth mode. These are skipped and recorded as failed.
- JavaScript-heavy SPAs (React, Angular) may have lower feature quality if content renders slowly. Increase `wait_after_load_ms` if needed.
- The `others` class is the hardest to classify (F1 0.932) since it is defined by the absence of list and detail signals rather than having clear positive indicators of its own.

---

## Dependencies

```
playwright
playwright-stealth
beautifulsoup4
lxml
xgboost
scikit-learn
shap
pandas
numpy
joblib
rich
pyyaml
click
```

Install:

```bash
pip install -r requirements.txt
playwright install chromium
playwright install-deps chromium
```

---

## Author

Raghuram L — [github.com/Raghu-2005](https://github.com/Raghu-2005)