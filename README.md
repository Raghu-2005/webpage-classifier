# 🌐 Webpage Classifier

> An end-to-end Machine Learning pipeline that automatically classifies any webpage URL into **Detail**, **List**, or **Others** — using 200 engineered features and an XGBoost classifier.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/Macro%20F1-0.950-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Classes](#-classes)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage — Step by Step](#-usage--step-by-step)
  - [Step 1: Scrape URLs](#step-1-scrape-urls)
  - [Step 2: Extract Features](#step-2-extract-features)
  - [Step 3: Train the Model](#step-3-train-the-model)
  - [Step 4: Predict a URL](#step-4-predict-a-url)
  - [Step 5: Batch Predict](#step-5-batch-predict)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Top 20 SHAP Features](#-top-20-shap-features)
- [Pipeline Utilities](#-pipeline-utilities)
- [.gitignore Notes](#-gitignore-notes)
- [Requirements](#-requirements)

---

## 🧠 Overview

**Webpage Classifier** is a supervised machine learning system that takes any URL, fetches its HTML, extracts 200 structural and linguistic features, and classifies the page into one of three categories. It is designed for use in web crawlers, content aggregators, SEO tools, and automated scraping pipelines.

The full pipeline covers:

```
URL Input → Scraping → Feature Extraction → Model Training → Real-time Prediction
```

| Stat | Value |
|------|-------|
| Training Samples | 1,499 |
| Feature Count | 200 |
| Algorithm | XGBoost (GridSearchCV tuned) |
| Test Accuracy | **95.00%** |
| Macro F1 Score | **0.950** |
| Best CV F1 | 0.9330 |

---

## 🏷️ Classes

| Class | Description | Examples |
|-------|-------------|---------|
| `detail` | A page about a single entity — product, article, person, job posting | Product pages, blog posts, Wikipedia articles, job descriptions |
| `list` | An index or collection page — search results, category listings, archives | Search results, category pages, job boards, archive pages |
| `others` | Everything else — homepages, about pages, login pages, landing pages | Home pages, /about, /contact, login pages, error pages |

---

## 📁 Project Structure

```
webpage-classifier/
│
├── config/
│   └── settings.yaml              # Project-wide config (paths, settings, constants)
│
├── data/
│   ├── training_urls.csv          # Labeled URL dataset for training
│   ├── others_augmented.csv       # Augmented 'others' class data
│   ├── checkpoint.json            # ⚙️ Auto-generated scrape progress (gitignored)
│   └── mapping.json               # ⚙️ URL-to-file mapping (gitignored)
│
├── docs/
│   ├── doc.txt                    # Project documentation notes
│   └── reference.txt              # Reference material
│
├── extraction/
│   ├── __init__.py
│   ├── feature_extractor.py       # Core: computes all 200 features from HTML
│   └── pre_classifier.py          # Rule-based composite signal scores
│
├── logs/
│   └── pipeline.log               # Runtime logs (gitignored)
│
├── models/
│   ├── classifier.joblib          # 🔒 Trained XGBoost model (gitignored — large)
│   ├── label_encoder.joblib       # 🔒 LabelEncoder (gitignored — large)
│   ├── feature_columns.json       # ✅ Ordered list of 200 feature names
│   ├── feature_importance.json    # ✅ SHAP importance scores
│   └── training_report.json       # ✅ Full training metrics report
│
├── output/
│   ├── detail/                    # Scraped JSON pages for 'detail' class
│   ├── list/                      # Scraped JSON pages for 'list' class
│   └── others/                    # Scraped JSON pages for 'others' class
│
├── pipeline/
│   ├── __init__.py
│   └── utils.py                   # Pipeline orchestration utilities
│
├── predicted/                     # Batch prediction output results
│
├── scraper/
│   ├── __init__.py
│   └── scraper.py                 # Web scraping logic (requests + BeautifulSoup)
│
├── venv/                          # Virtual environment (gitignored)
│
├── extract_run.py                 # Run feature extraction
├── fix_lables.py                  # Fix mislabeled training samples
├── inspect_utils.py               # Debugging and inspection helpers
├── predict_batch.py               # Batch prediction on multiple URLs
├── predict_run.py                 # Real-time single URL prediction
├── rebuild_page_json.py           # Rebuild page JSON from raw scrapes
├── requirements.txt               # Python dependencies
├── reset.py                       # Reset pipeline state
├── scrap_run.py                   # Run the URL scraping pipeline
├── status.py                      # Show pipeline run status
├── train_run.py                   # Train the XGBoost model
├── validate.py                    # Validate pipeline data integrity
└── test_urls.csv                  # Test URL samples
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.9+**
- `pip`
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Raghu-2005/webpage-classifier.git
cd webpage-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

All project settings are managed in `config/settings.yaml`. Review and adjust before running:

```yaml
# config/settings.yaml (example)
data:
  training_urls: data/training_urls.csv

scraper:
  timeout: 10          # HTTP request timeout in seconds
  max_retries: 3       # Retry count for failed URLs
  delay: 0.5           # Delay between requests (seconds)

model:
  test_size: 0.20      # 80/20 train-test split
  cv_folds: 5          # GridSearchCV cross-validation folds
  random_state: 42

output:
  models_dir: models/
  output_dir: output/
```

---

## 🚀 Usage — Step by Step

### Step 1: Scrape URLs

Fetch HTML content from all URLs in `data/training_urls.csv`. Scraped pages are saved as JSON files under `output/detail/`, `output/list/`, and `output/others/`.

```bash
python scrap_run.py
```

> **Note:** Failed/incomplete URLs are automatically logged and skipped. Check `logs/pipeline.log` for details.

Check scraping progress at any time:

```bash
python status.py
```

---

### Step 2: Extract Features

Parse the scraped HTML and compute **200 features** per page. Outputs a structured feature matrix.

```bash
python extract_run.py
```

This will:
- Load all scraped JSON files from `output/`
- Run `feature_extractor.py` and `pre_classifier.py`
- Build the feature matrix `(N, 202)` — 200 features + URL + label

---

### Step 3: Train the Model

Run the full training pipeline — GridSearchCV hyperparameter tuning, XGBoost training, evaluation, and SHAP analysis.

```bash
python train_run.py
```

**What happens:**
- Loads the feature matrix
- Performs stratified 80/20 train-test split
- Runs GridSearchCV with 5-fold CV to find best hyperparameters
- Trains the final XGBoost model
- Evaluates on the test set (accuracy, F1, confusion matrix)
- Computes SHAP feature importances
- Saves model artifacts to `models/`

**Output files saved:**
```
models/
├── classifier.joblib          # Trained model
├── label_encoder.joblib       # Class label encoder
├── feature_columns.json       # Feature column order
├── feature_importance.json    # SHAP scores
└── training_report.json       # Full metrics
```

---

### Step 4: Predict a URL

Classify any single URL in real time:

```bash
python predict_run.py --url https://example.com/some-page
```

**Example output:**
```
URL     : https://example.com/some-page
Class   : detail
Confidence : 0.9721
```

---

### Step 5: Batch Predict

Classify multiple URLs from a CSV file:

```bash
python predict_batch.py --input test_urls.csv --output predicted/results.csv
```

The input CSV should have a `url` column. Results are saved with `predicted_class` and `confidence` columns appended.

---

## 📊 Model Performance

### Overall Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **95.00%** |
| Macro Avg F1 | **0.950** |
| Weighted Avg F1 | **0.950** |
| Macro Avg Precision | 0.950 |
| Macro Avg Recall | 0.952 |
| Total Test Samples | 300 |
| Misclassified | 15 |

### Per-Class Report

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| `detail` | 0.990 | 0.980 | **0.985** | 99 |
| `list` | 0.961 | 0.908 | **0.934** | 109 |
| `others` | 0.899 | 0.967 | **0.932** | 92 |
| **macro avg** | **0.950** | **0.952** | **0.950** | — |

### Confusion Matrix

```
               Predicted
               detail   list   others
Actual detail    97       1       1
Actual list       1      99       9
Actual others     0       3      89
```

### Best Hyperparameters (GridSearchCV)

```python
{
    'n_estimators':     200,
    'max_depth':        8,
    'learning_rate':    0.1,
    'subsample':        0.8,
    'colsample_bytree': 0.8
}
```

---

## 🔬 Feature Engineering

**200 features** are extracted per page across 7 categories:

<details>
<summary><b>📎 URL-Based Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `url_path_depth` | Number of path segments — **#1 most important feature** |
| `url_depth_normalized` | Path depth normalized to 0–1 scale |
| `url_has_any_param` | Binary: URL contains query parameters (`?key=val`) |
| `url_has_id` | Binary: URL contains a numeric ID in the path |
| `others_path_signal` | Rule-based score for known 'others' path patterns (`/about`, `/login`) |
| `academic_listing_signal` | Detects academic archive listing patterns |

</details>

<details>
<summary><b>🏗️ HTML Tag Count Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `tag_h2_count` | Count of `<h2>` tags |
| `tag_h3_count` | Count of `<h3>` tags — **#3 most important** |
| `tag_tr_count` | Count of table rows `<tr>` — **#8 most important** |
| `tag_table_count` | Count of `<table>` elements |
| `tag_li_count` | Count of list items `<li>` |
| `tag_a_count` | Count of anchor links |
| `tag_img_count` | Count of images |
| `tag_p_count` | Count of paragraphs |
| `tag_form_count` | Count of forms |
| `tag_input_count` | Count of input fields |
| `tag_div_count` | Count of div elements |
| `tag_script_count` | Count of script elements |

</details>

<details>
<summary><b>📝 Content & Text Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `avg_paragraph_length` | Average `<p>` character length — **#2 most important** |
| `title_word_count` | Words in `<title>` tag — **#5 most important** |
| `numeric_token_density` | Fraction of numeric tokens in text — **#9 most important** |
| `entity_name_repetition` | Frequency of the most repeated named entity — **#18** |
| `text_length` | Total character count |
| `word_count` | Total word count |
| `unique_word_ratio` | Vocabulary diversity |
| `has_og_tags` | Presence of Open Graph tags |
| `has_json_ld` | Presence of JSON-LD structured data |
| `content_density` | Ratio of visible text to total HTML size |

</details>

<details>
<summary><b>🔗 Link Graph Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `external_link_ratio` | Fraction of outbound external links — **#4 most important** |
| `content_links` | Links inside the main content area — **#11** |
| `internal_anchor_links` | Count of same-page anchor links — **#12** |
| `anchor_text_diversity` | Entropy of anchor text — **#15** |
| `link_density` | Total links per 100 words |
| `nav_link_count` | Links in navigation elements |
| `footer_link_count` | Links in footer elements |

</details>

<details>
<summary><b>🧱 Structural & Layout Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `has_pagination` | Presence of pagination — strong list page signal |
| `has_breadcrumb` | Presence of breadcrumb navigation |
| `has_sidebar` | Presence of sidebar layout |
| `article_tag_present` | Presence of `<article>` tag — detail page signal |
| `dom_depth` | Maximum depth of the DOM tree |
| `dom_node_count` | Total number of DOM nodes |
| `main_content_ratio` | Content proportion in `<main>`/`<article>` |

</details>

<details>
<summary><b>🧮 Composite Pre-Classifier Signals (click to expand)</b></summary>

Computed by `pre_classifier.py` — weighted combinations of raw signals turned into a single score:

| Feature | Description |
|---------|-------------|
| `composite_listing_score` | Weighted combination of list-page signals — **#14** |
| `composite_others_score` | Weighted combination of others-page signals — **#16** |
| `landing_page_score` | Likelihood score of being a landing/home page — **#13** |
| `others_path_signal` | Rule-based others URL path score — **#7** |
| `academic_listing_signal` | Score for academic archive pages — **#10** |

</details>

<details>
<summary><b>🗂️ Metadata & Schema Features (click to expand)</b></summary>

| Feature | Description |
|---------|-------------|
| `has_canonical` | Presence of canonical link tag |
| `has_author_meta` | Presence of author metadata |
| `has_publish_date` | Presence of publish date |
| `schema_type` | JSON-LD/microdata schema type (Product, Article, etc.) |
| `og_type` | Open Graph type (article, website, product) |
| `viewport_meta` | Presence of responsive viewport meta tag |

</details>

---

## 🏆 Top 20 SHAP Features

SHAP (SHapley Additive exPlanations) was used to rank feature importance globally across all predictions:

| Rank | Feature | SHAP Score | Key Insight |
|------|---------|------------|-------------|
| 1 | `url_path_depth` | 0.4859 | Detail pages have deeper URL paths |
| 2 | `avg_paragraph_length` | 0.3353 | Detail pages have long paragraphs |
| 3 | `tag_h3_count` | 0.2268 | Content structure indicator |
| 4 | `external_link_ratio` | 0.2261 | Others pages link out more |
| 5 | `title_word_count` | 0.2122 | Detail pages have descriptive titles |
| 6 | `url_has_any_param` | 0.1936 | List/search pages carry URL params |
| 7 | `others_path_signal` | 0.1895 | Rule-based others path detection |
| 8 | `tag_tr_count` | 0.1720 | Tables indicate structured listings |
| 9 | `numeric_token_density` | 0.1591 | Prices/dates are dense in list pages |
| 10 | `academic_listing_signal` | 0.1450 | Academic archive detection |
| 11 | `content_links` | 0.1382 | High on list pages |
| 12 | `internal_anchor_links` | 0.1314 | Common on docs and detail pages |
| 13 | `landing_page_score` | 0.1285 | Composite home/landing score |
| 14 | `composite_listing_score` | 0.1269 | Composite listing score |
| 15 | `anchor_text_diversity` | 0.1256 | List pages have diverse anchor text |
| 16 | `composite_others_score` | 0.1195 | Composite others score |
| 17 | `url_depth_normalized` | 0.1155 | Normalized URL depth |
| 18 | `entity_name_repetition` | 0.1037 | Detail pages repeat entity name |
| 19 | `tag_table_count` | 0.1027 | Table count |
| 20 | `tag_h2_count` | 0.0969 | H2 heading count |

> **Key takeaway:** `url_path_depth` alone is the single strongest signal — page structure in the URL predicts class before any content is even read.

---

## 🛠️ Pipeline Utilities

| Script | Description |
|--------|-------------|
| `status.py` | Show current pipeline state (how many URLs scraped, extracted, etc.) |
| `validate.py` | Validate data integrity across all pipeline stages |
| `fix_lables.py` | Utility to correct mislabeled training samples |
| `inspect_utils.py` | Debugging helpers — inspect features, pages, predictions |
| `rebuild_page_json.py` | Rebuild page-level JSON from raw scrapes |
| `reset.py` | ⚠️ Reset the full pipeline state for a fresh run |

```bash
# Check pipeline status
python status.py

# Validate all data
python validate.py

# Reset everything (use with caution)
python reset.py
```

---

## 📄 .gitignore Notes

The following are **excluded** from version control to keep the repo clean:

- `venv/` — Virtual environment
- `output/`, `predicted/`, `logs/` — Generated runtime data
- `models/classifier.joblib`, `models/label_encoder.joblib` — Large binary model files
- `data/checkpoint.json`, `data/mapping.json` — Auto-generated pipeline state
- `misclassified_*.csv` — Runtime debug files

The following model files **are tracked** (small JSON, useful for reference):
```
models/feature_columns.json      
models/feature_importance.json   
models/training_report.json      
```

> To get a fully working model, clone the repo and run the full pipeline (`scrap → extract → train`)

---

## 📦 Requirements

Key dependencies from `requirements.txt`:

```
xgboost
scikit-learn
shap
beautifulsoup4
lxml
requests
pandas
numpy
joblib
rich
pyyaml
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Raghuram L** — [@Raghu-2005](https://github.com/Raghu-2005)

---

<div align="center">
  <sub>Built with Python 🐍 · XGBoost · SHAP · BeautifulSoup</sub>
</div>