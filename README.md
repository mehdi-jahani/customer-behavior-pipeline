# Customer Behavior Analysis – Senior Data Engineer Project

A full pipeline for **analyzing and predicting customer behavior** for a small online handmade store, using **scattered and noisy** internal and public data.

## Scenario (from test.md)

- **Business:** Small online store selling handmade products; limited resources, incomplete customer data.
- **Goals:** Understand what drives interest, find potential customers, tailor marketing, and **predict churn** with limited data.
- **Data:** Public (social/forums, sentiment datasets, web) + internal (email list, few orders, scattered feedback).

## Repository Structure

```
.
├── config.py                 # Paths (DATA_RAW, DATA_PROCESSED, SCRIPTS_DIR), .env load, REDDIT_CLIENT_ID/SECRET
├── requirements.txt          # Full dependencies (incl. optional Jupyter)
├── requirements-core.txt     # Core only: pandas, numpy, pyarrow, requests, beautifulsoup4, nltk, scikit-learn, matplotlib, seaborn, python-dotenv
├── data/
│   ├── raw/                  # Input: customers.csv, orders.csv, feedback.csv, sentiment_public.csv, reddit_posts.csv, reddit_posts.json, web_content.csv
│   └── processed/            # Output of ETL and models
│       ├── customers_clean.csv
│       ├── orders_clean.csv
│       ├── feedback_clean.csv
│       ├── single_source_of_truth.csv
│       ├── single_source_of_truth.parquet
│       ├── customers_with_segments.csv   # Created by models.py (K-Means labels)
│       ├── plots/            # All EDA and segment distribution plots (see list below)
│       └── models/           # joblib: churn_logistic.joblib, churn_naive_bayes.joblib, kmeans_segment.joblib
├── scripts/
│   ├── extract_synthetic.py      # generate_customers(), generate_orders(), generate_feedback(); run() → data/raw: customers.csv, orders.csv, feedback.csv
│   ├── extract_from_csv.py      # extract_csv(file_name, **read_kwargs) – load any CSV from data/raw
│   ├── extract_web.py           # extract_page_text(url), extract_urls_to_df(); run_example_and_save() → data/raw/web_content.csv
│   ├── extract_public_dataset.py # get_sample_sentiment_data() → data/raw/sentiment_public.csv (URL or fallback synthetic)
│   ├── extract_reddit_api.py    # get_reddit_token(), fetch_subreddit_posts(); run() → data/raw/reddit_posts.csv + reddit_posts.json (or sample if no API keys)
│   ├── transform.py             # ETL: normalize_dates, fill_missing_numeric, clean_text_column, remove_stopwords_and_stem, add_text_features, add_time_features, drop_rows_missing_key_columns; transform_customers/orders/feedback; build_single_source_of_truth; run_all()
│   ├── load.py                  # save_cleaned(df, name), load_cleaned(name) – CSV/Parquet in data/processed
│   ├── eda.py                   # load_data(), plot_*(), keyword_frequency(), get_practical_insights(), run_eda()
│   └── models.py                # load_processed(), build_churn_dataset(), get_churn_features_target(), train_churn_models(), run_segmentation(), run_pretrained_sentiment_if_available(), run_all_models()
├── notebooks/
│   └── 01_full_pipeline.ipynb  # Live report: extract → transform → EDA → models → key plots (Image) → practical insights → story
├── SUMMARY.md                  # Business insights, recommendations, key visualizations list
├── PROJECT_GUIDE_FA.md         # Full project guide in Persian (رویکرد و جزئیات کدنویسی)
├── CHECKLIST_TEST_MD.md       # test.md requirements vs implementation
├── test.md                     # Original task description
└── README.md                   # This file
```

## Output Files (exact names)

- **data/processed/plots/:** `sentiment_distribution.png`, `word_count_distribution.png`, `source_type_counts.png`, `feedback_by_month.png`, `trend_line.png`, `scatter_wordcount_sentiment.png`, `boxplot_sentiment_noise.png`, `segment_distribution.png` (segment plot is also created by `models.run_segmentation()` in the same folder).
- **data/processed/models/:** `churn_logistic.joblib`, `churn_naive_bayes.joblib`, `kmeans_segment.joblib` (each contains `{"model", "scaler"}` when joblib is available).

## How to Run

### 1. Virtual environment and dependencies

```bash
cd "/home/mehdi/my_folder/Projects/data engineer test"
python3 -m venv venv
source venv/bin/activate   # Linux/macOS; Windows: venv\Scripts\activate
pip install -r requirements-core.txt
# Optional: pip install notebook
```

### 2. Extraction (order optional)

```bash
python scripts/extract_synthetic.py      # customers.csv, orders.csv, feedback.csv
python scripts/extract_public_dataset.py # sentiment_public.csv
python scripts/extract_reddit_api.py     # reddit_posts.csv, reddit_posts.json (or sample if no .env keys)
# Optional: python scripts/extract_web.py  # web_content.csv
```

### 3. Transform and single source of truth

```bash
python scripts/transform.py
```

Writes: `customers_clean.csv`, `orders_clean.csv`, `feedback_clean.csv`, `single_source_of_truth.csv`, `single_source_of_truth.parquet`. Expects in `data/raw/`: `customers.csv`, `orders.csv`, `feedback.csv`; optional `sentiment_public.csv`.

### 4. Models (run before EDA if you want segment_distribution.png from EDA)

```bash
python scripts/models.py
```

Creates: `data/processed/customers_with_segments.csv`, `data/processed/plots/segment_distribution.png`, and joblib files in `data/processed/models/`. Prints churn metrics (precision, recall, F1, feature_importance) and segmentation summary.

### 5. EDA and visualizations

```bash
python scripts/eda.py
```

Reads `single_source_of_truth.csv` and (if present) `customers_with_segments.csv`. Writes all plots under `data/processed/plots/` and returns `run_eda()` dict with `plot_paths`, `top_keywords`, `practical_insights`.

### 6. Jupyter (optional)

```bash
pip install notebook
jupyter notebook notebooks/01_full_pipeline.ipynb
```

Run all cells for the full pipeline; the notebook displays key plots (sentiment, segment, scatter) and prints practical insights and project story.

## Challenges Addressed

- **Noisy data:** Missing values → `fill_missing_numeric(strategy="median")`, mode for categoricals; optional `drop_rows_missing_key_columns(key_columns, min_fill_ratio=0.5)` for feedback. Inconsistent formats → `normalize_dates()`, `clean_text_column()` (lowercase, alphanumeric, collapse spaces).
- **Text:** NLTK stopwords, Porter stemming (or optional WordNet lemmatization in `transform_feedback(use_lemmatization=True)`), VADER sentiment; optional Hugging Face in `run_pretrained_sentiment_if_available()`.
- **Limited data:** Logistic Regression and Naive Bayes with 3-fold CV; Precision, Recall, F1; K-Means (n_clusters=3) for segmentation; models saved with joblib when available.
- **Single source of truth:** `build_single_source_of_truth()` concatenates feedback (source_type=internal_feedback) and public sentiment rows (source_type=public_sentiment) into one DataFrame.

## Findings (summary)

- **Sentiment:** VADER compound score on `text_clean`; negative scores help prioritize at-risk feedback.
- **Churn:** `build_churn_dataset(..., last_days=120)`; features include order_count, total_amount, day_of_week, month, season, channel_code, region_code. Logistic coefficients in `feature_importance`.
- **Segmentation:** K-Means on order_count and total_amount (StandardScaler); segment labels in `customers_with_segments.csv` and bar plot `segment_distribution.png`.

## Limitations and Next Steps

- Data is synthetic; production needs live APIs (e.g. Reddit via .env keys) and more internal data.
- Hugging Face sentiment runs only if `transformers` is installed; otherwise VADER only.
- Churn window (120 days) and min data (e.g. 20 rows, both classes) are fixed in code; could be configurable.
- Weather/events and demographics are not implemented; documented as future work.

## License

Internal project for promotion assessment.
