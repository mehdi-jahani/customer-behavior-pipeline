# Project Summary – Key Insights and Recommendations

## Business Context

Small online handmade store; limited internal data (email list, few orders, scattered feedback) plus public sources (sentiment CSV, optional Reddit and web). Goal: actionable insights and churn prediction with noisy data.

---

## Key Insights

1. **Sentiment from feedback**  
   VADER sentiment on cleaned feedback text (`text_clean`) produces `sentiment_compound`, `sentiment_neg`, `sentiment_pos`, `sentiment_neu`. Negative compound scores highlight at-risk or dissatisfied customers to follow up. Optional pre-trained Hugging Face sentiment is available via `run_pretrained_sentiment_if_available()` when `transformers` is installed (model: distilbert-base-uncased-finetuned-sst-2-english).

2. **Churn prediction**  
   Churn = 1 if customer has no order in the last 120 days (from max `order_date`). Features: `order_count`, `total_amount`, `day_of_week`, `month`, `season`, `channel_code`, `region_code`. Logistic Regression and Naive Bayes are trained with 3-fold CV; metrics (precision, recall, F1) and Logistic `feature_importance` (dict of coefficient per feature) are in the models output. More orders and higher total amount tend to reduce churn.

3. **Customer segments**  
   K-Means (n_clusters=3) on `order_count` and `total_amount` (StandardScaler). Results: `customers_with_segments.csv` (column `segment`), bar plot `segment_distribution.png`, and joblib `kmeans_segment.joblib`. Use segments for: high value (retention), medium (upsell), low/zero (reactivation).

4. **Keywords and practical insights**  
   `keyword_frequency(df, text_col="text_clean", top_n=15)` and `get_practical_insights(df, top_keywords)` answer: topics discussed (top keywords), sentiment towards products (mean compound, positive/negative counts), time pattern (peak month), and which source gave more data (`best_insight_source`). Top terms in feedback include product, delivery, design, etc., useful for content and ads.

---

## Practical Recommendations

- **Marketing:** Tailor messages by segment (value and churn risk). Use sentiment to avoid pushing hard to recently negative customers.
- **Content:** Emphasize quality, design, and delivery in social and SEO (from keyword and sentiment analysis).
- **Churn:** Focus on customers with few orders and low total amount; offer targeted incentives before they churn (use churn model and feature_importance).
- **Data:** Collect more structured feedback and order history; add Reddit/web via existing scripts and .env for Reddit API; consider weather/events if relevant.

---

## Limitations

- Data used is synthetic; results are illustrative.
- Sentiment: VADER by default; Hugging Face only if `transformers` is installed.
- Churn definition (no order in last 120 days) and minimum sample size (e.g. 20 rows, both classes) are fixed in code.
- Reddit data: without API keys, `extract_reddit_api.py` saves sample data only (CSV + JSON).

---

## Key Visualizations (exact paths under data/processed/plots/)

| File | Description |
|------|-------------|
| `sentiment_distribution.png` | Histogram of `sentiment_compound` (customer sentiment). |
| `word_count_distribution.png` | Histogram of `word_count` (text length). |
| `source_type_counts.png` | Bar chart: internal_feedback vs public_sentiment. |
| `feedback_by_month.png` | Bar chart: feedback/review count by month. |
| `trend_line.png` | Line chart: trend of interaction counts over time (from `date`). |
| `scatter_wordcount_sentiment.png` | Scatter: `word_count` vs `sentiment_compound`. |
| `boxplot_sentiment_noise.png` | Boxplot of `sentiment_compound` for outlier/noise detection. |
| `segment_distribution.png` | Bar chart: number of customers per K-Means segment (also produced by `models.run_segmentation()`). |

These support the narrative in the Jupyter notebook (`01_full_pipeline.ipynb`) and this summary. The notebook displays sentiment, segment distribution, and scatter as key plots and prints `practical_insights` and the project story.
