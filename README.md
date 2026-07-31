# 🎬 MovieIQ — Studio Decision Intelligence

An interactive Streamlit dashboard that analyzes and predicts movie box-office success using key performance indicators — budget, revenue, popularity, runtime, and audience rating.

A movie is labeled **successful** when its revenue exceeds its production budget.

## Features

- **Performance Snapshot** — genre distribution, budget vs. revenue trends, genre success rates, and a correlation heatmap
- **Hypothesis Testing** — live T-test (popularity vs. success) and Chi-square test (genre vs. success)
- **Prediction Engine** — Random Forest classifier with accuracy/precision/recall, confusion matrix, and feature importance
- **What-If Predictor** — enter a hypothetical movie's details and get a live greenlight/hold prediction
- **Movie Comparison** — compare any two movies side by side
- **Raw Data & Export** — sortable, filterable data table with CSV export
- **Insights & Recommendations** — auto-generated, filter-aware business insights

## Tech Stack

Python · Pandas · NumPy · scikit-learn · SciPy · Seaborn · Matplotlib · Streamlit

## Run Locally

```bash
streamlit run MovieIQ.py
```

## Dataset

`movies.csv` — budget, revenue, popularity, runtime, vote_average, title, and genres for 2,000 movies.
