# Distributed SMS Spam Intelligence

An end-to-end Spark project for SMS spam detection, built as a notebook-first workflow in Google Colab.

## Project Scope
- Distributed data processing with PySpark.
- Data cleaning and EDA on the SMS Spam Collection dataset.
- TF-IDF feature engineering with a Logistic Regression classifier.
- Model save/load and inference on new messages.

## Notebooks
1. `notebooks/01_data_quality_and_distributed_eda.ipynb`
   - Download dataset with `kagglehub`.
   - Clean labels/messages.
   - Class distribution and message length analysis.
   - Export artifacts to `/content/artifacts`.
2. `notebooks/02_modeling_pipeline.ipynb`
   - Load cleaned parquet artifacts.
   - Build Spark ML pipeline (Tokenizer -> StopWordsRemover -> HashingTF -> IDF -> LogisticRegression).
   - Evaluate and export metrics.
   - Save/reload model and run inference samples.
3. `notebooks/03_inference_and_error_analysis.ipynb`
   - Load saved model and run full-dataset scoring.
   - Inspect uncertain predictions.
   - Export false positives and false negatives for manual review.

## Dataset
Kaggle: SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`).

The notebook downloads the data directly using `kagglehub`, so no manual CSV placement is required in Colab.

## Reproducing Results (Colab)
1. Open notebook 1 and run all cells.
2. Open notebook 2 and run all cells.
3. Ensure both notebooks use the same runtime so notebook 2 can read artifacts from `/content/artifacts`.

## Latest Modeling Results
From `notebooks/02_modeling_pipeline.ipynb` on test split (seed=42):

| Metric | Value |
|---|---:|
| Accuracy | 0.9748 |
| Precision | 0.9353 |
| Recall | 0.8784 |
| F1-score | 0.9059 |

Confusion counts:
- TP: 130
- TN: 914
- FP: 9
- FN: 18

## Artifacts
- Clean dataset parquet: `/content/artifacts/clean_sms.parquet`
- Saved model: `/content/artifacts/models/sms_spam_pipeline`
- Metrics CSV: `/content/artifacts/model_metrics.csv`
- False positives CSV: `/content/artifacts/false_positives.csv`
- False negatives CSV: `/content/artifacts/false_negatives.csv`
