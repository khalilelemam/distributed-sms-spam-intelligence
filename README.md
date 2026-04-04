# Distributed SMS Spam Intelligence

An end-to-end Spark project for SMS spam detection, built as a notebook-first workflow in Google Colab.

## Project Scope
- Distributed data processing with PySpark.
- Data cleaning and EDA on the SMS Spam Collection dataset.
- TF-IDF feature engineering with multiple Spark classifiers.
- Validation-based hyperparameter tuning and model comparison.
- Model save/load and inference on new messages.

## Notebooks
1. `notebooks/01_data_quality_and_distributed_eda.ipynb`
   - Download dataset with `kagglehub`.
   - Clean labels/messages.
   - Class distribution and message length analysis.
   - Export artifacts to `/content/artifacts`.
2. `notebooks/02_modeling_pipeline.ipynb`
   - Load cleaned parquet artifacts.
   - Build Spark ML pipeline (Tokenizer -> StopWordsRemover -> HashingTF -> IDF -> classifier).
   - Split data into train, validation, and test.
   - Tune Logistic Regression on validation.
   - Compare tuned Logistic Regression vs LinearSVC vs NaiveBayes on test.
   - Evaluate and export metrics/comparison tables.
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
From `notebooks/02_modeling_pipeline.ipynb` using train/validation/test split (seed=42):

Tuned Logistic Regression on final test split:

| Metric | Value |
|---|---:|
| Accuracy | 0.9715 |
| Precision | 0.9009 |
| Recall | 0.8929 |
| F1-score | 0.8969 |

Confusion counts:
- TP: 100
- TN: 683
- FP: 11
- FN: 12

Model comparison (test split):

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LinearSVC | 0.9789 | 0.9524 | 0.8929 | 0.9217 |
| LogisticRegression (tuned) | 0.9715 | 0.9009 | 0.8929 | 0.8969 |
| NaiveBayes | 0.9553 | 0.7836 | 0.9375 | 0.8537 |

## Artifacts
- Clean dataset parquet: `/content/artifacts/clean_sms.parquet`
- Saved model: `/content/artifacts/models/sms_spam_pipeline`
- Metrics CSV: `/content/artifacts/model_metrics.csv`
- Model comparison CSV: `/content/artifacts/model_comparison.csv`
- False positives CSV: `/content/artifacts/false_positives.csv`
- False negatives CSV: `/content/artifacts/false_negatives.csv`
