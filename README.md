# Distributed SMS Spam Intelligence

A portfolio-grade distributed data analytics project that turns a class lab idea into a production-style workflow.

## Project Goal
Build a scalable SMS spam detection system with Apache Spark and PySpark, then package it as a reproducible and recruiter-friendly project.

## Why This Project Is Portfolio-Ready
- Distributed processing with Spark for realistic data pipelines.
- End-to-end workflow from raw data to model inference.
- Reproducible notebooks designed for Google Colab.
- Modular structure so this can evolve into API serving and MLOps workflows.

## Planned Notebook Roadmap
1. `notebooks/01_data_quality_and_distributed_eda.ipynb`
   - Data ingestion, cleaning, quality checks, and distributed EDA.
2. `notebooks/02_modeling_pipeline_and_evaluation.ipynb`
   - Feature engineering, training, evaluation, and artifact export.
3. `notebooks/03_inference_playground_and_error_analysis.ipynb`
   - Batch inference, confidence inspection, and error analysis.

## Dataset
Use the SMS Spam Collection dataset from Kaggle.

Raw data should not be committed. Place it in:
- `data/raw/spam.csv` or
- `data/raw/SMSSpamCollection`

## Execution Strategy
Primary environment: Google Colab.

This repo is written to run notebook-first in Colab. When running locally, make sure Spark and required packages are already available.

## Commit Strategy
Work in incremental commits with clear milestones. Keep each commit focused on one project stage (scaffold, EDA, modeling, inference, docs, polish).
