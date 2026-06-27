# 7006SCN — NHS Prescription Cost Prediction

## Student: [Your Initials] | [Your Student ID]
## Module: Machine Learning and Big Data (7006SCN) — 2526MAYJUL

## Project Overview

This project applies PySpark MLlib to predict NHS prescription 
costs, addressing two related problems:
- **Regression**: predicting exact ACTUAL_COST (£) of a prescription
- **Classification**: predicting whether a prescription exceeds 
  a £50 high-cost threshold (HIGH_COST)

## Dataset

**Source**: NHS English Prescribing Dataset (EPD) with SNOMED Codes  
**Publisher**: NHS Business Services Authority (NHSBSA)  
**URL**: https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code  
**Period**: March 2026  
**Scale**: 18,364,409 rows × 27 columns, 7.15GB  
**Licence**: Open Government Licence v3.0 (OGL)

## Repository Structure

Task1/Notebooks/Task1.ipynb   — Problem definition, dataset loading, 5 V's justification

Task2/Notebooks/Task2.ipynb   — Data cleaning, feature engineering, PySpark Pipeline

Task3/Notebooks/Task3.ipynb   — 4 ML models (Linear Regression, Random Forest,
                                Logistic Regression, Decision Tree) with CrossValidator

Task4/Notebooks/Task4.ipynb   — Spark UI analysis, caching strategy, resource config

Task5/Notebooks/Task5.ipynb   — Model evaluation, ROC/PR curves, perturbation analysis, SHAP

Task6/Notebooks/Task6.ipynb   — Data export for Tableau dashboards

README.md

## Pipeline Summary

1. **Ingestion**: PySpark reads 18.3M-row CSV, repartitioned to 8 partitions
2. **Cleaning**: Drop 13 non-predictive columns, fill missing POSTCODE/SNOMED_CODE
3. **Feature Engineering**: Log transforms (NIC, QUANTITY, TOTAL_QUANTITY), 
   HIGH_COST binary flag creation
4. **Pipeline**: StringIndexer → OneHotEncoder (low-cardinality only) → 
   VectorAssembler → StandardScaler
5. **Modelling**: 4 algorithms tuned via CrossValidator (2-fold, due to 
   Colab memory constraints), trained on 3% stratified sample
6. **Evaluation**: Full metrics, confusion matrices, ROC/PR curves, 
   perturbation stability testing, SHAP explainability
7. **Visualisation**: 4 Tableau Public dashboards

## Key Findings

- **NIC (Net Ingredient Cost)** is the dominant predictor across all 
  4 models, accounting for 58.4% of Random Forest feature importance 
  (combined with its log transform)
- **Linear Regression** outperforms Random Forest for cost prediction 
  (R²=0.9988 vs 0.7400), indicating a fundamentally linear cost relationship
- **Decision Tree** achieves the best classification performance 
  (99.29% accuracy) with full rule interpretability


## Resource Configuration

- spark.driver.memory: 8g
- spark.executor.memory: 4g  
- spark.sql.shuffle.partitions: 8
- spark.default.parallelism: 8

## Tableau Public Dashboard
https://public.tableau.com/app/profile/firdous.khilji/viz/NHSPrescriptionCostAnalysis/NHSPrescriptionCostAnalysis?publish=yes
