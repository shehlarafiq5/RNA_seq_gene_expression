# Multi-Cancer Classification using ML with Explainability (LIME & SHAP)

This project implements and compares various machine learning algorithms for multi-class cancer type classification using gene expression data. It also integrates XAI techniques (LIME and SHAP) to interpret model decisions.

## 🧬 Dataset Description

- `data.csv`: Contains numerical gene expression features per sample.
- `labels.csv`: Includes cancer type labels for each sample with the following classes:
  - BRCA (Breast)
  - KIRC (Kidney)
  - COAD (Colon)
  - LUAD (Lung)
  - PRAD (Prostate)

> Samples are merged using a common `Sample_id`.

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
