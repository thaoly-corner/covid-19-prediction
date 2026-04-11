# COVID-19 Patient Precondition & Mortality Analysis 

A high-performance data science pipeline designed to analyze patient medical histories and predict COVID-19 mortality risk (`is_dead`) using Machine Learning and Deep Learning architectures.
##  Dataset Description
* **Source:** [Kaggle COVID-19 patient pre-condition dataset](https://www.kaggle.com/datasets/tanmoyx/covid19-patient-precondition-dataset/data)
* **Samples:** 500k+ observations.
* **Task:** Binary Classification

### Features Breakdown
| Category | Features |
| :--- | :--- |
| **Demographics** | `age`, `sex` (1: Female, 2: Male) |
| **Pre-existing Conditions** | `diabetes`, `copd`, `asthma`, `inmsupr`, `hypertension`, `cardiovascular`, `obesity`, `renal_chronic`, `tobacco`, `other_disease` |
| **Clinical Information** | `patient_type`, `icu`, `intubed`, `pneumonia`, `covid_res` |
| **Temporal Data** | `entry_date`, `date_symptoms`, `date_died` |

* **Target:** `is_dead` (Derived from `date_died`: 0 for recovery, 1 for mortality).

## ⚙️ Pipeline Overview
1. **Data Ingestion:** Automated fetching via `kagglehub` in `modules/data_loader.py`.
2. **Preprocessing:** Target engineering (`is_dead` creation), handling missing codes (97, 98, 99), and feature binning.
3. **Handling Imbalance:** Comparative analysis was performed using SMOTENC (over-sampling) and RandomUnderSampler (under-sampling) strategies to optimize model performance."
4. **Modeling:** Comparative analysis between Machine Learning and Deep Learning.
5. **Evaluation:** Comprehensive testing using Recall, F1-Score, and AUC-ROC.
6. **Explainability:** Global and local interpretation using **SHAP**.

## Tech Stack
### Modeling & Deep Learning
* **Deep Learning:** `Keras`, `TensorFlow`, `PyTorch`, `PyTorch-TabNet`
* **Gradient Boosting:** `XGBoost`, `LightGBM`
* **Explainability:** `SHAP` (Shapley Additive Explanations)
* **Machine Learning:** `Scikit-learn`, `SciPy`

### Data Science & Processing
* **Data Manipulation:** `Pandas`, `NumPy`
* **Imbalanced Data:** `Imbalanced-learn` (Applied **SMOTENN** and **RandomUnderSampler**)
* **Data Ingestion:** `Kagglehub` (Automated fetching)
* **Utilities:** `Tqdm`, `Joblib`

### Visualization
* **Static & Interactive:** `Matplotlib`, `Seaborn`
* 
## Project Structure
```text
├── modules/            # Fully decoupled pipeline modules
│   ├── config.py       # Configuration (Dataset URL, Feature mappings)
│   ├── data_loader.py  # Automated data acquisition via Kagglehub
│   ├── preprocessing.py# Data cleaning
│   ├── eda.py          # Distribution analysis & Correlation heatmaps
│   ├── models.py       # ML Pipeline (GBDTs, Training & Evaluation)
│   ├── deep_learning.py# Advanced DL models (ResNet, TabNet, Keras)
│   └── utils.py        
├── main.ipynb          # End-to-end execution notebook 
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
