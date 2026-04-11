# COVID-19 Patient Precondition & Mortality Analysis 

A high-performance data science pipeline designed to analyze patient medical histories and predict COVID-19 mortality risk (`is_dead`) using Hybrid Machine Learning and Deep Learning architectures.
## Dataset Description
The dataset contains clinical records of patients, including their demographic data and pre-existing medical conditions.

### Metadata & Features
| Feature | Description | Values |
| :--- | :--- | :--- |
| **sex** | Gender of the patient | 1: Female, 2: Male |
| **patient_type** | Care setting | 1: Not hospitalized, 2: Hospitalized |
| **intubed** | Used ventilator support | 1: Yes, 2: No, 97-99: Missing |
| **pneumonia** | Air sacs inflammation | 1: Yes, 2: No, 97-99: Missing |
| **age** | Age of the patient | Numerical |
| **pregnancy** | Whether the patient is pregnant | 1: Yes, 2: No, 97-99: Missing |
| **pre-existing conditions** | Diabetes, COPD, Asthma, Inmsupr, Hypertension, Cardiovascular, Obesity, Renal Chronic | 1: Yes, 2: No, 97-99: Missing |
| **tobacco** | Is a tobacco user | 1: Yes, 2: No, 97-99: Missing |
| **contact_other_covid**| Contacted another COVID-19 patient | 1: Yes, 2: No, 97-99: Missing |
| **icu** | Admitted to Intensive Care Unit | 1: Yes, 2: No, 97-99: Missing |
| **covid_res** | COVID-19 test result | 1: Positive, 2: Negative, 3: Awaiting |
| **is_dead (Target)** | Generated from `date_died` | 0: Recovered, 1: Mortality |

## ⚙️ Pipeline Overview
1. **Data Ingestion:** Automated fetching via `kagglehub` in `modules/data_loader.py`.
2. **Preprocessing:** Target engineering (`is_dead` creation), handling missing codes (97, 98, 99), and feature binning.
3. **Handling Imbalance:** Comparative analysis was performed using SMOTENC (over-sampling) and RandomUnderSampler (under-sampling) strategies to optimize model performance."
4. **Modeling:** Comparative analysis between Machine Learning and Deep Learning.
5. **Evaluation:** Comprehensive testing using Recall, F1-Score, and AUC-ROC.
6. **Explainability:** Global and local interpretation using **SHAP**.

## Tech Stack
### Modeling & Deep Learning
* **Deep Learning:** `Keras`, `TensorFlow`, `PyTorch`, `PyTorch-TabNet` (ResNet-style DL architectures)
* **Gradient Boosting:** `XGBoost`, `LightGBM`, `CatBoost`
* **Explainability:** `SHAP` 
* **Machine Learning:** `Scikit-learn`, `SciPy`

### Data Science & Processing
* **Data Manipulation:** `Pandas`, `NumPy`
* **Imbalanced Data:** `Imbalanced-learn` (SMOTE, RandomUnderSampler)
* **Data Ingestion:** `Kagglehub` (Automated fetching)
* **Utilities:** `Tqdm`, `Joblib`

### Visualization
* **Static & Interactive:** `Matplotlib`, `Seaborn`, `Plotly`

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
