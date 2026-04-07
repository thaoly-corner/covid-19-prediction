"""Configuration constants for the Covid-19 project."""

CATEGORICAL_FEATURES = [
    'sex', 'patient_type', 'intubed', 'pneumonia', 'pregnancy', 
    'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 
    'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 
    'tobacco', 'contact_other_covid', 'covid_res', 'icu'
]

TARGET_COLUMN = 'is_dead'

NUMERICAL_FEATURES = ['age']
ORDINAL_FEATURES = ['delay_ordinal']

DATE_COLUMNS = ['entered_date', 'date_symptoms', 'date_died']

# Kaggle dataset information
DATASET_URL = "tanmoyx/covid19-patient-precondition-dataset"
DATA_PATH = "covid.csv"