from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def generate_balanced_datasets(X_train_raw, y_train, preprocessor, cat_indices):
    """
    Tạo ra các phiên bản dữ liệu đã cân bằng (RUS, SMOTE).
    Trả về một Dictionary chứa các tập dữ liệu đã được xử lý (Processed).
    """
    datasets = {}
    
    # 0. Prepare Base Processed Data
    X_train_proc = preprocessor.transform(X_train_raw)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    # --- 1. RANDOM UNDERSAMPLING (RUS) ---
    print("\n--- BALANCING DATA WITH RANDOM UNDERSAMPLER ---")
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    start_time = time.time()
    X_rus, y_rus = rus.fit_resample(X_train_proc, y_train)
    print(f"Class distribution AFTER RUS: {Counter(y_rus)}")
    print(f"RUS took {time.time() - start_time:.2f} seconds.")
    datasets['Original + RUS'] = (X_rus, y_rus)

    # --- 2. SMOTE-NC ---
    print("\n--- BALANCING DATA WITH SMOTE-NC ---")
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    start_time = time.time()
    X_smote, y_smote = smotenc.fit_resample(X_train_raw, y_train)
    X_smote_proc = preprocessor.transform(X_smote)
    if hasattr(X_smote_proc, "toarray"): X_smote_proc = X_smote_proc.toarray()
    print(f"Class distribution AFTER SMOTE:  {Counter(y_smote)}")
    print(f"SMOTE-NC took {time.time() - start_time:.2f} seconds.")
    datasets['SMOTE-NC'] = (X_smote_proc, y_smote)

    return datasets



def evaluate_model(models_dict, X_train_prepared, y_train_prepared, X_test_processed, y_test, dataset_type=""):
  print(f"STARTING TRAINING ON {X_train_prepared.shape[0]} ROWS OF {dataset_type} DATA...\n" + "="*60)

  results_list = []

  for name, model in models_dict.items():
      # Timing and Fit
      start_time = time.time()
      model.fit(X_train_prepared, y_train_prepared)
      elapsed_time = time.time() - start_time

      # Predict on Test set
      y_pred = model.predict(X_test_processed)

      report = classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)'], output_dict=True)
      accuracy = accuracy_score(y_test, y_pred)

      # Print evaluation report
      print(f"MODEL: {name}")
      print(f"Training Time: {elapsed_time:.2f}s")
      print(f"Accuracy: {accuracy*100:.2f}%")
      print("\n DETAILED REPORT:")
      print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

      # Plot Confusion Matrix
      cm = confusion_matrix(y_test, y_pred)
      plt.figure(figsize=(6, 4))
      # Use different colors to easily tell SMOTEd vs Un-SMOTEd apart
      cmap_color = 'Reds' if 'SMOTEd' in dataset_type else 'Blues'
      sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
      plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
      plt.ylabel('Actual Label')
      plt.xlabel('Predicted Label')
      plt.show()
      print("-" * 60)

      # Store results for export
      results_list.append({
          'Model': name,
          'Accuracy': accuracy,
          'Training_Time': elapsed_time,
          'Precision_Deceased': report['Deceased (1)']['precision'],
          'Recall_Deceased': report['Deceased (1)']['recall'],
          'F1_Deceased': report['Deceased (1)']['f1-score']
      })
  return pd.DataFrame(results_list)

def train_and_evaluate_pipeline(X_train_bal, y_train_bal, X_test_proc, y_test, dataset_name):
    """
    Thực thi chuỗi: Build Models -> Build Ensemble -> Evaluate cho một tập dữ liệu.
    """
    print(f"\n{'='*70}")
    print(f" INITIATING PIPELINE FOR: {dataset_name}")
    print(f"{'='*70}")

    base_individual_models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_depth=10),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, learning_rate=0.1, max_depth=5),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, num_leaves=31, learning_rate=0.05)
    }

    # 2. Xây dựng các mô hình kết hợp (Ensemble Models)
    base_models_for_ensemble = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('j48_exact', DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, class_weight='balanced', random_state=42)),
        ('reptree_approx', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
    ]

    meta_learner = base_individual_models['Logistic Regression']

    ensembles = {
        'Voting Classifier': VotingClassifier(
            estimators=base_models_for_ensemble, 
            voting='soft', 
            n_jobs=-1
        ),
        'Stacking Classifier': StackingClassifier(
            estimators=base_models_for_ensemble, 
            final_estimator=meta_learner, 
            cv=5, 
            n_jobs=-1
        )
    }

    # 3. Gộp toàn bộ mô hình (Đơn lẻ + Kết hợp)
    all_models = {**base_individual_models, **ensembles}

    # 4. Đánh giá tất cả
    print(f"\n>>> Evaluating on {dataset_name} <<<")
    results = evaluate_model(
        all_models, 
        X_train_bal, 
        y_train_bal, 
        X_test_proc, 
        y_test, 
        dataset_type=dataset_name
    )
    
    return results, all_models