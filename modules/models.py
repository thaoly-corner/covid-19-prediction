from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
import time
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def generate_balanced_datasets(X_train_raw, y_train, preprocessor, cat_indices):
    """
    Tạo ra các phiên bản dữ liệu đã cân bằng (RUS, SMOTE 1:1, SMOTE 1:5).
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

    # --- 2. SMOTE-NC (1:1 Ratio) ---
    print("\n--- BALANCING DATA WITH SMOTE-NC (1:1) ---")
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    start_time = time.time()
    X_smote, y_smote = smotenc.fit_resample(X_train_raw, y_train)
    X_smote_proc = preprocessor.transform(X_smote)
    if hasattr(X_smote_proc, "toarray"): X_smote_proc = X_smote_proc.toarray()
    print(f"Class distribution AFTER SMOTE:  {Counter(y_smote)}")
    print(f"SMOTE-NC took {time.time() - start_time:.2f} seconds.")
    datasets['SMOTE-NC (1:1 Ratio)'] = (X_smote_proc, y_smote)

    # --- 3. SMOTE-NC (1:5 Ratio) ---
    print("\n--- BALANCING DATA WITH SMOTE-NC (1:5) ---")
    smotenc_15 = SMOTENC(categorical_features=cat_indices, random_state=42, sampling_strategy=0.2)
    start_time = time.time()
    X_smote_15, y_smote_15 = smotenc_15.fit_resample(X_train_raw, y_train)
    X_smote_15_proc = preprocessor.transform(X_smote_15)
    if hasattr(X_smote_15_proc, "toarray"): X_smote_15_proc = X_smote_15_proc.toarray()
    print(f"Class distribution AFTER SMOTE (1:5):  {Counter(y_smote_15)}")
    print(f"SMOTE-NC (1:5) took {time.time() - start_time:.2f} seconds.")
    datasets['SMOTE-NC (1:5 Ratio)'] = (X_smote_15_proc, y_smote_15)

    return datasets

def get_model_grids():
    """
    Khai báo và trả về danh sách các mô hình cùng không gian siêu tham số (Hyperparameter Grid) của chúng.
    """
    return {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, solver='liblinear'),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            'params': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'params': {
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [5, 10, -1],
                'n_estimators': [100, 200]
            }
        }
    }


def tune_base_models(X_train, y_train, n_iter=15, cv=3):
    """
    Hàm tổng điều phối: Gọi danh sách mô hình và lần lượt tối ưu hóa từng cái.
    """
    tuned_models = {}
    param_grids = get_model_grids()

    # Chỉ định đích ngắm là Recall cho nhãn 1 (Deceased/Đột quỵ)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    for name, config in param_grids.items():
        print(f"Đang tìm tham số tối ưu cho {name}...")
        start_time = time.time()

        # Thiết lập RandomizedSearchCV
        # n_iter=10 nghĩa là sẽ bốc ngẫu nhiên 10 tổ hợp để thử (tăng lên 20-30 nếu có thời gian)
        # cv=3 nghĩa là kiểm định chéo 3-fold để chống overfitting
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=10,
            scoring=recall_scorer,
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        # Huấn luyện để tìm best params
        search.fit(X_train, y_train)

        # Lưu lại mô hình đã được gán bộ tham số tốt nhất
        tuned_models[name] = search.best_estimator_

        print(f"   Bộ tham số tốt nhất: {search.best_params_}\n")

    return tuned_models

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
    Thực thi chuỗi: Tune Model -> Build Ensemble -> Evaluate cho một tập dữ liệu.
    """
    print(f"\n{'='*70}")
    print(f" INITIATING PIPELINE FOR: {dataset_name}")
    print(f"{'='*70}")

    # 1. Tối ưu hóa các mô hình cơ bản (Base Models)
    tuned_individual_models = tune_base_models(X_train_bal, y_train_bal)

    # 2. Xây dựng các mô hình kết hợp (Ensemble Models)
    tuned_base_models_for_ensemble = [
        ('rf', tuned_individual_models['Random Forest']),
        ('xgb', tuned_individual_models['XGBoost']),
        ('lgbm', tuned_individual_models['LightGBM'])
    ]

    tuned_meta_learner = LogisticRegression(class_weight='balanced', random_state=42, C=0.1)

    tuned_ensembles = {
        'Tuned Voting Classifier': VotingClassifier(
            estimators=tuned_base_models_for_ensemble, 
            voting='soft', 
            n_jobs=-1
        ),
        'Tuned Stacking Classifier': StackingClassifier(
            estimators=tuned_base_models_for_ensemble, 
            final_estimator=tuned_meta_learner, 
            cv=5, 
            n_jobs=-1
        )
    }

    # 3. Gộp toàn bộ mô hình (Đơn lẻ + Kết hợp)
    all_models = {**tuned_individual_models, **tuned_ensembles}

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
    
    return results