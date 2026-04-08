import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight # Thêm thư viện tính trọng số
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

def evaluate_dl_model(X_train, y_train, X_test, y_test, preprocessor, undersampler, dataset_type="SMOTEd"):
    print(f"STARTING DEEP LEARNING TRAINING...\n" + "="*60)

    # 1. Tiền xử lý & Cân bằng dữ liệu
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if undersampler:
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_processed, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_processed, y_train

    # Ép kiểu Keras & TabNet
    if hasattr(X_train_resampled, "toarray"):
        X_train_resampled = X_train_resampled.toarray()
        X_test_processed = X_test_processed.toarray()

    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    X_test_processed = np.array(X_test_processed, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled).flatten()
    y_test = np.array(y_test).flatten()

    # TẠO TẬP VALIDATION CHUNG CHO CẢ 3 MODEL
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    # 2. Xây dựng mô hình Keras
    input_dim = X_train_resampled.shape[1]

    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid') # Lớp output cho Binary Classification
    ])

    # Sử dụng Adam optimizer và Binary Crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Huấn luyện (Training) với Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    print(f"Training on {X_train_resampled.shape[0]} resampled rows...")

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # 4. Dự đoán và Đánh giá
    y_pred_prob = model.predict(X_test_processed, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Tên mô hình
    name = "Deep Learning (Keras MLP)"

    # ==========================================
    # BIỂU DIỄN KẾT QUẢ THEO FORMAT CHUẨN
    # ==========================================
    print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
    print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))

    # Red: SMOTENC-ed data; Blue: original data
    cmap_color = 'Reds' if 'SMOTE' in dataset_type.upper() else 'Blues'

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
    plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("="*60)

    return model, history

def build_resnet_model(input_dim):
    """Xây dựng kiến trúc Tabular ResNet"""
    inputs = Input(shape=(input_dim,))

    # Lớp Dense ban đầu để ánh xạ dữ liệu lên không gian chiều cao hơn
    x = Dense(128)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # --- Residual Block 1 ---
    res_1 = x  # Lưu lại đầu vào của block
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    # Skip Connection: Cộng tín hiệu đầu vào (res_1) với đầu ra của block
    x = Add()([res_1, x])
    x = Activation('relu')(x)

    # --- Residual Block 2 ---
    res_2 = x
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    # Skip Connection
    x = Add()([res_2, x])
    x = Activation('relu')(x)

    # --- Output Layer ---
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def evaluate_resnet_model(X_train, y_train, X_test, y_test, preprocessor, undersampler, dataset_type="SMOTEd"):
    print(f"STARTING TABULAR RESNET TRAINING...\n" + "="*60)

    # 1. Tiền xử lý & Cân bằng dữ liệu (giống hệt pipeline ML)
    # 1. Tiền xử lý & Cân bằng dữ liệu
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if undersampler:
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_processed, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_processed, y_train

    # Ép kiểu Keras & TabNet
    if hasattr(X_train_resampled, "toarray"):
        X_train_resampled = X_train_resampled.toarray()
        X_test_processed = X_test_processed.toarray()

    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    X_test_processed = np.array(X_test_processed, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled).flatten()
    y_test = np.array(y_test).flatten()

    # TẠO TẬP VALIDATION CHUNG CHO CẢ 3 MODEL
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    # 2. Khởi tạo mô hình
    input_dim = X_train_resampled.shape[1]
    model = build_resnet_model(input_dim)

    # 3. Callbacks để tối ưu quá trình train
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Giảm learning rate nếu mô hình ngừng hội tụ để "dò" đáy loss function tinh tế hơn
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    print(f"Training on {X_train_resampled.shape[0]} resampled rows...")
    start_time = time.time()

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    elapsed_time = time.time() - start_time

    # 4. Dự đoán và xuất báo cáo
    y_pred_prob = model.predict(X_test_processed, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    name = "Tabular ResNet"

    # ==========================================
    # BIỂU DIỄN KẾT QUẢ THEO FORMAT CHUẨN
    # ==========================================
    print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
    print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    cmap_color = 'Reds' if dataset_type == 'SMOTEd' else 'Blues'
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
    plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("="*60)

    return model, history


def evaluate_tabnet_model(X_train, y_train, X_test, y_test, preprocessor, undersampler, dataset_type="SMOTEd"):
    print(f"STARTING TABNET TRAINING...\n" + "="*60)

    # 1. Tiền xử lý & Cân bằng dữ liệu
    # 1. Tiền xử lý & Cân bằng dữ liệu
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if undersampler:
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_processed, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_processed, y_train

    # Ép kiểu Keras & TabNet
    if hasattr(X_train_resampled, "toarray"):
        X_train_resampled = X_train_resampled.toarray()
        X_test_processed = X_test_processed.toarray()

    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    X_test_processed = np.array(X_test_processed, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled).flatten()
    y_test = np.array(y_test).flatten()

    # TẠO TẬP VALIDATION CHUNG CHO CẢ 3 MODEL
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    # 3. Cấu hình mô hình TabNet
    clf = TabNetClassifier(
        n_d=16, n_a=16, n_steps=4,
        gamma=1.5,
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=1
    )

    # 4. Huấn luyện (Training)
    print(f"Training on {X_tr.shape[0]} rows, validating on {X_val.shape[0]} rows...")
    start_time = time.time()

    clf.fit(
        X_train=X_tr, y_train=y_tr,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['logloss'],
        max_epochs=100,
        patience=10,
        batch_size=256,
        num_workers=0,
        drop_last=False
    )

    elapsed_time = time.time() - start_time

    # 5. Dự đoán và Đánh giá
    y_pred = clf.predict(X_test_processed)

    name = "TabNet"

    # ==========================================
    # BIỂU DIỄN KẾT QUẢ THEO FORMAT CHUẨN
    # ==========================================
    print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
    print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    cmap_color = 'Reds' if dataset_type == 'SMOTEd' else 'Blues'
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
    plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("="*60)

    feature_importances = clf.feature_importances_

    return clf, feature_importances