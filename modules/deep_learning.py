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

def evaluate_dl_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd"):
    print(f"STARTING DEEP LEARNING TRAINING...\n" + "="*60)
    if undersampler:
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    if hasattr(X_train_resampled, "toarray"):
        X_train_resampled = X_train_resampled.toarray()
        X_test = X_test.toarray()

    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled).flatten()
    y_test = np.array(y_test).flatten()

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    input_dim = X_train_resampled.shape[1]

    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    # Sử dụng Adam optimizer và Binary Crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    name = "Deep Learning (Keras MLP)"

    print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
    print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))

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

    x = Dense(128)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # --- Residual Block 1 ---
    res_1 = x  
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

def evaluate_resnet_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd"):
    print(f"STARTING TABULAR RESNET TRAINING...\n" + "="*60)
    if undersampler:
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    if hasattr(X_train_resampled, "toarray"):
        X_train_resampled = X_train_resampled.toarray()
        X_test = X_test.toarray()

    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled).flatten()
    y_test = np.array(y_test).flatten()

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

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

    y_pred_prob = model.predict(X_test, verbose=0)
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
