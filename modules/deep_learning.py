# import time
# import numpy as np
# import pandas as pd
# from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight # Thêm thư viện tính trọng số
# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import torch
# from sklearn.model_selection import train_test_split

# def evaluate_dl_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd"):
#     print(f"STARTING DEEP LEARNING TRAINING...\n" + "="*60)
#     if undersampler:
#         X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
#     else:
#         X_train_resampled, y_train_resampled = X_train, y_train

#     if hasattr(X_train_resampled, "toarray"):
#         X_train_resampled = X_train_resampled.toarray()
#         X_test = X_test.toarray()

#     X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
#     X_test = np.array(X_test, dtype=np.float32)
#     y_train_resampled = np.array(y_train_resampled).flatten()
#     y_test = np.array(y_test).flatten()

#     from sklearn.model_selection import train_test_split
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train_resampled, y_train_resampled,
#         test_size=0.2, random_state=42, stratify=y_train_resampled
#     )

#     input_dim = X_train_resampled.shape[1]

#     model = Sequential([
#         Dense(64, activation='relu', input_dim=input_dim),
#         BatchNormalization(),
#         Dropout(0.3),

#         Dense(32, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),

#         Dense(1, activation='sigmoid')
#     ])

#     # Sử dụng Adam optimizer và Binary Crossentropy loss
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

#     print(f"Training on {X_train_resampled.shape[0]} resampled rows...")

#     history = model.fit(
#         X_tr, y_tr,
#         validation_data=(X_val, y_val),
#         epochs=100,
#         batch_size=256,
#         callbacks=[early_stop, reduce_lr],
#         verbose=0
#     )

#     y_pred_prob = model.predict(X_test, verbose=0)
#     y_pred = (y_pred_prob > 0.5).astype(int).flatten()

#     name = "Deep Learning (Keras MLP)"

#     print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
#     print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
#     print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
#     print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 4))

#     cmap_color = 'Reds' if 'SMOTE' in dataset_type.upper() else 'Blues'

#     sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
#     plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
#     plt.ylabel('Actual Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
#     print("="*60)

#     return model, history

# def build_resnet_model(input_dim):
#     """Xây dựng kiến trúc Tabular ResNet"""
#     inputs = Input(shape=(input_dim,))

#     x = Dense(128)(inputs)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     # --- Residual Block 1 ---
#     res_1 = x  
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.3)(x)

#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     # Skip Connection: Cộng tín hiệu đầu vào (res_1) với đầu ra của block
#     x = Add()([res_1, x])
#     x = Activation('relu')(x)

#     # --- Residual Block 2 ---
#     res_2 = x
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.3)(x)

#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     # Skip Connection
#     x = Add()([res_2, x])
#     x = Activation('relu')(x)

#     # --- Output Layer ---
#     x = Dropout(0.2)(x)
#     outputs = Dense(1, activation='sigmoid')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     return model

# def evaluate_resnet_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd"):
#     print(f"STARTING TABULAR RESNET TRAINING...\n" + "="*60)
#     if undersampler:
#         X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
#     else:
#         X_train_resampled, y_train_resampled = X_train, y_train

#     if hasattr(X_train_resampled, "toarray"):
#         X_train_resampled = X_train_resampled.toarray()
#         X_test = X_test.toarray()

#     X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
#     X_test = np.array(X_test, dtype=np.float32)
#     y_train_resampled = np.array(y_train_resampled).flatten()
#     y_test = np.array(y_test).flatten()

#     from sklearn.model_selection import train_test_split
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train_resampled, y_train_resampled,
#         test_size=0.2, random_state=42, stratify=y_train_resampled
#     )

#     input_dim = X_train_resampled.shape[1]
#     model = build_resnet_model(input_dim)

#     # 3. Callbacks để tối ưu quá trình train
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     # Giảm learning rate nếu mô hình ngừng hội tụ để "dò" đáy loss function tinh tế hơn
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

#     print(f"Training on {X_train_resampled.shape[0]} resampled rows...")
#     start_time = time.time()

#     history = model.fit(
#         X_tr, y_tr,
#         validation_data=(X_val, y_val),
#         epochs=100,
#         batch_size=256,
#         callbacks=[early_stop, reduce_lr],
#         verbose=1
#     )

#     elapsed_time = time.time() - start_time

#     y_pred_prob = model.predict(X_test, verbose=0)
#     y_pred = (y_pred_prob > 0.5).astype(int).flatten()

#     name = "Tabular ResNet"

#     # ==========================================
#     # BIỂU DIỄN KẾT QUẢ THEO FORMAT CHUẨN
#     # ==========================================
#     print(f"\n--- {name.upper()} ({dataset_type.upper()} DATA) RESULTS ---")
#     print(f"Recall (Deceased): {recall_score(y_test, y_pred):.4f}")
#     print(f"F1 Score (Deceased): {f1_score(y_test, y_pred):.4f}")
#     print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 4))
#     cmap_color = 'Reds' if dataset_type == 'SMOTEd' else 'Blues'
#     sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color)
#     plt.title(f'Confusion Matrix - {name} ({dataset_type} Data)')
#     plt.ylabel('Actual Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
#     print("="*60)

#     return model, history

import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
from sklearn.model_selection import train_test_split
import keras_tuner as kt # Thêm thư viện KerasTuner

# ==========================================
# 1. BUILDER FUNCTIONS CHO KERAS TUNER
# ==========================================
def build_tuned_mlp(hp, input_dim):
    """Hàm xây dựng Keras MLP hỗ trợ Tune Neuron"""
    model = Sequential()
    
    # Tune lớp ẩn thứ nhất (thử nghiệm từ 32 đến 256 neuron)
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(Dense(hp_units_1, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Tune lớp ẩn thứ hai (thử nghiệm từ 16 đến 128 neuron)
    hp_units_2 = hp.Int('units_2', min_value=16, max_value=128, step=16)
    model.add(Dense(hp_units_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    # Tune Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

def build_tuned_resnet(hp, input_dim):
    """Hàm xây dựng Tabular ResNet hỗ trợ Tune Neuron"""
    inputs = Input(shape=(input_dim,))

    # Tune số lượng neuron chung cho các khối ResNet để giữ tính cân bằng
    hp_units = hp.Int('resnet_units', min_value=64, max_value=256, step=64)

    x = Dense(hp_units)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # --- Residual Block 1 ---
    res_1 = x  
    x = Dense(hp_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(hp_units)(x)
    x = BatchNormalization()(x)
    x = Add()([res_1, x])
    x = Activation('relu')(x)

    # --- Residual Block 2 ---
    res_2 = x
    x = Dense(hp_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(hp_units)(x)
    x = BatchNormalization()(x)
    x = Add()([res_2, x])
    x = Activation('relu')(x)

    # --- Output Layer ---
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    # Tune Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model


# ==========================================
# 2. HÀM ĐÁNH GIÁ (CÓ TÍCH HỢP AUTO TUNING)
# ==========================================
def evaluate_dl_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd", max_trials=5):
    print(f"STARTING DEEP LEARNING (MLP) AUTO-TUNING & TRAINING...\n" + "="*60)
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

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    input_dim = X_train_resampled.shape[1]

    # Cấu hình KerasTuner
    tuner = kt.RandomSearch(
        lambda hp: build_tuned_mlp(hp, input_dim),
        objective='val_loss',
        max_trials=max_trials, # Số lượng cấu hình khác nhau sẽ thử nghiệm
        executions_per_trial=1,
        directory='kt_tuning_dir',
        project_name=f'mlp_{dataset_type}_{int(time.time())}'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    print(f"BƯỚC 1: Tìm kiếm siêu tham số tối ưu (Thử nghiệm {max_trials} cấu hình)...")
    # Search nhanh trên 30 epoch để tiết kiệm thời gian
    tuner.search(X_tr, y_tr, epochs=30, validation_data=(X_val, y_val), batch_size=256, callbacks=[early_stop], verbose=0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n=> Đã tìm thấy cấu hình TỐT NHẤT:")
    print(f" - Lớp ẩn 1: {best_hps.get('units_1')} neurons")
    print(f" - Lớp ẩn 2: {best_hps.get('units_2')} neurons")
    print(f" - Learning Rate: {best_hps.get('learning_rate')}")

    print(f"\nBƯỚC 2: Huấn luyện mô hình chính thức ({X_train_resampled.shape[0]} samples)...")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1 # Đổi thành 1 để theo dõi quá trình train chính thức
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    name = "Deep Learning (Keras MLP - Tuned)"

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


def evaluate_resnet_model(X_train, y_train, X_test, y_test, undersampler, dataset_type="SMOTEd", max_trials=5):
    print(f"STARTING TABULAR RESNET AUTO-TUNING & TRAINING...\n" + "="*60)
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

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    input_dim = X_train_resampled.shape[1]

    tuner = kt.RandomSearch(
        lambda hp: build_tuned_resnet(hp, input_dim),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='kt_tuning_dir',
        project_name=f'resnet_{dataset_type}_{int(time.time())}'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    print(f"BƯỚC 1: Tìm kiếm siêu tham số tối ưu (Thử nghiệm {max_trials} cấu hình)...")
    start_time = time.time()
    tuner.search(X_tr, y_tr, epochs=30, validation_data=(X_val, y_val), batch_size=256, callbacks=[early_stop], verbose=0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n=> Đã tìm thấy cấu hình TỐT NHẤT:")
    print(f" - ResNet Base Units: {best_hps.get('resnet_units')} neurons")
    print(f" - Learning Rate: {best_hps.get('learning_rate')}")

    print(f"\nBƯỚC 2: Huấn luyện mô hình chính thức ({X_train_resampled.shape[0]} samples)...")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTổng thời gian Tune + Train: {elapsed_time:.2f} giây")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    name = "Tabular ResNet (Tuned)"

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