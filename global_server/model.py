import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_model(input_shape=(15,)):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),  # Dropout added to reduce overfitting
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_local_model(data, labels, model):
    model.fit(data, labels, epochs=10, batch_size=16, verbose=1)
    return model.get_weights()

def aggregate_weights(weights_list):
    new_weights = list()
    for weights in zip(*weights_list):
        new_weights.append(np.median(np.array(weights), axis=0))
    return new_weights

def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob > 0.5).astype("int32")
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_pred_prob)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }
