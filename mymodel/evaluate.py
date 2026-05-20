import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

# ======================
# CONFIGURATION
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset"
MODEL_PATH = "lung_cancer_model.keras"

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ======================
# LOAD TEST DATASET
# ======================
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_dataset.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# ======================
# NORMALIZATION
# ======================
normalization_layer = layers.Rescaling(1./255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# ======================
# GET TRUE LABELS
# ======================
y_true = []

for images, labels in test_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)

# ======================
# MODEL PREDICTIONS
# ======================
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# ======================
# BASIC METRICS
# ======================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n===== BASIC METRICS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ======================
# CLASSIFICATION REPORT
# ======================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=class_names))

# ======================
# SENSITIVITY & SPECIFICITY
# ======================
print("\n===== SENSITIVITY & SPECIFICITY =====")

for i in range(num_classes):

    TP = cm[i,i]
    FN = sum(cm[i,:]) - TP
    FP = sum(cm[:,i]) - TP
    TN = sum(cm) - (TP + FP + FN)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print(f"\nClass: {class_names[i]}")
    print("Sensitivity (Recall):", sensitivity)
    print("Specificity:", specificity)

# ======================
# ROC AUC SCORE
# ======================
y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes)

roc_auc = roc_auc_score(y_true_onehot, predictions, multi_class="ovr")

print("\nROC-AUC Score:", roc_auc)

# ======================
# ROC CURVE
# ======================
plt.figure(figsize=(7,6))

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(y_true_onehot[:,i], predictions[:,i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ======================
# PRECISION RECALL CURVE
# ======================
plt.figure(figsize=(7,6))

for i in range(num_classes):

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true_onehot[:,i], predictions[:,i]
    )

    pr_auc = auc(recall_vals, precision_vals)

    plt.plot(recall_vals, precision_vals,
             label=f"{class_names[i]} (AUC={pr_auc:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

print("\nEvaluation Complete!")