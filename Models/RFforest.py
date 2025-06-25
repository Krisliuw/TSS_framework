import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import joblib  # For model saving

# Set matplotlib to support Chinese
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Data file paths
train_file_path = r'train_data.xlsx'
val_file_path = r'val_data.xlsx'
test_file_path = r'test_data.xlsx'

# Load data
def load_data(file_path):
    data_df = pd.read_excel(file_path)
    X = data_df.drop(columns=['area_num', 'groundtruth'])
    y = data_df['groundtruth']
    area_num = data_df['area_num']
    return X, y, area_num

X_train, y_train, area_train = load_data(train_file_path)
X_val, y_val, area_val = load_data(val_file_path)
X_test, y_test, area_test = load_data(test_file_path)

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())
X_test = X_test.fillna(X_test.mean())

# Data standardization (optional)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# Hyperparameter tuning: Find the best model and record the validation accuracy
n_estimators_list = range(10, 151, 10)
train_accuracies = []
val_accuracies = []

best_model = None
best_val_accuracy = 0
best_n_estimators = None

for n_estimators in n_estimators_list:
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=42,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(X_train, y_train)
    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model = rf_model
        best_n_estimators = n_estimators

# Save the best model
model_save_path = r'D:\data\Self_Attention\Dataset_70\models\best_rf_model.joblib'
joblib.dump(best_model, model_save_path)
print(f"\n✅ Best RF model has been saved to: {model_save_path} (n_estimators={best_n_estimators})")

# Use the best model for subsequent operations
rf_model = best_model

# Plot accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, train_accuracies, label='Models Accuracy', marker='o', color='blue')
plt.plot(n_estimators_list, val_accuracies, label='Validation Accuracy', marker='o', color='green')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Models vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot error curve
train_errors = [1 - acc for acc in train_accuracies]
val_errors = [1 - acc for acc in val_accuracies]
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, train_errors, label='Models Error', marker='o', color='red')
plt.plot(n_estimators_list, val_errors, label='Validation Error', marker='o', color='orange')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Error Rate')
plt.title('Models vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Prediction
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Accuracy output
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_val = accuracy_score(y_val, y_pred_val)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Models Accuracy: {accuracy_train * 100:.2f}%')
print(f'Validation Accuracy: {accuracy_val * 100:.2f}%')
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Confusion matrix
cm_test = confusion_matrix(y_test, y_pred_test)

class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR', 5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}

# Normalized confusion matrix
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[class_labels[i] for i in range(len(class_labels))],
            yticklabels=[class_labels[i] for i in range(len(class_labels))])
plt.title('Confusion Matrix (Test) - Normalized')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_pred_val, zero_division=1))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, zero_division=1))

# Kappa coefficient
kappa_val = cohen_kappa_score(y_val, y_pred_val)
kappa_test = cohen_kappa_score(y_test, y_pred_test)
print(f"\nKappa Score (Validation): {kappa_val:.4f}")
print(f"Kappa Score (Test): {kappa_test:.4f}")

# Feature importance
feature_importances = rf_model.feature_importances_
features = X_train.columns
sorted_indices = np.argsort(feature_importances)[::-1]
features_sorted = features[sorted_indices][:20]
importances_sorted = feature_importances[sorted_indices][:20]

plt.figure(figsize=(12, 8))
plt.barh(features_sorted, importances_sorted, color='skyblue')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Importances', fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
plt.show()

# Per-class performance metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=1)
print("\nDetailed Classification Metrics (Test Set):")
print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
for i, label in enumerate(class_labels.values()):
    print(f"{label:<10}{precision[i]:<12.4f}{recall[i]:<12.4f}{f1[i]:<12.4f}{support[i]:<10}")

# Averaged metrics
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)

print("\nAveraged Metrics (Test Set):")
print(f"{'Metric':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
print(f"{'Macro Avg':<15}{macro_precision:<12.4f}{macro_recall:<12.4f}{macro_f1:<12.4f}")
print(f"{'Weighted Avg':<15}{weighted_precision:<12.4f}{weighted_recall:<12.4f}{weighted_f1:<12.4f}")
