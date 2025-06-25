import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, \
    precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for saving the model

# Set matplotlib to support Chinese
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set font for Chinese support
matplotlib.rcParams['axes.unicode_minus'] = False  # Solve the issue with minus sign display

# Configure font
matplotlib.rcParams.update({
    'font.size': 12,
    'font.sans-serif': ['DejaVu Sans'],  # Use a font that supports minus signs
    'axes.unicode_minus': False  # Solve the issue with minus sign display
})

# Example plot
plt.plot([-5, 0, 5], [-25, 0, 25])
plt.title('Correct Minus Sign Display')
plt.show()

# Data file paths
train_file_path = r'train_data.xlsx'
val_file_path = r'val_data.xlsx'
test_file_path = r'test_data.xlsx'


# Load data
def load_data(file_path):
    data_df = pd.read_excel(file_path)
    X = data_df.drop(columns=['area_num', 'groundtruth'])  # Exclude area_num and groundtruth columns, keeping feature columns
    y = data_df['groundtruth']  # Label column
    area_num = data_df['area_num']  # Unique identifier
    return X, y, area_num


X_train, y_train, area_train = load_data(train_file_path)
X_val, y_val, area_val = load_data(val_file_path)
X_test, y_test, area_test = load_data(test_file_path)

# Handle missing values (if any)
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())
X_test = X_test.fillna(X_test.mean())

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Simulate changes in training and validation accuracy (by adjusting C parameter)
C_values = [0.01, 0.1, 1, 10]
train_accuracies = []
val_accuracies = []

best_model = None
best_val_accuracy = 0
best_C = None

for C in C_values:
    svm_model = SVC(C=C, kernel='rbf', random_state=42)  # Use RBF kernel
    svm_model.fit(X_train_scaled, y_train)
    y_pred_train = svm_model.predict(X_train_scaled)
    y_pred_val = svm_model.predict(X_val_scaled)
    train_accuracies.append(accuracy_score(y_train, y_pred_train))
    val_accuracies.append(accuracy_score(y_val, y_pred_val))

    # Save the best model (based on validation accuracy)
    if val_accuracies[-1] > best_val_accuracy:
        best_val_accuracy = val_accuracies[-1]
        best_C = C
        best_model = svm_model

# Plot training and validation accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, label='Models Accuracy', marker='o', color='blue')
plt.plot(C_values, val_accuracies, label='Validation Accuracy', marker='o', color='green')
plt.xlabel('C (Regularization Parameter)')
plt.xscale('log')  # Use logarithmic scale to display C values
plt.ylabel('Accuracy')
plt.title('Models vs Validation Accuracy (SVM)')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation error curves
train_errors = [1 - acc for acc in train_accuracies]
val_errors = [1 - acc for acc in val_accuracies]

plt.figure(figsize=(10, 6))
plt.plot(C_values, train_errors, label='Models Error', marker='o', color='red')
plt.plot(C_values, val_errors, label='Validation Error', marker='o', color='orange')
plt.xlabel('C (Regularization Parameter)')
plt.xscale('log')  # Use logarithmic scale to display C values
plt.ylabel('Error Rate')
plt.title('Models vs Validation Loss (SVM)')
plt.legend()
plt.grid(True)
plt.show()

# SVM model (final model, C=best C)
svm_model = best_model

# Save the best model
model_save_path = r'D:\data\Self_Attention\Dataset_70\models\best_svm_model.joblib'
joblib.dump(svm_model, model_save_path)
print(f'Best model saved to {model_save_path}')

# Prediction
y_pred_train = svm_model.predict(X_train_scaled)
y_pred_val = svm_model.predict(X_val_scaled)
y_pred_test = svm_model.predict(X_test_scaled)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_val = accuracy_score(y_val, y_pred_val)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f'Models Accuracy: {accuracy_train * 100:.2f}%')
print(f'Validation Accuracy: {accuracy_val * 100:.2f}%')
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Confusion matrix
cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

# Replace class numbers with labels
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR', 5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}

# Normalize confusion matrix by rows
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_test_normalized,
    annot=True,  # Show values
    fmt='.2f',  # Two decimal places
    cmap='Blues',  # Color map
    xticklabels=[class_labels[i] for i in range(len(class_labels))],  # X-axis labels
    yticklabels=[class_labels[i] for i in range(len(class_labels))]  # Y-axis labels
)
plt.title('Confusion Matrix (Test) - Normalized')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Print classification report for the test set (precision, recall, and F1 score to 4 decimal places)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=1)

print("\nDetailed Classification Metrics (Test Set):")
print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
for i, label in enumerate(class_labels.values()):
    print(f"{label:<10}{precision[i]:<12.4f}{recall[i]:<12.4f}{f1[i]:<12.4f}{support[i]:<10}")

# Print macro-average and weighted average metrics
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

# Calculate Kappa coefficient
kappa_val = cohen_kappa_score(y_val, y_pred_val)
kappa_test = cohen_kappa_score(y_test, y_pred_test)

print(f"\nKappa Score (Validation): {kappa_val:.4f}")
print(f"Kappa Score (Test): {kappa_test:.4f}")
