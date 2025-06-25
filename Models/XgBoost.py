import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Set matplotlib to support Chinese
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set font for Chinese support
matplotlib.rcParams['axes.unicode_minus'] = False  # Solve the issue with minus sign display

# Data file paths
# train_file_path = r'train_data.xlsx'
# val_file_path = r'val_data.xlsx'
# test_file_path = r'test_data.xlsx'
train_file_path = r'train_fused_features.xlsx'
val_file_path = r'val_fused_features.xlsx'
test_file_path = r'test_fused_features.xlsx'
# train_file_path = r'train_data_pca.xlsx'
# val_file_path = r'val_data_pca.xlsx'
# test_file_path = r'test_data_pca.xlsx'
# train_file_path = r'train_umap.xlsx'
# val_file_path = r'D:val_umap.xlsx'
# test_file_path = r'test_umap.xlsx'
# train_file_path = r'train_fused_features.xlsx'
# val_file_path = r'val_fused_features.xlsx'
# test_file_path = r'test_fused_features.xlsx'
# train_file_path = r'train_data_lda9.xlsx'
# val_file_path = r'val_data_lda9.xlsx'
# test_file_path = r'test_data_lda9.xlsx'

# Load data
def load_data(file_path):
    data_df = pd.read_excel(file_path)
    X = data_df.drop(columns=['area_num', 'groundtruth'])  # Remove area_num and groundtruth columns, keep feature columns
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

# Over-sample class 7
smote = SMOTE(sampling_strategy={7: 30}, random_state=42)  # Over-sample class 7, aiming to have 30 samples
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Set model parameters
params = {
    'objective': 'multi:softmax',  # Multi-class task
    'eval_metric': 'merror',       # Use multi-class error rate (merror)
    'num_class': len(np.unique(y_train)),  # Number of classes
    'max_depth': 5,                # Max depth of trees
    'learning_rate': 0.01,         # Learning rate
    'gamma': 0,                    # Parameter used to control whether pruning happens
    'subsample': 0.7,              # Proportion of training data used for each tree
    'colsample_bytree': 0.7,       # Proportion of features used for each tree
    'lambda': 2.5,                 # L2 regularization (Ridge)
    'alpha': 1.5,                  # L1 regularization (Lasso)
    'min_child_weight': 5          # Minimum number of samples required at each leaf node
}

# Convert to XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Cross-validation
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=250,
    nfold=10,
    early_stopping_rounds=10,
    verbose_eval=True,
    show_stdv=True
)

# Get best number of boosting rounds
best_num_round = cv_results.shape[0] - 1
print(f'Best number of boosting rounds: {best_num_round}')

# Models XGBoost model on the entire training set
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_num_round
)

# Save the trained model
model_save_path = r"model\best_xgb_TBFFmodel.model"
bst.save_model(model_save_path)
print(f"Best model saved to {model_save_path}")

# If you need to load the saved model
if os.path.exists(model_save_path):
    loaded_bst = xgb.Booster()
    loaded_bst.load_model(model_save_path)
    print("Best model loaded")
else:
    print("The model file at the specified path does not exist. Please check the path!")

# Prediction
y_pred_train = bst.predict(dtrain)
y_pred_val = bst.predict(dval)
y_pred_test = bst.predict(dtest)

# Calculate accuracy
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)
accuracy_val = accuracy_score(y_val, y_pred_val)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f'Models Accuracy: {accuracy_train * 100:.2f}%')
print(f'Validation Accuracy: {accuracy_val * 100:.2f}%')
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Plot accuracy and error rate curves
validation_accuracy = 1 - cv_results['test-merror-mean']
train_accuracy = 1 - cv_results['train-merror-mean']
train_error_rate = cv_results['train-merror-mean']
test_error_rate = cv_results['test-merror-mean']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Models Accuracy (CV)', color='blue')
plt.plot(validation_accuracy, label='Validation Accuracy (CV)', color='green')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('XGBoost Cross-Validation Accuracy per Iteration')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_error_rate, label='Models Error Rate (CV)', color='blue')
plt.plot(test_error_rate, label='Test Error Rate (CV)', color='red')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.title('XGBoost Cross-Validation Error Rate per Iteration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrix
cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

# Replace numeric labels with string labels
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR', 5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}

# Normalize confusion matrix by row
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_test_normalized,
    annot=True,               # Show values
    fmt='.2f',                # Two decimal places
    cmap='Blues',             # Color map
    xticklabels=[class_labels[i] for i in range(len(class_labels))],  # X-axis labels
    yticklabels=[class_labels[i] for i in range(len(class_labels))]   # Y-axis labels
)
plt.title('Confusion Matrix (Test) - Normalized')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Print classification report for the test set (precision, recall, and F1 scores to 4 decimal places)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=1)

print("\nDetailed Classification Metrics (Test Set):")
print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
for i, label in enumerate(class_labels.values()):
    print(f"{label:<10}{precision[i]:<12.4f}{recall[i]:<12.4f}{f1[i]:<12.4f}{support[i]:<10}")

# Print macro average and weighted average
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

# Kappa coefficient
kappa_val = cohen_kappa_score(y_val, y_pred_val)
kappa_test = cohen_kappa_score(y_test, y_pred_test)
print(f'Kappa Score (Validation): {kappa_val:.4f}')
print(f'Kappa Score (Test): {kappa_test:.4f}')
