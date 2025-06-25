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
import matplotlib.ticker as ticker

# Set matplotlib font to support Chinese
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set font that supports Chinese
matplotlib.rcParams['axes.unicode_minus'] = False  # Solve the issue of displaying negative signs

# Data file paths
train_file_path = r'rain_data.xlsx'
val_file_path = r'val_data.xlsx'
test_file_path = r'test_data.xlsx'
# train_file_path = r'train_fused_features.xlsx'
# val_file_path = r'D:val_fused_features.xlsx'
# test_file_path = r'test_fused_features.xlsx'
# train_file_path = r'train_data_pca.xlsx'
# val_file_path = r'val_data_pca.xlsx'
# test_file_path = r'test_data_pca.xlsx'
# train_file_path = r'train_umap.xlsx'
# val_file_path = r'val_umap.xlsx'
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
    X = data_df.drop(columns=['area_num', 'groundtruth'])  # Remove 'area_num' and 'groundtruth' columns, keep feature columns
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

# Perform oversampling on class 7
smote = SMOTE(sampling_strategy={7: 30}, random_state=42)  # Perform oversampling on class 7 to target 30 samples
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Set model parameters
params = {
    'objective': 'multi:softmax',  # Multi-class task
    'eval_metric': 'merror',      # Use multi-class error rate (merror)
    'num_class': len(np.unique(y_train)),  # Number of classes
    'max_depth': 5,               # Maximum depth of trees
    'learning_rate': 0.01,        # Learning rate
    'gamma': 0,                   # Parameter to control whether to do pruning
    'subsample': 0.7,             # Random sampling ratio of the training set
    'colsample_bytree': 0.7,      # Randomly select features for each tree
    'lambda': 2.5,                # L2 regularization (Ridge)
    'alpha': 1.5,                 # L1 regularization (Lasso)
    'min_child_weight': 5         # Minimum number of samples in each leaf node
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

# Get best number of rounds
best_num_round = cv_results.shape[0] - 1
print(f'Best training round: {best_num_round}')

# Models XGBoost model on the entire training set
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_num_round
)
# Compute feature importance (based on gain)
importance_dict = bst.get_score(importance_type='gain')  # You can also use 'weight', 'cover', 'total_gain'

# Convert importance_dict to DataFrame and normalize
importance_df = pd.DataFrame(
    list(importance_dict.items()),
    columns=['Feature', 'Importance']
)

# Fill in missing features (XGBoost sometimes only shows used features)
all_features = X_train.columns.tolist()
for f in all_features:
    if f not in importance_df['Feature'].values:
        importance_df = pd.concat([
            importance_df,
            pd.DataFrame([[f, 0]], columns=['Feature', 'Importance'])
        ], ignore_index=True)

# Normalize (divide by the total sum)
importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

# Sort and print
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nNormalized feature importance (based on gain):")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']:<30} {row['Importance']:.4f}")

# Save the trained model
model_save_path = r"models\best_xgb_YUANdata.model"
bst.save_model(model_save_path)
print(f"Best model saved to {model_save_path}")

# If you need to load the saved model
if os.path.exists(model_save_path):
    loaded_bst = xgb.Booster()
    loaded_bst.load_model(model_save_path)
    print("Best model loaded")
else:
    print("The model file does not exist at the specified path. Please check if the path is correct!")

# Prediction
y_pred_train = bst.predict(dtrain)
y_pred_val = bst.predict(dval)
y_pred_test = bst.predict(dtest)

# Compute accuracy
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

# Confusion Matrix
cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

# Replace class numbers with letter labels
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR', 5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}

# Normalize confusion matrix by row
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_test_normalized,
    annot=True,               # Display numbers
    fmt='.2f',                # Two decimal places
    cmap='Blues',             # Color map
    xticklabels=[class_labels[i] for i in range(len(class_labels))],  # X-axis letter labels
    yticklabels=[class_labels[i] for i in range(len(class_labels))]   # Y-axis letter labels
)
plt.title('Confusion Matrix (Test) - Normalized')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Visualization (showing the importance of all features)
plt.figure(figsize=(10, max(6, 0.3 * len(importance_df))))  # Automatically adjust the image height to prevent features from being cut off
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("All Features by Normalized Importance (Gain)")
plt.tight_layout()
plt.show()

# Specify three types of features
glcm_features = [f'rs_{i}' for i in range(1, 25)]
vari_features = [
    'rs_StdDev', 'rs_Min', 'rs_Max', 'rs_Mean', 'rs_Range', 'rs_Normalized Range',
    'rs_Range-to-StdDev Ratio', 'rs_Mean-to-StdDev Ratio', 'rs_Midpoint',
    'rs_Relative StdDev', 'rs_Range-to-Mean Offset', 'rs_Skewness Proxy',
    'rs_Range Ratio', 'rs_Normalized StdDev', 'rs_Normalized Offset',
    'rs_Concentration-to-Dispersion Ratio', 'rs_Range-Squared-to-StdDev Ratio',
    'rs_Comprehensive Index'
]
tfidf_features = ['se_public', 'se_commerce', 'se_resident', 'se_industry', 'se_greenspace']

# Create a function to categorize features (using new category names)
def get_feature_category(name):
    if name in glcm_features:
        return 'Texture'
    elif name in vari_features:
        return 'Spectral'
    elif name in tfidf_features:
        return 'Social semantic'
    else:
        return 'Other'

# Apply categorization
importance_df['Category'] = importance_df['Feature'].apply(get_feature_category)

# Set top N features to keep for each category
category_top_n = {
    'Texture': 6,
    'Spectral': 7,
    'Social semantic': 3
}

# Extract top N important features for each category and normalize within the class
top_features_list = []

for category, top_n in category_top_n.items():
    sub_df = importance_df[importance_df['Category'] == category].copy()
    sub_df = sub_df.sort_values(by='Importance', ascending=False).head(top_n).copy()
    sub_df['Importance_In_Class'] = sub_df['Importance'] / sub_df['Importance'].sum()
    sub_df['Importance_In_Class'] *= 2  # Visual enhancement: amplify 3 times
    top_features_list.append(sub_df)

# Merge the top N important features from all categories
final_df = pd.concat(top_features_list, ignore_index=True)

plt.rcParams['font.family'] = 'Times New Roman'

# Draw a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=final_df, x='Category', y='Importance_In_Class',
            palette='Set2', linewidth=1.5, fliersize=4)

# Set Y-axis range and ticks
plt.ylim(0.0, 1.1)  # Correct range, the top limit is 1.1
plt.yticks(np.arange(0.0, 1.11, 0.1))  # Small step of 0.1

# Set major ticks (every 0.2 for a big tick)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
ax.tick_params(axis='both', which='minor', length=4, width=1)

# Title and label fonts
plt.title('', fontsize=14)
plt.xlabel('', fontsize=13)
plt.ylabel('Feature Importance', fontsize=13)

plt.grid(False)
plt.tight_layout()
# Save the image (high resolution, change the save path to your desired location)
output_path = r"boxplot"  # ← Replace with your own path
plt.savefig(output_path, dpi=600, bbox_inches='tight')  # 600 DPI, suitable for paper figures
plt.show()

# ————————
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

# =================== Statistical range of importance for each feature type ===================
summary = final_df.groupby('Category')['Importance_In_Class'].agg(['min', 'max']).reset_index()
print("Importance range for each category (min - max):")
print(summary)

# =================== Custom "工" shape for whisker legend ===================
class WhiskerHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        top = Line2D([center_x - 5, center_x + 5], [center_y + 5, center_y + 5], color='black', lw=1)
        mid = Line2D([center_x, center_x], [center_y - 5, center_y + 5], color='black', lw=1)
        bot = Line2D([center_x - 5, center_x + 5], [center_y - 5, center_y - 5], color='black', lw=1)
        for l in (top, mid, bot): l.set_transform(trans)
        return [top, mid, bot]

# =================== Draw final boxplot ===================
# Custom colors: Convert RGB to matplotlib format in 0-1 range
custom_palette = {
    'Texture': (247/255, 183/255, 210/255),          # Texture
    'Spectral': (238/255, 193/255, 134/255),         # Spectral
    'Social semantic': (184/255, 229/255, 250/255)  # Social semantic
}

# Draw the final boxplot
plt.figure(figsize=(8, 6))
ax = sns.boxplot(
    data=final_df,
    x='Category',
    y='Importance_In_Class',
    palette=custom_palette,  # Use custom colors
    linewidth=1.5,
    fliersize=4,
    width=0.6,
    showmeans=True,
    meanprops={
        'marker': 's',
        'markerfacecolor': 'white',
        'markeredgecolor': 'black',
        'markersize': 6
    }
)

# Y-axis style
plt.ylim(0.0, 1.1)
plt.yticks(np.arange(0.0, 1.11, 0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
ax.tick_params(axis='both', which='minor', length=4, width=1)

# Legend elements
legend_elements = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='25%-75%'),
    Line2D([0], [0], color='black', lw=1, label='whisker'),  # Add normal line first
    Line2D([0], [0], color='black', lw=1, label='median'),
    Line2D([0], [0], color='black', marker='s', linestyle='None', markerfacecolor='white',
           markeredgecolor='black', markersize=6, label='mean')
]

# Replace whisker with custom "工" shape in legend
plt.legend(
    handles=legend_elements,
    handler_map={legend_elements[1]: WhiskerHandler()},
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    frameon=False
)

# Other labels and save
plt.xlabel('')
plt.ylabel('Feature importance', fontsize=13)
plt.grid(False)
plt.tight_layout()
plt.savefig(r"boxplot_a_final.png", dpi=600, bbox_inches='tight')
plt.show()

# ——————————
# Print mean, median, 25% and 75% percentiles for each category
stats = final_df.groupby('Category')['Importance_In_Class'].agg(
    Mean='mean',
    Median='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()

print("\nStatistical features importance (Mean, Median, Q1, Q3):")
print(stats.to_string(index=False, float_format='%.4f'))

# Print classification report for the test set (precision, recall, and F1-score for each class up to 4 decimal places)
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
