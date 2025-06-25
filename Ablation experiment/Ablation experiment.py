import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Define the three types of features you already have
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

# Group the features into categories
feature_groups = {
    "All Features": glcm_features + vari_features + tfidf_features,
    "Texture Only": glcm_features,
    "Spectral Only": vari_features,
    "Semantic Only": tfidf_features,
    "w/o Texture": vari_features + tfidf_features,
    "w/o Spectral": glcm_features + tfidf_features,
    "w/o Semantic": glcm_features + vari_features
}

# Load data function
def load_data(path):
    df = pd.read_excel(path)
    X = df.drop(columns=['area_num', 'groundtruth'])
    y = df['groundtruth']
    return X, y

# Load training and test data
X_train, y_train = load_data(r'train_data.xlsx')
X_test, y_test = load_data(r'test_data.xlsx')

# Training parameters
params = {
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'num_class': len(y_train.unique()),
    'max_depth': 5,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 2.5,
    'alpha': 1.5,
    'min_child_weight': 5
}

results = []

# Iterate through each feature group and train the model
for name, features in feature_groups.items():
    X_tr = X_train[features].fillna(0)
    X_te = X_test[features].fillna(0)

    # Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
    smote = SMOTE(sampling_strategy={7: 30}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tr, y_train)

    dtrain = xgb.DMatrix(X_resampled, label=y_resampled)
    dtest = xgb.DMatrix(X_te, label=y_test)

    # Models the model
    bst = xgb.train(params, dtrain, num_boost_round=150)

    # Make predictions
    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)

    # Calculate accuracy
    acc_train = accuracy_score(y_resampled, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    results.append({
        'Feature Combination': name,
        'Models Accuracy': round(acc_train, 4),
        'Test Accuracy': round(acc_test, 4)
    })

# Output the results
results_df = pd.DataFrame(results)
print(results_df)
