import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Read the data
train_data = pd.read_excel(r'train_data.xlsx')
val_data = pd.read_excel(r'val_data.xlsx')
test_data = pd.read_excel(r'test_data.xlsx')

# 2. Separate area_num, groundtruth, and feature columns
def separate_columns(data):
    area_num = data['area_num']
    groundtruth = data['groundtruth']
    features = data.drop(columns=['area_num', 'groundtruth'])
    return area_num, groundtruth, features

# 3. Separate columns for each dataset
area_num_train, groundtruth_train, features_train = separate_columns(train_data)
area_num_val, groundtruth_val, features_val = separate_columns(val_data)
area_num_test, groundtruth_test, features_test = separate_columns(test_data)

# 4. Standardize the features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_val_scaled = scaler.transform(features_val)
features_test_scaled = scaler.transform(features_test)

# 5. Perform dimensionality reduction using PCA (can specify the number of components to retain)
pca = PCA(n_components=0.99)  # Retain 99% of the variance
features_train_pca = pca.fit_transform(features_train_scaled)
features_val_pca = pca.transform(features_val_scaled)
features_test_pca = pca.transform(features_test_scaled)

# 6. Combine the reduced data with the original area_num and groundtruth
train_result = pd.DataFrame(features_train_pca, columns=[f'PC{i+1}' for i in range(features_train_pca.shape[1])])
train_result['area_num'] = area_num_train
train_result['groundtruth'] = groundtruth_train

val_result = pd.DataFrame(features_val_pca, columns=[f'PC{i+1}' for i in range(features_val_pca.shape[1])])
val_result['area_num'] = area_num_val
val_result['groundtruth'] = groundtruth_val

test_result = pd.DataFrame(features_test_pca, columns=[f'PC{i+1}' for i in range(features_test_pca.shape[1])])
test_result['area_num'] = area_num_test
test_result['groundtruth'] = groundtruth_test

# 7. Save the results to new Excel files
train_result.to_excel(r'train_data_pca.xlsx', index=False)
val_result.to_excel(r'val_data_pca.xlsx', index=False)
test_result.to_excel(r'test_data_pca.xlsx', index=False)

print("Dimensionality reduced data has been saved")
