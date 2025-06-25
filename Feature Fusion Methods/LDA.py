import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# 5. Perform dimensionality reduction using LDA
lda = LinearDiscriminantAnalysis(n_components=7)  # Set the maximum number of classes-1 to retain (n_components can be adjusted)
features_train_lda = lda.fit_transform(features_train_scaled, groundtruth_train)
features_val_lda = lda.transform(features_val_scaled)
features_test_lda = lda.transform(features_test_scaled)

# 6. Combine the reduced data with the original area_num and groundtruth
train_result = pd.DataFrame(features_train_lda, columns=[f'LDA{i+1}' for i in range(features_train_lda.shape[1])])
train_result['area_num'] = area_num_train
train_result['groundtruth'] = groundtruth_train

val_result = pd.DataFrame(features_val_lda, columns=[f'LDA{i+1}' for i in range(features_val_lda.shape[1])])
val_result['area_num'] = area_num_val
val_result['groundtruth'] = groundtruth_val

test_result = pd.DataFrame(features_test_lda, columns=[f'LDA{i+1}' for i in range(features_test_lda.shape[1])])
test_result['area_num'] = area_num_test
test_result['groundtruth'] = groundtruth_test

# 7. Save the results to new Excel files
train_result.to_excel(r'train_data_lda7.xlsx', index=False)
val_result.to_excel(r'val_data_lda7.xlsx', index=False)
test_result.to_excel(r'test_data_lda7.xlsx', index=False)

print("LDA reduced data has been saved")
