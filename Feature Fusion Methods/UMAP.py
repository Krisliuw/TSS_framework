import pandas as pd
import umap
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

# 5. Perform dimensionality reduction using UMAP
umap_model = umap.UMAP(n_components=2)  # Reduce to 2 dimensions, can be set to other dimensions as well

features_train_umap = umap_model.fit_transform(features_train_scaled)
features_val_umap = umap_model.transform(features_val_scaled)
features_test_umap = umap_model.transform(features_test_scaled)

# 6. Combine the reduced data with the original area_num and groundtruth
train_umap = pd.DataFrame(features_train_umap, columns=['umap1', 'umap2'])
train_umap['area_num'] = area_num_train
train_umap['groundtruth'] = groundtruth_train

val_umap = pd.DataFrame(features_val_umap, columns=['umap1', 'umap2'])
val_umap['area_num'] = area_num_val
val_umap['groundtruth'] = groundtruth_val

test_umap = pd.DataFrame(features_test_umap, columns=['umap1', 'umap2'])
test_umap['area_num'] = area_num_test
test_umap['groundtruth'] = groundtruth_test

# 7. Output the reduced data, can be saved as new Excel files
train_umap.to_excel(r'train_umap.xlsx', index=False)
val_umap.to_excel(r'val_umap.xlsx', index=False)
test_umap.to_excel(r'test_umap.xlsx', index=False)

print("Dimensionality reduced data has been saved.")
