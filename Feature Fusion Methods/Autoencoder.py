import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1. Read the data
file_path = r"D:alldata.xlsx"
df = pd.read_excel(file_path)

# 2. Extract features, area_num, and groundtruth
features = df.drop(columns=['area_num', 'groundtruth'])  # Drop unnecessary columns
area_num = df['area_num']  # Keep the area_num column
groundtruth = df['groundtruth']  # Keep the groundtruth column

# 3. Standardize the feature data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Build the autoencoder model

# Input layer (dimension of the original data)
input_layer = Input(shape=(features_scaled.shape[1],))

# Encoder part
encoded = Dense(256, activation='relu')(input_layer)  # Increase the number of neurons
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Higher dimensional latent space

# Encoder part model of the autoencoder
encoder_model = Model(input_layer, encoded)

# 5. Models the autoencoder and record the reconstruction error at each epoch
autoencoder = Model(input_layer, Dense(features_scaled.shape[1], activation='sigmoid')(encoded))  # Complete autoencoder model
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Use early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Models the autoencoder
history = autoencoder.fit(features_scaled, features_scaled,
                          epochs=100,  # Models for 100 epochs
                          batch_size=256,
                          shuffle=True,
                          validation_data=(features_scaled, features_scaled),
                          verbose=1,
                          callbacks=[early_stopping])  # Add early stopping

# 6. Get the low-dimensional features from the encoder part
encoded_features = encoder_model.predict(features_scaled)

# 7. Plot the reconstruction error during training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Reconstruction Error During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()

# 8. Output the low-dimensional features extracted by the encoder part
encoded_df = pd.DataFrame(encoded_features, columns=[f"encoded_{i}" for i in range(encoded_features.shape[1])])

# 9. Merge `area_num` and `groundtruth`
encoded_df['area_num'] = area_num
encoded_df['groundtruth'] = groundtruth

# Output the merged DataFrame (can be saved as a new Excel file)
encoded_df.to_excel(r"autoencoded_data.xlsx", index=False)

# Print part of the results to check the merged data
print(encoded_df.head())
