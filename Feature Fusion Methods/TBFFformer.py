import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# File paths
train_file_path = r'train_data.xlsx'
val_file_path = r'val_data.xlsx'
test_file_path = r'test_data.xlsx'

# Read data
def load_data(file_path):
    data_df = pd.read_excel(file_path)
    X_remote_sensing = data_df.filter(regex='^rs_')  # Remote sensing features
    X_social_economic = data_df.filter(regex='^se_')  # Socio-economic features
    y = data_df['groundtruth']  # Label column
    area_num = data_df['area_num']  # Area number column

    # Ensure all feature data is numeric
    X_remote_sensing = X_remote_sensing.apply(pd.to_numeric, errors='coerce')
    X_social_economic = X_social_economic.apply(pd.to_numeric, errors='coerce')

    # Check for missing values and fill them
    X_remote_sensing = X_remote_sensing.fillna(X_remote_sensing.mean())
    X_social_economic = X_social_economic.fillna(X_social_economic.mean())

    return X_remote_sensing, X_social_economic, y, area_num

# Load training, validation, and test datasets
X_rs_train, X_se_train, y_train, area_train = load_data(train_file_path)
X_rs_val, X_se_val, y_val, area_val = load_data(val_file_path)
X_rs_test, X_se_test, y_test, area_test = load_data(test_file_path)

# Standardize features
scaler = StandardScaler()
X_rs_train_scaled = scaler.fit_transform(X_rs_train)
X_rs_val_scaled = scaler.transform(X_rs_val)
X_rs_test_scaled = scaler.transform(X_rs_test)

X_se_train_scaled = scaler.fit_transform(X_se_train)
X_se_val_scaled = scaler.transform(X_se_val)
X_se_test_scaled = scaler.transform(X_se_test)

# Convert to PyTorch tensors
X_rs_train_tensor = torch.tensor(X_rs_train_scaled, dtype=torch.float32)
X_rs_val_tensor = torch.tensor(X_rs_val_scaled, dtype=torch.float32)
X_rs_test_tensor = torch.tensor(X_rs_test_scaled, dtype=torch.float32)

X_se_train_tensor = torch.tensor(X_se_train_scaled, dtype=torch.float32)
X_se_val_tensor = torch.tensor(X_se_val_scaled, dtype=torch.float32)
X_se_test_tensor = torch.tensor(X_se_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim_rs, input_dim_se, output_dim, num_heads=8, ff_hidden_dim=128, num_layers=4,
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.num_heads_rs = min(num_heads, input_dim_rs)
        self.num_heads_se = min(num_heads, input_dim_se)

        self.embed_dim_rs = (input_dim_rs // self.num_heads_rs) * self.num_heads_rs
        self.embed_dim_se = (input_dim_se // self.num_heads_se) * self.num_heads_se

        print(f"Adjusted embed dims - Remote sensing: {self.embed_dim_rs}, Social economic: {self.embed_dim_se}")
        print(f"Adjusted num_heads - Remote sensing: {self.num_heads_rs}, Social economic: {self.num_heads_se}")

        self.rs_linear = nn.Linear(input_dim_rs, self.embed_dim_rs)
        self.se_linear = nn.Linear(input_dim_se, self.embed_dim_se)

        self.transformer_rs = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim_rs, nhead=self.num_heads_rs,
                                       dim_feedforward=ff_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.transformer_se = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim_se, nhead=self.num_heads_se,
                                       dim_feedforward=ff_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.weight_net_rs = nn.Sequential(nn.Linear(self.embed_dim_rs, 1), nn.Sigmoid())
        self.weight_net_se = nn.Sequential(nn.Linear(self.embed_dim_se, 1), nn.Sigmoid())

        self.fc = nn.Sequential(nn.Linear(self.embed_dim_rs + self.embed_dim_se, ff_hidden_dim),
                                nn.ReLU(), nn.Linear(ff_hidden_dim, output_dim))

    def forward(self, x_rs, x_se):
        x_rs = self.rs_linear(x_rs).unsqueeze(0)
        x_se = self.se_linear(x_se).unsqueeze(0)

        for layer in self.transformer_rs:
            x_rs = layer(x_rs)
        for layer in self.transformer_se:
            x_se = layer(x_se)

        x_rs = x_rs.squeeze(0)
        x_se = x_se.squeeze(0)

        weight_rs = self.weight_net_rs(x_rs)
        weight_se = self.weight_net_se(x_se)

        weighted_rs = x_rs * weight_rs
        weighted_se = x_se * weight_se

        fused_features = torch.cat((weighted_rs, weighted_se), dim=1)

        output = self.fc(fused_features)
        return output

# Initialize the model
input_dim_rs = X_rs_train_scaled.shape[1]
input_dim_se = X_se_train_scaled.shape[1]
output_dim = len(y_train.unique())
model = TransformerModel(input_dim_rs, input_dim_se, output_dim, num_heads=8, ff_hidden_dim=128, num_layers=3,
                         dropout=0.1)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)

# Data loading
train_dataset = TensorDataset(X_rs_train_tensor, X_se_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_rs_val_tensor, X_se_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training function
best_val_accuracy = 0.0
best_weights = None
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
lr_list = []

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    global best_val_accuracy, best_weights
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs_rs, inputs_se, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_rs, inputs_se)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        lr_list.append(scheduler.get_last_lr()[0])

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs_rs, inputs_se, labels in val_loader:
                outputs = model(inputs_rs, inputs_se)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = model.state_dict()

# Models the model
train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100)

# Load the best weights
model.load_state_dict(best_weights)
print("Loaded the best weights based on validation accuracy.")

# Visualize training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Visualize learning rate changes
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(lr_list) + 1), lr_list, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.show()

# Extract fused features
def extract_fused_features(model, X_rs_tensor, X_se_tensor):
    model.eval()
    with torch.no_grad():
        X_rs_tensor = model.rs_linear(X_rs_tensor).unsqueeze(0)
        X_se_tensor = model.se_linear(X_se_tensor).unsqueeze(0)

        for layer in model.transformer_rs:
            X_rs_tensor = layer(X_rs_tensor)
        for layer in model.transformer_se:
            X_se_tensor = layer(X_se_tensor)

        fused_features = torch.cat((X_rs_tensor.squeeze(0), X_se_tensor.squeeze(0)), dim=1)
        return fused_features.float()

# Save fused features
def save_fused_features(model, X_rs_tensor, X_se_tensor, area_num, y, output_file_path):
    fused_features = extract_fused_features(model, X_rs_tensor, X_se_tensor)
    fused_features_df = pd.DataFrame(fused_features.numpy())
    fused_features_df['area_num'] = area_num.values
    fused_features_df['groundtruth'] = y.values
    fused_features_df.to_excel(output_file_path, index=False)
    print(f"Fused features have been saved to: {output_file_path}")

# Save fused features for training, validation, and test datasets
save_fused_features(model, X_rs_train_tensor, X_se_train_tensor, area_train, y_train,
                    r'train_fused_features.xlsx')
save_fused_features(model, X_rs_val_tensor, X_se_val_tensor, area_val, y_val,
                    r'val_fused_features.xlsx')
save_fused_features(model, X_rs_test_tensor, X_se_test_tensor, area_test, y_test,
                    r'test_fused_features.xlsx')
