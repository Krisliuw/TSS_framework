import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== Data Loading =====================
train_data = pd.read_excel(r'train_data.xlsx')
val_data = pd.read_excel(r'val_data.xlsx')
test_data = pd.read_excel(r'test_data.xlsx')

X_train = train_data.drop(columns=['area_num', 'groundtruth']).values
y_train = train_data['groundtruth'].values
X_val = val_data.drop(columns=['area_num', 'groundtruth']).values
y_val = val_data['groundtruth'].values
X_test = test_data.drop(columns=['area_num', 'groundtruth']).values
y_test = test_data['groundtruth'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===================== CNN Model Definition =====================
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(704, 128)  # Note: This feature size must be dynamically determined based on your input dimensions
        self.fc2 = nn.Linear(128, 11)  # Multi-class classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNNModel(input_dim=47)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ===================== Metric Calculation Function =====================
def calculate_metrics(all_labels, all_preds):
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    kappa = cohen_kappa_score(all_labels, all_preds)

    metrics_df = pd.DataFrame({
        'Class': range(1, len(precision) + 1),
        'Precision': precision.round(4),
        'Recall': recall.round(4),
        'F1-score': f1.round(4)
    })

    overall_metrics = pd.DataFrame({
        'Class': ['Overall'],
        'Precision': [precision.mean().round(4)],
        'Recall': [recall.mean().round(4)],
        'F1-score': [f1.mean().round(4)]
    })

    metrics_df = pd.concat([metrics_df, overall_metrics], ignore_index=True)
    print(metrics_df)
    print("Kappa coefficient:", round(kappa, 4))

# ===================== Accuracy Visualization =====================
def plot_accuracy(train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Models Accuracy")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

# ===================== Model Save Path =====================
best_model_path = r'D:\data\Self_Attention\Dataset_70\models\best_cnn_model.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# ===================== Training Function (Save the Best Model) =====================
def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_accuracies.append(train_acc)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Models Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # ✅ Save the best model based on validation performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved: {best_model_path}")

        # Output detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            calculate_metrics(np.array(all_labels), np.array(all_preds))

    plot_accuracy(train_accuracies, val_accuracies, epochs)

# ===================== Test Function =====================
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    calculate_metrics(np.array(all_labels), np.array(all_preds))

# ===================== Run Training and Testing =====================
train(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# ✅ Load and test the best model
model.load_state_dict(torch.load(best_model_path))
print("\n📌 Loading the best model for test evaluation:")
test(model, test_loader)
