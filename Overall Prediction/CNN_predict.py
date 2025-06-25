import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os

# ========== Set paths ==========
model_path = r'models\best_cnn_model.pth'
data_path = r'alldata.xlsx'
save_path = r'predictions_cnn.xlsx'

# ========== Define CNN architecture ==========
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(704, 128)
        self.fc2 = nn.Linear(128, 11)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# ========== Load model ==========
model = CNNModel(input_dim=47)
model.load_state_dict(torch.load(model_path))
model.eval()

# ========== Data processing ==========
df = pd.read_excel(data_path)
X = df.drop(columns=['area_num', 'groundtruth']).values
y_true = df['groundtruth'].values

scaler = StandardScaler()
X_tensor = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)

# ========== Model prediction ==========
with torch.no_grad():
    y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()

# ========== Save results ==========
df['predicted'] = y_pred
save_dir = os.path.dirname(save_path)
os.makedirs(save_dir, exist_ok=True)
df.to_excel(save_path, index=False)

# ========== Performance evaluation ==========
print(f"OA: {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")

# ========== Confusion matrix configuration ==========
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR',
    5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}
labels = [class_labels[i] for i in range(11)]  # English label list

# Calculate normalized matrix
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# ========== Color map definition ==========
color_segments = [
    (0.00, 0.12, (255,247,236), (254,233,202)),
    (0.12, 0.25, (253,231,199), (253,210,157)),
    (0.25, 0.38, (252,205,149), (253,166,113)),
    (0.38, 0.50, (252,175,121), (247,126,83)),
    (0.50, 0.60, (245,119,79), (228,78,54)),
    (0.60, 0.75, (225,70,48), (211,41,26)),
    (0.75, 0.88, (186,10,5), (170,0,0)),
    (0.88, 1.00, (164,0,0), (128,0,0))
]

colors = []
for seg in color_segments:
    _, _, start_rgb, end_rgb = seg
    colors.extend([tuple(c/255 for c in start_rgb),
                 tuple(c/255 for c in end_rgb)])

cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    colors,
    N=512
)

# ========== Visualization settings ==========
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.unicode_minus': False
})

# ========== Plot confusion matrix ==========
plt.figure(figsize=(14, 10))  # Adjust canvas size

heatmap = sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    annot_kws={
        'size': 20,
        'weight': 'bold'
    },
    cbar_kws={
        'label': 'Accuracy',
        'extend': 'both'
    },
    linewidths=0,
    linecolor='none',
    xticklabels=labels,
    yticklabels=labels
)

# ========== Set diagonal numbers to white ==========
ax = heatmap.axes
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            text = ax.texts[i*len(labels) + j]
            text.set_color('white')

# ========== Axis label style settings ==========
# X-axis labels (centered horizontally, no tilt)
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=0,
    horizontalalignment='center',
    fontsize=14,
    fontweight='semibold'
)

# Y-axis labels (centered vertically, no tilt)
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0,
    fontsize=14,
    verticalalignment='center',
    fontweight='semibold'
)

# ========== Title and label settings ==========
plt.title('Confusion Matrix (CNN)',
         fontsize=18,
         pad=25,
         fontweight='bold')
plt.xlabel('Predicted Label',
          fontsize=18,
          labelpad=15)
plt.ylabel('True Label',
          fontsize=18,
          labelpad=20)

# ========== Color bar settings ==========
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=14)
colorbar.ax.set_ylabel('Accuracy',
                      rotation=270,
                      labelpad=30,
                      fontsize=16,
                      fontweight='bold')

# ========== Save and display ==========
plt.tight_layout(pad=3.0)
cm_image_path = os.path.join(save_dir, "confusion_matrix_cnn.png")
plt.savefig(cm_image_path, dpi=350, bbox_inches='tight')
print(f"✅ Confusion matrix image saved to: {cm_image_path}")
plt.show()

# Save the raw matrix data
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_excel(os.path.join(save_dir, "confusion_matrix_raw.xlsx"))
