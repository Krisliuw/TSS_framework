import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib.colors as mcolors
import os
import matplotlib

# Font support
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Path settings
model_path = r'best_rf_model.joblib'
data_path = r'alldata.xlsx'
save_dir = r'predictMarch31'
os.makedirs(save_dir, exist_ok=True)
save_result_path = os.path.join(save_dir, 'RF_predictions.xlsx')
save_cm_image = os.path.join(save_dir, 'confusion_matrix_rf.png')
save_cm_excel = os.path.join(save_dir, 'confusion_matrix_rf_raw.xlsx')

# Load model and data
rf_model = joblib.load(model_path)
df = pd.read_excel(data_path)

X = df.drop(columns=['area_num', 'groundtruth'])
y_true = df['groundtruth']
X = X.fillna(X.mean())

# Model prediction
y_pred = rf_model.predict(X)

# Save prediction results
df['predicted'] = y_pred
df.to_excel(save_result_path, index=False)
print(f"✅ Prediction results have been saved to: {save_result_path}")

# Accuracy and Kappa coefficient
oa = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"📊 Overall Accuracy (OA): {oa * 100:.2f}%")
print(f"📈 Kappa Score: {kappa:.4f}")

# Confusion matrix and normalization
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Class labels
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR',
    5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}
labels = [class_labels[i] for i in range(len(class_labels))]

# CNN-style color mapping
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
for _, _, rgb1, rgb2 in color_segments:
    colors.extend([tuple(c / 255 for c in rgb1), tuple(c / 255 for c in rgb2)])
cmap = mcolors.LinearSegmentedColormap.from_list("custom_rf", colors, N=512)

# Font settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.unicode_minus': False
})

# Plotting
plt.figure(figsize=(14, 10))
heatmap = sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={'size': 20, 'weight': 'bold'},
    cbar_kws={'label': 'Accuracy', 'extend': 'both'}
)

# Set diagonal text to white
ax = heatmap.axes
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.texts[i * len(labels) + j]
        if i == j:
            text.set_color('white')
        else:
            text.set_color('black')

# Axis labels
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=0,
    horizontalalignment='center',
    fontsize=14,
    fontweight='semibold'
)
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0,
    fontsize=14,
    verticalalignment='center',
    fontweight='semibold'
)

# Title and labels
plt.title('Confusion Matrix (RF)', fontsize=18, pad=25, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, labelpad=15, fontweight='bold')
plt.ylabel('True Label', fontsize=18, labelpad=20, fontweight='bold')

# Color bar
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=30, fontsize=16, fontweight='bold')

# Save image
plt.tight_layout(pad=3.0)
plt.savefig(save_cm_image, dpi=350, bbox_inches='tight')
plt.show()
print(f"✅ Confusion matrix image has been saved to: {save_cm_image}")

# Save raw confusion matrix data
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_excel(save_cm_excel)
print(f"✅ Confusion matrix data has been saved to: {save_cm_excel}")
