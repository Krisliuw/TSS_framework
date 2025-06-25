import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import os

# Set up font support
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== Step 1: Load model ==========
model_path = r'best_xgb_fusemodel.model'
bst = xgb.Booster()
bst.load_model(model_path)
print(f"✅ Successfully loaded model: {model_path}")

# ========== Step 2: Load data ==========
data_path = r'Allfused_noA.xlsx'
df = pd.read_excel(data_path)

# Confirm columns
print(f"Loaded data column names: {df.columns}")

# ========== Step 3: Data preprocessing ==========
X = df.iloc[:, :-2]  # Remove 'area_num' and 'groundtruth' columns
y_true = df['groundtruth'].values
area_num = df['area_num']
X = X.fillna(X.mean())

print(f"Column names of X: {X.columns}")

# ========== Step 4: Model prediction ==========
dX = xgb.DMatrix(X)
y_pred = bst.predict(dX).astype(int)

# ========== Step 5: Save prediction results ==========
df['predict_result'] = y_pred
df_result = df[['area_num', 'predict_result']]
save_dir = r'predictMarch31'
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, 'predictions_fusedxgb.xlsx')
df_result.to_excel(output_path, index=False)
print(f"✅ Prediction results have been saved to: {output_path}")

# ========== Step 6: Performance metrics ==========
oa = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"\n📊 Overall Accuracy (OA): {oa * 100:.2f}%")
print(f"📈 Kappa Score: {kappa:.4f}")

# ========== Step 7: Confusion matrix plotting ==========
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR',
    5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}
labels = [class_labels[i] for i in range(11)]
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Custom color map
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
for _, _, start_rgb, end_rgb in color_segments:
    colors.extend([tuple(c/255 for c in start_rgb), tuple(c/255 for c in end_rgb)])
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=512)

# Visualization settings
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
    annot_kws={'size': 20, 'weight': 'bold'},
    cbar_kws={'label': 'Accuracy', 'extend': 'both'},
    linewidths=0,
    linecolor='none',
    xticklabels=labels,
    yticklabels=labels
)

# Set diagonal text to white
ax = heatmap.axes
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            text = ax.texts[i * len(labels) + j]
            text.set_color('white')

# Label settings
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

# Title and axis labels
plt.title('Confusion Matrix (Fused XGBoost)', fontsize=18, pad=25, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, labelpad=15)
plt.ylabel('True Label', fontsize=18, labelpad=20)

# Colorbar settings
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=14)
colorbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=30, fontsize=16, fontweight='bold')

# Save image
plt.tight_layout(pad=3.0)
cm_image_path = os.path.join(save_dir, "confusion_matrix_fusedxgb.png")
plt.savefig(cm_image_path, dpi=350, bbox_inches='tight')
print(f"✅ Confusion matrix image has been saved to: {cm_image_path}")
plt.show()

# Save raw confusion matrix
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_excel_path = os.path.join(save_dir, "confusion_matrix_raw_fusedxgb.xlsx")
cm_df.to_excel(cm_excel_path)
print(f"✅ Confusion matrix data has been saved to: {cm_excel_path}")
