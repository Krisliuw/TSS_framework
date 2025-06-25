import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.colors as mcolors

# Set up font support
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== Step 1: Load the saved model ==========
model_path = r'best_xgb_model.model'
bst = xgb.Booster()
bst.load_model(model_path)
print(f"✅ Successfully loaded model: {model_path}")

# ========== Step 2: Load data ==========
data_path = r'alldata.xlsx'
df = pd.read_excel(data_path)

# ========== Step 3: Data preprocessing ==========
X = df.iloc[:, 1:-1]  # Extract feature columns (excluding 'area_num' and 'groundtruth')
y_true = df['groundtruth'].values  # Extract true labels
area_num = df['area_num']  # Extract area number column

# Handle missing values
X = X.fillna(X.mean())

# ========== Step 4: Prediction ==========
dX = xgb.DMatrix(X)
y_pred = bst.predict(dX).astype(int)

# ========== Step 5: Add prediction results and save ==========
df['predict_result'] = y_pred
df_result = df[['area_num', 'predict_result']]
save_dir = r'predictMarch31'
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, 'predictions_xgb.xlsx')
df_result.to_excel(output_path, index=False)
print(f"✅ Prediction results have been saved to: {output_path}")

# ========== Step 6: Metrics calculation ==========
oa = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"\n📊 Overall Accuracy (OA): {oa * 100:.2f}%")
print(f"📈 Kappa Score: {kappa:.4f}")

# ========== Step 7: Confusion matrix configuration ==========
class_labels = {
    0: 'P', 1: 'C', 2: 'R', 3: 'PR', 4: 'CR',
    5: 'I', 6: 'U', 7: 'W', 8: 'G', 9: 'A', 10: 'F'
}

# Calculate confusion matrix
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
    colors.extend([tuple(c/255 for c in start_rgb), tuple(c/255 for c in end_rgb)])

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=512)

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
    annot_kws={'size': 20, 'weight': 'bold'},
    cbar_kws={'label': 'Accuracy', 'extend': 'both'},
    linewidths=0,
    linecolor='none',
    xticklabels=[class_labels[i] for i in range(len(class_labels))],
    yticklabels=[class_labels[i] for i in range(len(class_labels))]
)

# ========== Set diagonal text to white ==========
ax = heatmap.axes
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        if i == j:
            text = ax.texts[i * len(class_labels) + j]
            text.set_color('white')

# ========== Axis label style settings ==========
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

# ========== Title and label settings ==========
plt.title('Confusion Matrix (XGBoost)', fontsize=18, pad=25, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, labelpad=15)
plt.ylabel('True Label', fontsize=18, labelpad=20)

# ========== Colorbar settings ==========
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=14)
colorbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=30, fontsize=16, fontweight='bold')

# ========== Save and display ==========
plt.tight_layout(pad=3.0)
cm_image_path = os.path.join(save_dir, "confusion_matrix_xgb.png")
plt.savefig(cm_image_path, dpi=350, bbox_inches='tight')
print(f"✅ Confusion matrix image has been saved to: {cm_image_path}")
plt.show()

# ========== Save raw confusion matrix data ==========
cm_df = pd.DataFrame(cm, index=[class_labels[i] for i in range(len(class_labels))], columns=[class_labels[i] for i in range(len(class_labels))])
cm_excel_path = os.path.join(save_dir, "confusion_matrix_raw_xgb.xlsx")
cm_df.to_excel(cm_excel_path)
print(f"✅ Confusion matrix data has been saved to: {cm_excel_path}")
