import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Read data
file_path = r"alldata.xlsx"
# file_path = r"Allfused.xlsx"
df = pd.read_excel(file_path)

# Drop unnecessary columns
df = df.drop(columns=["area_num", "groundtruth"], errors='ignore')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a mask to display only the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Define custom color palette
colors = [
    # blue area (0.0 to -1.0)
    (66 / 255, 127 / 255, 156 / 255),  # RGB(66, 127, 156) (Deep Blue)
    (109 / 255, 157 / 255, 180 / 255),  # RGB(109, 157, 180) (-0.75)
    (155 / 255, 188 / 255, 205 / 255),  # RGB(155, 188, 205) (-0.50)
    (200 / 255, 218 / 255, 228 / 255),  # RGB(200, 218, 228) (-0.25)
    (242 / 255, 242 / 255, 242 / 255),  # RGB(242, 242, 242) (Light Blue)
    # red area (1.0 to 0.0)
    (242 / 255, 242 / 255, 242 / 255),  # RGB(242, 242, 242) (Light Red)
    (239 / 255, 206 / 255, 197 / 255),  # RGB(239, 206, 197) (0.25)
    (223 / 255, 165 / 255, 151 / 255),  # RGB(223, 165, 151) (0.50)
    (209 / 255, 124 / 255, 103 / 255),  # RGB(209, 124, 103) (0.75)
    (194 / 255, 86 / 255, 58 / 255),  # RGB(194, 86, 58) (Deep Red)
]

# Create a custom color map using LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Create and display the heatmap
plt.figure(figsize=(12, 10))  # Adjust the figure size for high resolution
ax = sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap=custom_cmap, center=0, linewidths=0.5, vmin=-1.0,
                 vmax=1.0)

# # Get the mappable from the ax object and create the colorbar
# plt.colorbar(ax.collections[0], label='Correlation Coefficient')
# Set the y-axis labels to horizontal
# Make the x-axis and y-axis labels bold
# Set axis labels to horizontal, bold font, and use Times New Roman
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
plt.title("Feature Correlation Heatmap (Triangular)")

# Save the heatmap to the specified path with high resolution (600 dpi)
save_path = r"alldata_heatmap"
plt.savefig(save_path, bbox_inches='tight', dpi=600)  # 600 dpi for higher resolution

# Display the heatmap
plt.show()
