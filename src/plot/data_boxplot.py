import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./dataset/europe.csv')

scaler = StandardScaler()

numerical_cols = data.columns[1:]

normalized_data = scaler.fit_transform(data[numerical_cols])

normalized_df = pd.DataFrame(normalized_data, columns=numerical_cols)

normalized_df['Country'] = data['Country']

plt.figure(figsize=(15, 10))

colors = plt.cm.viridis(np.linspace(0, 1, len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    box = plt.boxplot(
        normalized_df[col],
        positions=[i + 1],
        boxprops=dict(facecolor=colors[i], edgecolor='darkgrey', linewidth=2),  # Enhanced outline
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5, color='grey'),
        capprops=dict(linewidth=2, color='grey'),
        widths=0.5,
        patch_artist=True
    )


plt.title('Boxplots of Normalized Country Variables')
plt.xticks(ticks=np.arange(1, len(numerical_cols) + 1), labels=numerical_cols, rotation=45)
plt.ylabel('Normalized Values')
plt.xlabel('Variables')

plt.grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

for i, col in enumerate(numerical_cols):
    box = plt.boxplot(
        data[col],
        positions=[i + 1],
        boxprops=dict(facecolor=colors[i], edgecolor='darkgrey', linewidth=2),  # Enhanced outline
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5, color='grey'),
        capprops=dict(linewidth=2, color='grey'),
        widths=0.5,
        patch_artist=True
    )


plt.title('Boxplots of Country Variables')
plt.xticks(ticks=np.arange(1, len(numerical_cols) + 1), labels=numerical_cols, rotation=45)
plt.ylabel('Values')
plt.xlabel('Variables')

plt.grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)

plt.figure(figsize=(10, 8))

plt.scatter(pca_result[:, 0], pca_result[:, 1], color='blue')

for i, col in enumerate(numerical_cols):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              color='red', alpha=0.5, head_width=0.05, head_length=0.1)
    plt.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, col, color='red')

for i, country in enumerate(data['Country']):
    plt.text(pca_result[i, 0], pca_result[i, 1], country, fontsize=9, ha='right')

plt.title('Biplot of PCA on Normalized Country Statistics')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.tight_layout()
plt.show()

pc1 = pca_result[:, 0]

plt.figure(figsize=(12, 6))
plt.bar(data['Country'], pc1, color='skyblue')
plt.title('Bar Graph of Principal Component 1 (PC1)')
plt.xlabel('Country')
plt.ylabel('PC1 Value')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()