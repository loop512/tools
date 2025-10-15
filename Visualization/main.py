import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data_file = './data.npy'
label_file = './729CNV_English_0.5_0.3S.csv'

data = np.load(data_file)

label_f = pd.read_csv(label_file)
labels = label_f["True Label"].values

pca = PCA(n_components=50, random_state=42)
B_pca = pca.fit_transform(data)

# Step 2: 用 t-SNE 降到 3 维
tsne = TSNE(n_components=3, perplexity=40, learning_rate=200, random_state=42, n_iter=1000)
B_3D = tsne.fit_transform(B_pca)

# Step 3: 3D 可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(B_3D[:, 0], B_3D[:, 1], B_3D[:, 2], c=labels, cmap="coolwarm", alpha=0.7)

ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_zlabel("t-SNE Dimension 3")
ax.set_title("3D t-SNE Visualization")

plt.colorbar(scatter, label="Label (0 or 1)")
plt.show()
