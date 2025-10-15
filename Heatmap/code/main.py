import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import dcor
import matplotlib.patches as patches

# Set Variable names
var_names = ['L0', 'L1', 'L2', 'L3', 'P1', 'P0', 'C1', 'S1', 'GA1', 'GB1', 'P2', 'C2', 'S2', 'GA2', 'GB2']
data_root = '../data'
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# load the data
def load_data(path):
    return pd.read_csv(path).values

def load_all_data(root):
    cover = load_data(os.path.join(root, 'cover', 'data.csv'))
    stego05 = load_data(os.path.join(root, 'stego0.5', 'data.csv'))
    stego10 = load_data(os.path.join(root, 'stego1.0', 'data.csv'))
    return cover, stego05, stego10

# calculate the MI matrix
def compute_mi_matrix(data):
    n = data.shape[1]
    mi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mi = mutual_info_regression(data[:, [i]], data[:, j], discrete_features=False)
                mi_matrix[i, j] = mi[0]
    return mi_matrix

# calculate the distance correlation matrix
def compute_dcor_matrix(data):
    n = data.shape[1]
    dcor_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dcor_matrix[i, j] = dcor.distance_correlation(data[:, i], data[:, j])
    return dcor_matrix

# save the matrices
def save_matrix_csv(matrix, var_names, filename):
    df = pd.DataFrame(matrix, columns=var_names, index=var_names)
    df.to_csv(filename)
    print(f"Saved matrix to: {filename}")

# 带变化/未变化标注的热力图，框均为虚线 generate the heatmaps (changes are marked)
def plot_side_by_side_diff_with_dual_box(matrix1, matrix2, title1, title2,
                                         vmin=None, vmax=None,
                                         change_mask1=None, change_mask2=None,
                                         unchange_mask1=None, unchange_mask2=None,
                                         color_change='#E67E22',  # 橙色虚线：变化
                                         color_unchange='#2ECC71',  # 绿色虚线：未变化
                                         save_as=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    for ax, matrix, title, mask_change, mask_unchange in zip(
        axes,
        [matrix1, matrix2],
        [title1, title2],
        [change_mask1, change_mask2],
        [unchange_mask1, unchange_mask2]
    ):
        sns.heatmap(matrix, ax=ax,
                    xticklabels=var_names, yticklabels=var_names,
                    cmap='bwr', center=0, vmin=vmin, vmax=vmax,
                    square=True, cbar=True, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        if mask_change is not None:
            for i in range(mask_change.shape[0]):
                for j in range(mask_change.shape[1]):
                    if mask_change[i, j]:
                        rect = patches.Rectangle((j, i), 1, 1, fill=False,
                                                 edgecolor=color_change, linewidth=1, linestyle='--')
                        ax.add_patch(rect)
        if mask_unchange is not None:
            for i in range(mask_unchange.shape[0]):
                for j in range(mask_unchange.shape[1]):
                    if mask_unchange[i, j]:
                        rect = patches.Rectangle((j, i), 1, 1, fill=False,
                                                 edgecolor=color_unchange, linewidth=1, linestyle='--')
                        ax.add_patch(rect)

    plt.tight_layout()
    if save_as:
        filepath = os.path.join(output_dir, save_as)
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to: {filepath}")
    plt.show()

# generate the heatmaps (changes are not marked)
def plot_side_by_side_plain(matrix1, matrix2, title1, title2,
                            vmin=None, vmax=None, save_as=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    for ax, matrix, title in zip(axes, [matrix1, matrix2], [title1, title2]):
        sns.heatmap(matrix, ax=ax,
                    xticklabels=var_names, yticklabels=var_names,
                    cmap='bwr', center=0, vmin=vmin, vmax=vmax,
                    square=True, cbar=True, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    if save_as:
        filepath = os.path.join(output_dir, save_as)
        plt.savefig(filepath, dpi=300)
        print(f"Saved plain figure to: {filepath}")
    plt.show()

def main():
    # load data
    cover, stego05, stego10 = load_all_data(data_root)

    # calculate MI and dCor matrices
    mi_cover = compute_mi_matrix(cover)
    mi_05 = compute_mi_matrix(stego05)
    mi_10 = compute_mi_matrix(stego10)

    dcor_cover = compute_dcor_matrix(cover)
    dcor_05 = compute_dcor_matrix(stego05)
    dcor_10 = compute_dcor_matrix(stego10)

    # save CSV
    save_matrix_csv(mi_cover, var_names, 'MI_cover.csv')
    save_matrix_csv(mi_05, var_names, 'MI_stego0.5.csv')
    save_matrix_csv(mi_10, var_names, 'MI_stego1.0.csv')
    save_matrix_csv(dcor_cover, var_names, 'dCor_cover.csv')
    save_matrix_csv(dcor_05, var_names, 'dCor_stego0.5.csv')
    save_matrix_csv(dcor_10, var_names, 'dCor_stego1.0.csv')

    # difference matrices
    mi_diff_05 = mi_05 - mi_cover
    mi_diff_10 = mi_10 - mi_cover
    dcor_diff_05 = dcor_05 - dcor_cover
    dcor_diff_10 = dcor_10 - dcor_cover

    absmax_mi = np.max(np.abs([mi_diff_05, mi_diff_10]))
    absmax_dcor = np.max(np.abs([dcor_diff_05, dcor_diff_10]))

    # mask
    mi_nonzero_mask_05 = mi_diff_05 != 0
    mi_nonzero_mask_10 = mi_diff_10 != 0
    mi_zero_mask_05 = mi_diff_05 == 0
    mi_zero_mask_10 = mi_diff_10 == 0

    dcor_nonzero_mask_05 = dcor_diff_05 != 0
    dcor_nonzero_mask_10 = dcor_diff_10 != 0
    dcor_zero_mask_05 = dcor_diff_05 == 0
    dcor_zero_mask_10 = dcor_diff_10 == 0

    # MI heatmap

    plot_side_by_side_diff_with_dual_box(mi_diff_05, mi_diff_10,
                                         "MI Δ (Stego 0.5 - Cover) [Changed]",
                                         "MI Δ (Stego 1.0 - Cover) [Changed]",
                                         vmin=-absmax_mi, vmax=absmax_mi,
                                         change_mask1=mi_nonzero_mask_05,
                                         change_mask2=mi_nonzero_mask_10,
                                         save_as='MI_diff_changed.png')

    plot_side_by_side_diff_with_dual_box(mi_diff_05, mi_diff_10,
                                         "MI Δ (Stego 0.5 - Cover) [Unchanged]",
                                         "MI Δ (Stego 1.0 - Cover) [Unchanged]",
                                         vmin=-absmax_mi, vmax=absmax_mi,
                                         unchange_mask1=mi_zero_mask_05,
                                         unchange_mask2=mi_zero_mask_10,
                                         save_as='MI_diff_unchanged.png')

    plot_side_by_side_plain(mi_diff_05, mi_diff_10,
                            "MI Δ (Stego 0.5 - Cover)",
                            "MI Δ (Stego 1.0 - Cover)",
                            vmin=-absmax_mi, vmax=absmax_mi,
                            save_as='MI_diff_plain.png')

    # dCor heatmap
    plot_side_by_side_diff_with_dual_box(dcor_diff_05, dcor_diff_10,
                                         "dCor Δ (Stego 0.5 - Cover) [Changed]",
                                         "dCor Δ (Stego 1.0 - Cover) [Changed]",
                                         vmin=-absmax_dcor, vmax=absmax_dcor,
                                         change_mask1=dcor_nonzero_mask_05,
                                         change_mask2=dcor_nonzero_mask_10,
                                         save_as='dCor_diff_changed.png')

    plot_side_by_side_diff_with_dual_box(dcor_diff_05, dcor_diff_10,
                                         "dCor Δ (Stego 0.5 - Cover) [Unchanged]",
                                         "dCor Δ (Stego 1.0 - Cover) [Unchanged]",
                                         vmin=-absmax_dcor, vmax=absmax_dcor,
                                         unchange_mask1=dcor_zero_mask_05,
                                         unchange_mask2=dcor_zero_mask_10,
                                         save_as='dCor_diff_unchanged.png')

    plot_side_by_side_plain(dcor_diff_05, dcor_diff_10,
                            "dCor Δ (Stego 0.5 - Cover)",
                            "dCor Δ (Stego 1.0 - Cover)",
                            vmin=-absmax_dcor, vmax=absmax_dcor,
                            save_as='dCor_diff_plain.png')


if __name__ == '__main__':
    main()
