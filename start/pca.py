import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA

# Load your data paths
lh_beta_path = "lh.betas_session01.mgh"
rh_beta_path = "rh.betas_session01.mgh"

# Load data
lh_img = nib.load(lh_beta_path)
rh_img = nib.load(rh_beta_path)

# Extract data: shape (vertices, 1, 1, trials)
lh_data_4d = lh_img.get_fdata()
rh_data_4d = rh_img.get_fdata()

# Reshape to (vertices, trials)
lh_data = lh_data_4d[:, 0, 0, :]  # shape (vertices, trials)
rh_data = rh_data_4d[:, 0, 0, :]

print("LH data shape:", lh_data.shape)
print("RH data shape:", rh_data.shape)

# Apply PCA: keep top 5 components for example
pca_lh = PCA(n_components=5)
pca_rh = PCA(n_components=5)

# Fit PCA on the data (vertices x trials)
# We transpose so PCA is done across trials
pca_lh.fit(lh_data.T)
pca_rh.fit(rh_data.T)

# Transform data: principal components are in trial space
# To get spatial maps (per vertex), get components_
# Components shape: (n_components, vertices)
components_lh = pca_lh.components_  # shape (5, vertices)
components_rh = pca_rh.components_

print("LH PCA components shape:", components_lh.shape)

# To visualize a component spatially, transpose it to (vertices,)
comp_idx = 0  # first principal component

lh_pc_map = components_lh[comp_idx, :]
rh_pc_map = components_rh[comp_idx, :]

# Now visualize with nilearn
from nilearn import plotting

lh_surf = "/Applications/freesurfer/8.0.0/subjects/fsaverage/surf/lh.inflated"
rh_surf = "/Applications/freesurfer/8.0.0/subjects/fsaverage/surf/rh.inflated"

view_lh = plotting.view_surf(
    lh_surf, lh_pc_map,
    hemi='left',
    title='LH PCA Component 1',
    darkness=None
)

view_rh = plotting.view_surf(
    rh_surf, rh_pc_map,
    hemi='right',
    title='RH PCA Component 1',
    darkness=None
)

view_lh.save_as_html("lh_pca_comp1.html")
view_rh.save_as_html("rh_pca_comp1.html")
print("Saved PCA component visualizations as HTML files.")
