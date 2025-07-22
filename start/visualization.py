import nibabel as nib
from nilearn import plotting
import webbrowser

# Paths to your beta data files (change to your actual paths)
lh_beta_path = "lh.betas_session01.mgh"
rh_beta_path = "rh.betas_session01.mgh"

# Paths to fsaverage inflated surfaces
lh_surf = "/Applications/freesurfer/8.0.0/subjects/fsaverage/surf/lh.inflated"
rh_surf = "/Applications/freesurfer/8.0.0/subjects/fsaverage/surf/rh.inflated"

# Load beta data (e.g., trial 10, zero-based index 9)
lh_img = nib.load(lh_beta_path)
rh_img = nib.load(rh_beta_path)

lh_data = lh_img.get_fdata()[:, 0, 0, 9]
rh_data = rh_img.get_fdata()[:, 0, 0, 9]

# Create interactive surface plots
view_lh = plotting.view_surf(
    lh_surf, lh_data,
    hemi='left',
    title='Left Hemisphere - Trial 10',
    darkness=None
)

view_rh = plotting.view_surf(
    rh_surf, rh_data,
    hemi='right',
    title='Right Hemisphere - Trial 10',
    darkness=None
)

# Save interactive plots as HTML
html_lh = "lh_trial10.html"
html_rh = "rh_trial10.html"

view_lh.save_as_html(html_lh)
view_rh.save_as_html(html_rh)

print(f"Saved Left Hemisphere plot to {html_lh}")
print(f"Saved Right Hemisphere plot to {html_rh}")

# Open both HTML files in your default web browser
webbrowser.open(html_lh)
webbrowser.open(html_rh)
