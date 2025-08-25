#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import nibabel as nib
import nibabel.freesurfer as fs
from glob import glob

# ----------------------------- #
# Hardcoded configuration paths #
# ----------------------------- #
LABEL_DIR   = "/mnt/mahdipou/nsd/freesurfer"  # contains .annot and visual_mask_*.npy
PPDATA_ROOT = "/mnt/akgokce/datasets/neural/nsd/nsddata_betas/ppdata"
OUT_ROOT    = "/mnt/mahdipou/nsd/visual_betas"

# Subjects to process; set to None to auto-discover subj* under PPDATA_ROOT
SUBJECTS = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]

# fsaverage vertices per hemisphere
HEMI_VERTS = 163842

# Input file extension for betas; NSD fsaverage betas are .mgh in your tree
EXT = ".mgh"

# If masks are missing, try to build from .annot in the same directory:
PREFER_A2009S = True  # use aparc.a2009s if available; otherwise fall back to aparc


# ----------------------------- #
# Utilities                     #
# ----------------------------- #

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def parse_session_id(filename):
    m = re.search(r"session(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else None

def load_mgh_trialsxverts(path, hemi_expected_verts):
    """
    Load MGH/MGZ and return array shaped as [n_trials, n_verts_hemi].
    """
    img = nib.load(path)
    X = np.asarray(img.get_fdata()).squeeze()

    if X.ndim == 1:
        # Single-trial vector
        return X.reshape(1, -1)

    if X.ndim == 2:
        r, c = X.shape
        if r == hemi_expected_verts:  # [verts, trials]
            return X.T
        if c == hemi_expected_verts:  # [trials, verts]
            return X
        raise ValueError(f"{path}: shape {X.shape} does not match expected hemi verts {hemi_expected_verts}")

    if X.ndim == 3:
        # Common: [verts, 1, trials] or [verts, trials, 1]
        axes = list(X.shape)
        if hemi_expected_verts in axes:
            v_axis = axes.index(hemi_expected_verts)
        else:
            v_axis = int(np.argmax(axes))
        Xr = np.moveaxis(X, v_axis, -1)   # verts -> last
        n_verts = Xr.shape[-1]
        trials = int(np.prod(Xr.shape[:-1]))
        return Xr.reshape(trials, n_verts)

    raise ValueError(f"{path}: unexpected ndim={X.ndim}")

def try_load_or_build_masks(label_dir, prefer_a2009s=True):
    """
    Load visual masks if present; otherwise build from .annot in label_dir.
    """
    lh_mask_path = os.path.join(label_dir, "visual_mask_lh.npy")
    rh_mask_path = os.path.join(label_dir, "visual_mask_rh.npy")

    if os.path.exists(lh_mask_path) and os.path.exists(rh_mask_path):
        lh_mask = np.load(lh_mask_path).astype(bool)
        rh_mask = np.load(rh_mask_path).astype(bool)
        if lh_mask.size != HEMI_VERTS or rh_mask.size != HEMI_VERTS:
            raise ValueError("Existing masks have unexpected length.")
        print("Loaded existing masks from:", label_dir)
        return lh_mask, rh_mask

    # Build from annot
    lh_a2009s = os.path.join(label_dir, "lh.aparc.a2009s.annot")
    rh_a2009s = os.path.join(label_dir, "rh.aparc.a2009s.annot")
    lh_aparc  = os.path.join(label_dir, "lh.aparc.annot")
    rh_aparc  = os.path.join(label_dir, "rh.aparc.annot")

    if prefer_a2009s and os.path.exists(lh_a2009s) and os.path.exists(rh_a2009s):
        atlas = "a2009s"
        lh_ann, rh_ann = lh_a2009s, rh_a2009s
    elif os.path.exists(lh_aparc) and os.path.exists(rh_aparc):
        atlas = "aparc"
        lh_ann, rh_ann = lh_aparc, rh_aparc
    else:
        raise FileNotFoundError("No suitable .annot files found to build masks.")

    print(f"Building masks from atlas: {atlas}")

    def read_annot(path):
        labels, ctab, names = fs.read_annot(path)
        names = np.array(names)  # list[bytes] -> array[bytes]
        return labels, names

    def make_mask_hemisphere(annot_path, atlas_name):
        labels, names = read_annot(annot_path)
        if atlas_name == "aparc":
            exact = {b"pericalcarine", b"cuneus", b"lingual", b"lateraloccipital"}
            optional = {b"fusiform"}
            good = np.isin(names, list(exact) + list(optional))
            good_idx = np.where(good)[0]
            hemi_mask = np.isin(labels, good_idx)
        else:
            substrs = [b"occipital", b"calcarine", b"cuneus", b"lingual", b"occipitotemporal", b"fusiform"]
            def is_visual(name):
                nl = name.lower()
                return any(sub in nl for sub in substrs)
            good = np.array([is_visual(nm) for nm in names])
            good_idx = np.where(good)[0]
            hemi_mask = np.isin(labels, good_idx)
        return hemi_mask.astype(bool)

    lh_mask = make_mask_hemisphere(lh_ann, atlas)
    rh_mask = make_mask_hemisphere(rh_ann, atlas)

    # Persist for reuse
    np.save(os.path.join(label_dir, "visual_mask_lh.npy"), lh_mask)
    np.save(os.path.join(label_dir, "visual_mask_rh.npy"), rh_mask)
    np.save(os.path.join(label_dir, "visual_mask_fsaverage.npy"), np.concatenate([lh_mask, rh_mask]).astype(bool))
    print("Saved masks to:", label_dir)
    return lh_mask, rh_mask

def discover_subject_fsavg_dirs(ppdata_root, subjects=None):
    if subjects is not None:
        return [os.path.join(ppdata_root, s, "fsaverage") for s in subjects]
    fsavg_dirs = []
    for name in sorted(os.listdir(ppdata_root)):
        if re.match(r"subj\d+", name):
            d = os.path.join(ppdata_root, name, "fsaverage")
            if os.path.isdir(d):
                fsavg_dirs.append(d)
    return fsavg_dirs

def build_subject_visual_matrix(fsavg_dir, lh_mask, rh_mask, out_path, meta_path, ext=EXT):
    """
    Build a [total_trials, n_visual_vertices_total] matrix for a subject and save as .npy (memmap).
    """
    betas_dir = os.path.join(fsavg_dir, "betas_fithrf")
    if not os.path.isdir(betas_dir):
        raise FileNotFoundError(f"betas_fithrf not found under {fsavg_dir}")

    lh_files = sorted(glob(os.path.join(betas_dir, f"lh.betas_session*{ext}")))
    rh_files = sorted(glob(os.path.join(betas_dir, f"rh.betas_session*{ext}")))
    if not lh_files or not rh_files:
        raise FileNotFoundError(f"No LH/RH session files in {betas_dir}")

    lh_map = {parse_session_id(f): f for f in lh_files if parse_session_id(f) is not None}
    rh_map = {parse_session_id(f): f for f in rh_files if parse_session_id(f) is not None}
    ses_ids = sorted(set(lh_map).intersection(rh_map))
    if not ses_ids:
        raise RuntimeError("No matching LH/RH session IDs found.")

    if lh_mask.size != HEMI_VERTS or rh_mask.size != HEMI_VERTS:
        raise ValueError("Mask length does not match HEMI_VERTS.")

    vis_lh_idx = np.where(lh_mask)[0]
    vis_rh_idx = np.where(rh_mask)[0]
    n_vis_total = vis_lh_idx.size + vis_rh_idx.size

    # First pass: count total trials
    session_trial_counts = {}
    total_trials = 0
    for sid in ses_ids:
        X_lh = load_mgh_trialsxverts(lh_map[sid], hemi_expected_verts=HEMI_VERTS)
        n_tr = X_lh.shape[0]
        session_trial_counts[sid] = n_tr
        total_trials += n_tr

    # Allocate output memmap
    out_mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(total_trials, n_vis_total))

    # Second pass: fill data
    row0 = 0
    session_row_ranges = {}
    for sid in ses_ids:
        X_lh = load_mgh_trialsxverts(lh_map[sid], hemi_expected_verts=HEMI_VERTS)
        X_rh = load_mgh_trialsxverts(rh_map[sid], hemi_expected_verts=HEMI_VERTS)
        if X_lh.shape[0] != X_rh.shape[0]:
            raise ValueError(f"Trial mismatch in session {sid}: LH {X_lh.shape[0]} vs RH {X_rh.shape[0]}")

        X_vis = np.concatenate([X_lh[:, vis_lh_idx], X_rh[:, vis_rh_idx]], axis=1)  # [trials, vis_lh+vis_rh]
        n_tr = X_vis.shape[0]
        out_mm[row0:row0 + n_tr, :] = X_vis
        session_row_ranges[sid] = [row0, row0 + n_tr]  # half-open
        row0 += n_tr
        print(f"{os.path.basename(os.path.dirname(fsavg_dir))} s{sid:02d} -> rows {session_row_ranges[sid]}")

    # Flush memmap
    del out_mm

    # Save metadata
    meta = {
        "subject": os.path.basename(os.path.dirname(fsavg_dir)),
        "fsavg_dir": fsavg_dir,
        "betas_dir": betas_dir,
        "hemi_verts": HEMI_VERTS,
        "n_visual_lh": int(vis_lh_idx.size),
        "n_visual_rh": int(vis_rh_idx.size),
        "n_visual_total": int(n_vis_total),
        "total_trials": int(total_trials),
        "feature_order": "columns = [LH_visual_vertices..., RH_visual_vertices...]",
        "sessions": {
            f"{sid:02d}": {
                "trial_count": int(session_trial_counts[sid]),
                "row_range": session_row_ranges[sid]
            } for sid in ses_ids
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved: {meta_path}")


# ----------------------------- #
# Main                          #
# ----------------------------- #
if __name__ == "__main__":
    ensure_outdir(OUT_ROOT)

    # Load or build masks from LABEL_DIR
    lh_mask, rh_mask = try_load_or_build_masks(LABEL_DIR, prefer_a2009s=PREFER_A2009S)

    # Discover subjects
    if SUBJECTS is None:
        subj_fsavg_dirs = discover_subject_fsavg_dirs(PPDATA_ROOT, subjects=None)
    else:
        subj_fsavg_dirs = discover_subject_fsavg_dirs(PPDATA_ROOT, subjects=SUBJECTS)

    if not subj_fsavg_dirs:
        raise RuntimeError("No subject fsaverage directories found.")

    # Process each subject
    for fsavg_dir in subj_fsavg_dirs:
        subj = os.path.basename(os.path.dirname(fsavg_dir))
        out_npy  = os.path.join(OUT_ROOT, f"{subj}_betas_visual_only.npy")
        out_meta = os.path.join(OUT_ROOT, f"{subj}_betas_visual_only.meta.json")
        build_subject_visual_matrix(
            fsavg_dir, lh_mask, rh_mask, out_npy, out_meta, ext=EXT
        )
