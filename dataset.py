"""
dataset.py

Dataset for loading preprocessed CT volumes and segmentation masks saved as .npz
for the Enhanced Conditional Diffusion pipeline.

Each item returns:
    healthy_t     → inpainted slice (healthy CT),   tensor (1,H,W), in [-1,1]
    target_t      → original CT slice,              tensor (1,H,W), in [-1,1]
    tumor_mask_t  → tumor mask (0/1),               tensor (1,H,W)
    liver_mask_t  → liver mask (0/1),               tensor (1,H,W)
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
from typing import List, Tuple


# ---------------------------------------------------------
# Build slice entries for dataset loading
# ---------------------------------------------------------
def build_slice_entries_for_pairs(pairs):
    """
    pairs = [(inpainted_npz, original_npz, mask_npz), ...]

    Returns list of dict entries:
        {
            "inpaint": path_to_npz,
            "orig":    path_to_npz,
            "mask":    path_to_npz,
            "slice":   slice_index
        }
    """

    entries = []

    for inpaint_p, orig_p, mask_p in pairs:

        # Load shape from the inpainted file
        inpaint_vol = np.load(inpaint_p)["inpainted"]
        Z = inpaint_vol.shape[0]

        for s in range(Z):
            entries.append({
                "inpaint": inpaint_p,
                "orig": orig_p,
                "mask": mask_p,
                "slice": s
            })

    return entries


# ---------------------------------------------------------
# Utility: enforce (1,H,W) shape on tensors
# ---------------------------------------------------------
def ensure_channel_first(x, force_size):
    """
    Input x can be:
        (H,W), (1,H,W), (1,1,H,W)

    Output → always shape (1, H, W)
    """

    x = np.asarray(x)

    # Remove ALL singleton dims first → (H,W) or (C,H,W)
    x = np.squeeze(x)

    # After squeeze:
    #   Case A: (H,W)
    #   Case B: (H,W,C) ← shouldn't happen, but guard anyway
    #   Case C: (C,H,W)

    # If (H,W) only → add channel dim
    if x.ndim == 2:
        x = x[np.newaxis, ...]    # (1,H,W)

    # If somehow (H,W,C) → transpose to (C,H,W)
    elif x.ndim == 3 and x.shape[0] != 1:
        # weird case: last channel → move to first
        x = np.moveaxis(x, -1, 0)

    # Now ensure first dim is 1
    if x.shape[0] != 1:
        x = x[0:1, :, :]

    # Resize safely to target shape (1,H,W)
    H, W = force_size
    x_resized = np.zeros((1, H, W), dtype=x.dtype)
    x_resized[0] = cv2.resize(x[0], (W, H), interpolation=cv2.INTER_LINEAR)

    return x_resized


# ---------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------
class CTNPZDataset(Dataset):
    def __init__(
        self,
        entries: List[Tuple[str, str, str, int]],
        preprocess_fn,
        clip_min: int = -100,
        clip_max: int = 300,
        force_size: Tuple[int, int] = (256, 256),
        augment_fn=None,
        aug_prob: float = 0.3,
    ):
        self.entries = entries
        self.preprocess_fn = preprocess_fn

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.force_size = force_size
        self.augment_fn = augment_fn
        self.aug_prob = aug_prob

        if len(self.entries) == 0:
            raise RuntimeError("No entries provided to CTNPZDataset.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        entry = self.entries[idx]
        s = entry["slice"]

        # ---------------------------
        # Load NPZ files
        # ---------------------------
        inpaint_vol = np.load(entry["inpaint"])["inpainted"].astype(np.float32)
        orig_vol    = np.load(entry["orig"])["image"].astype(np.float32)
        seg_vol     = np.load(entry["mask"])["mask"].astype(np.float32)

        # Extract slices
        healthy_slice = inpaint_vol[s]
        target_slice  = orig_vol[s]
        seg_slice     = seg_vol[s]

        # ---------------------------
        # Construct masks
        # ---------------------------
        if np.max(seg_slice) >= 2:
            tumor_mask = (seg_slice == 2).astype(np.float32)
            liver_mask = (seg_slice >= 1).astype(np.float32)
        else:
            tumor_mask = (seg_slice > 0).astype(np.float32)
            liver_mask = (seg_slice > 0).astype(np.float32)

        # ---------------------------
        # Preprocess into [-1,1]
        # preprocess_fn expects shape (1,H,W)
        # ---------------------------
        healthy_pp = self.preprocess_fn(
            healthy_slice[np.newaxis, ...], self.clip_min, self.clip_max, self.force_size
        )[0]

        target_pp = self.preprocess_fn(
            target_slice[np.newaxis, ...], self.clip_min, self.clip_max, self.force_size
        )[0]

        # ---------------------------
        # Enforce strict shapes on CT
        # ---------------------------
        healthy_pp = ensure_channel_first(healthy_pp, self.force_size)
        target_pp  = ensure_channel_first(target_pp,  self.force_size)

        # ---------------------------
        # Resize masks and enforce shape
        # ---------------------------
        H, W = self.force_size

        tumor_mask_r = cv2.resize(tumor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        liver_mask_r = cv2.resize(liver_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        tumor_mask_r = ensure_channel_first(tumor_mask_r, self.force_size)
        liver_mask_r = ensure_channel_first(liver_mask_r, self.force_size)

        # ---------------------------
        # Convert to tensors
        # ---------------------------
        healthy_t    = torch.from_numpy(healthy_pp).float()
        target_t     = torch.from_numpy(target_pp).float()
        tumor_mask_t = torch.from_numpy(tumor_mask_r).float()
        liver_mask_t = torch.from_numpy(liver_mask_r).float()

        # ---------------------------
        # Optional augmentation
        # Always maintain (1,H,W)
        # ---------------------------
        if self.augment_fn is not None and random.random() < self.aug_prob:
            healthy_t, liver_mask_t = self.augment_fn(healthy_t, liver_mask_t)
            target_t,  tumor_mask_t = self.augment_fn(target_t,  tumor_mask_t)

            # Re-enforce shape after augmentation
            healthy_t    = healthy_t.unsqueeze(0) if healthy_t.ndim == 2 else healthy_t
            target_t     = target_t.unsqueeze(0) if target_t.ndim == 2 else target_t
            tumor_mask_t = tumor_mask_t.unsqueeze(0) if tumor_mask_t.ndim == 2 else tumor_mask_t
            liver_mask_t = liver_mask_t.unsqueeze(0) if liver_mask_t.ndim == 2 else liver_mask_t

        return healthy_t, target_t, tumor_mask_t, liver_mask_t
