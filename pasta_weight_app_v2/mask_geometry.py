"""Projected dry-pasta geometry from a binary mask. Ratios carry no physical scale."""
import numpy as np


def mask_geometry(mask, *, quantile_low=.01, quantile_high=.99):
    """Robust PCA projections; no absolute-pixel predictor, skeleton or texture."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area = len(xs)
    features = dict(mask_area_px=area, mask_fraction=area/mask.size,
                    image_pixels=int(mask.size), background_fraction=1-area/mask.size)
    length = width = density = cv = 0.
    profile = [0.] * 20
    if area > 2:
        pts = np.column_stack((xs, ys)).astype(float)
        pts -= pts.mean(axis=0)
        _, axes = np.linalg.eigh(pts.T @ pts / area)
        along = pts @ axes[:, -1]
        across = pts @ axes[:, 0]
        lo, hi = np.quantile(along, [quantile_low, quantile_high])
        low, high = np.quantile(across, [quantile_low, quantile_high])
        length, width = max(hi-lo, 1.), max(high-low, 1.)
        density = float(np.count_nonzero((along >= lo) & (along <= hi) & (across >= low) & (across <= high)) / (length*width))
        bins = np.clip(((along-lo)/length*20).astype(int), 0, 19)
        for i in range(20):
            v = across[bins == i]
            if len(v) > 2:
                profile[i] = float(np.quantile(v, .95) - np.quantile(v, .05))
        cv = float(np.std(profile) / max(np.mean(profile), 1.))
    features.update(quantile_length_px=length, quantile_width_px=width,
                    quantile_length_relative=length/max(h, w), quantile_width_relative=width/max(h, w),
                    quantile_density=density, profile_width_cv=cv,
                    area_over_length_squared=area/max(length*length, 1.), width_over_length=width/max(length, 1.),
                    projection_widths_px=profile)
    return features
