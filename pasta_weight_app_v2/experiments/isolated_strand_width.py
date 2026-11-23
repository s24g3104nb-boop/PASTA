"""Measure the apparent width of separated, visible strand segments."""

from dataclasses import dataclass

import cv2
import numpy as np

# Sampling distances at the canonical resolution. These control which edges
# are resolved in the image, without supplying a physical strand diameter.
_PROFILE = np.arange(-12., 12.25, .25, dtype=np.float32)
_MID = len(_PROFILE) // 2
_MIN_WIDTH = 2.5
_TILE = 32


@dataclass(frozen=True)
class WidthMeasurement:
    width_px: float | None = None
    sections: int = 0
    tiles: int = 0
    spread_px: float | None = None


def _profile_edges(samples):
    """Locate the first foreground-to-background transitions on one side."""
    outside = samples < .5
    # Three further samples should also be background, so a tiny hole in the
    # foreground does not become an apparent boundary between two strands.
    first = outside.argmax(axis=1)
    rows = np.arange(len(samples))
    indices = np.minimum(first[:, None] + np.arange(4), samples.shape[1] - 1)
    valid = (first > 0) & (first + 3 < samples.shape[1]) & outside[rows[:, None], indices].all(axis=1)
    previous = samples[rows, np.maximum(0, first - 1)]
    boundary = (first - 1 + (previous - .5) /
                np.maximum(previous - samples[rows, first], 1e-6)) * _PROFILE[_MID + 1]
    return np.where(valid, boundary, np.nan)


def _normal_profiles(mask, x, y, nx, ny):
    """Sample the same width at three neighbouring positions along a strand."""
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    sections = []
    for tangent_step in (-4, 0, 4):
        x0 = xf - tangent_step * ny
        y0 = yf + tangent_step * nx
        xmap = x0[:, None] + nx[:, None] * _PROFILE
        ymap = y0[:, None] + ny[:, None] * _PROFILE
        values = cv2.remap(mask, xmap, ymap, cv2.INTER_LINEAR)
        left = _profile_edges(values[:, _MID::-1])
        right = _profile_edges(values[:, _MID:])
        width = left + right
        centered = np.abs(left - right) <= .45 * width
        sections.append(np.where(centered & (width >= _MIN_WIDTH), width, np.nan))
    return np.stack(sections, axis=1)


def isolated_strand_width(mask, centerline):
    """Use both visible edges of locally separated segments, then pool by tile."""
    neighbors = cv2.filter2D(centerline, -1, np.ones((3, 3), np.uint8)) - centerline
    junctions = ((centerline != 0) & (neighbors > 2)).astype(np.uint8)
    near_junction = cv2.dilate(junctions, np.ones((9, 9), np.uint8)) != 0
    ys, xs = np.nonzero((centerline != 0) & (neighbors == 2) & ~near_junction)
    if not len(xs):
        return WidthMeasurement()
    # The gradient tensor estimates a normal from nearby parallel edges.
    binary = (mask != 0).astype(np.float32)
    gx = cv2.Sobel(binary, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary, cv2.CV_32F, 0, 1, ksize=3)
    xx, yy, xy = [cv2.GaussianBlur(v, (0, 0), 2.5)[ys, xs] for v in (gx*gx, gy*gy, gx*gy)]
    direction = np.hypot(xx - yy, 2*xy) > .8*(xx + yy)
    # Require space within the image for the full cross-section and its neighbors.
    border = int(_PROFILE[-1]) + 5
    inside = ((xs >= border) & (xs < mask.shape[1] - border) &
              (ys >= border) & (ys < mask.shape[0] - border))
    positions = np.flatnonzero(direction & inside)
    positions = positions[::max(4, (len(positions) + 5999)//6000)]
    if not len(positions):
        return WidthMeasurement()
    angle = .5 * np.arctan2(2*xy[positions], xx[positions] - yy[positions])
    xs, ys = xs[positions], ys[positions]
    profiles = _normal_profiles(binary, xs, ys, np.cos(angle), np.sin(angle))
    widths = np.median(profiles, axis=1)
    stable = np.ptp(profiles, axis=1) <= np.maximum(.85, .20*widths)
    if not stable.any():
        return WidthMeasurement()
    widths, xs, ys = widths[stable], xs[stable], ys[stable]
    # Side-by-side strands sharing both outside edges can still pass the local
    # checks. Use the repeatedly observed narrow class instead of the widest
    # sections, without weighting by how many samples a long region supplies.
    narrow = np.quantile(widths, .25)
    close = (widths >= .80*narrow) & (widths <= 1.25*narrow)
    widths, xs, ys = widths[close], xs[close], ys[close]
    tile_index = ys//_TILE * (mask.shape[1]//_TILE + 1) + xs//_TILE
    tiles = np.unique(tile_index)
    if len(tiles) < 3 or len(widths) < 8:
        return WidthMeasurement(sections=len(widths), tiles=len(tiles))
    tile_widths = [np.median(widths[tile_index == t]) for t in tiles]
    p10, middle, p90 = np.quantile(tile_widths, [.1, .5, .9])
    return WidthMeasurement(float(middle), len(widths), len(tiles), float(p90 - p10))
