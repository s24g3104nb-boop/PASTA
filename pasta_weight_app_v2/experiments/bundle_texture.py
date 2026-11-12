"""Offline orientation-aligned texture descriptors for visible parallel filaments.

A periodic ridge is not a confirmed pasta strand: reflections may create multiple
ridges per strand, and buried layers are invisible. This module never returns a
strand count or a weight, and never uses labels, filenames or background pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = 'visible-bundle-texture-v1'
TEXTURE_KEYS = [
    'texture_available', 'texture_width_over_pitch',
    'texture_area_over_pitch_length', 'texture_pitch_over_length',
    'texture_periodicity', 'texture_relative_contrast',
    'texture_pitch_cv', 'texture_directional_coherence',
]


def _empty(reason):
    return {**dict.fromkeys(TEXTURE_KEYS, 0.), 'version': VERSION,
            'texture_status': 'unavailable', 'qc_flags': [reason],
            'pitch_px': None, 'profiles': []}


def _longest_run(valid):
    padded = np.r_[False, valid, False].astype(np.int8)
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if not len(starts):
        return slice(0, 0)
    i = int(np.argmax(ends - starts))
    return slice(int(starts[i]), int(ends[i]))


def _period(profile):
    """Fixed spectral test; unavailable when texture is absent or undersampled."""
    n = len(profile)
    if n < 32:
        return None
    x = np.linspace(-1., 1., n)
    residual = profile - np.polyval(np.polyfit(x, profile, 2), x)
    contrast = float(np.std(residual))
    if contrast < 1.5:
        return None
    power = np.abs(np.fft.rfft(residual * np.hanning(n)))**2
    frequencies = np.fft.rfftfreq(n)
    # At least five observed cycles, and >2.8 pixels/cycle to avoid declaring a
    # near-Nyquist artifact as resolved texture. This is an image QC criterion,
    # not a physical diameter constraint.
    eligible = (frequencies >= max(5/n, 1/32)) & (frequencies <= 1/2.8)
    if not eligible.any():
        return None
    indices = np.flatnonzero(eligible)
    peak = int(indices[np.argmax(power[eligible])])
    total = float(power[eligible].sum())
    coherence = float(power[max(1, peak-1):peak+2].sum()/max(total, 1e-12))
    if coherence < .3 or peak == indices[0] or peak == indices[-1]:
        return None
    # Sub-bin interpolation reduces discrete FFT-bin jumps after digital resize.
    values = np.log(np.maximum(power[peak-1:peak+2], 1e-12))
    denominator = values[0] - 2*values[1] + values[2]
    offset = .5*(values[0]-values[2])/denominator if abs(denominator) > 1e-12 else 0.
    offset = float(np.clip(offset, -.5, .5))
    pitch = float(n/(peak+offset))
    return pitch, min(coherence, 1.), contrast/max(float(np.mean(profile)), 1.)


def extract_bundle_texture(rgb, mask):
    """Return fixed finite model features plus private inspection profiles.

    ``mask`` is a candidate foreground, not a hand-approved segmentation. Only
    masked pixels enter intensity profiles. Absolute pitch/length are diagnostics
    and deliberately excluded from TEXTURE_KEYS. Missing texture is explicit;
    model fitting must not turn it into a known physical scale or hidden mass.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image')
    if not isinstance(mask, np.ndarray) or mask.shape != rgb.shape[:2] or min(mask.shape) < 16:
        raise ValueError('Expected same-size two-dimensional foreground mask')
    if mask.dtype != np.uint8 and mask.dtype != np.bool_:
        raise ValueError('Expected uint8/bool foreground mask')
    if max(rgb.shape[:2]) > 1400:
        scale = 1400/max(rgb.shape[:2])
        size = tuple(max(1, round(v*scale)) for v in rgb.shape[1::-1])
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    foreground = (mask > 0).astype(np.uint8)
    y, x = np.nonzero(foreground)
    if len(x) < 128:
        return _empty('insufficient_foreground')
    points = np.column_stack((x, y)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points-center
    values, vectors = np.linalg.eigh(centered.T@centered/len(points))
    if values[1]/max(values[0], 1e-9) < 1.5:
        return _empty('ambiguous_bundle_axis')
    along = vectors[:, 1]
    across = np.array([-along[1], along[0]])
    basis = np.stack((along, across))
    projected = centered@basis.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    dimensions = np.ceil(high-low+3).astype(int)
    if min(dimensions) < 16:
        return _empty('insufficient_resolved_width')
    affine = np.column_stack((basis, -basis@center-low+1))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    aligned = cv2.warpAffine(gray, affine, tuple(dimensions), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(foreground, affine, tuple(dimensions), flags=cv2.INTER_NEAREST)
    # Do not let excluded background enter a valid pixel through bilinear warp.
    coverage = cv2.warpAffine(foreground.astype(np.float32), affine, tuple(dimensions), flags=cv2.INTER_LINEAR)
    support = (support > 0) & (coverage > .999)
    ys, xs = np.nonzero(support)
    if len(xs) < 128:
        return _empty('insufficient_foreground_after_alignment')
    length = max(1., float(np.quantile(xs, .99)-np.quantile(xs, .01)))
    width = max(1., float(np.quantile(ys, .99)-np.quantile(ys, .01)))
    profiles, observations = [], []
    for fraction in np.linspace(.2, .8, 5):
        middle = int(np.quantile(xs, fraction))
        half = max(2, round(.035*length))
        start, end = max(0, middle-half), min(aligned.shape[1], middle+half+1)
        valid = support[:, start:end]
        counts = valid.sum(axis=1)
        enough = counts >= max(2, .6*(end-start))
        run = _longest_run(enough)
        profile = np.divide((aligned[:, start:end]*valid).sum(axis=1), counts,
                            out=np.zeros(len(counts), float), where=counts > 0)[run]
        period = _period(profile)
        profiles.append({'fraction': float(fraction), 'start_y': run.start,
                         'intensity': profile.tolist(), 'pitch_px': period[0] if period else None})
        if period is not None:
            observations.append(period)
    if len(observations) < 3:
        result = _empty('unresolved_or_nonperiodic_visible_texture')
        result['profiles'] = profiles
        return result
    pitches = np.array([row[0] for row in observations])
    pitch = float(np.median(pitches))
    pitch_cv = float(np.std(pitches)/max(np.mean(pitches), 1e-9))
    if pitch_cv > .25:
        result = _empty('inconsistent_texture_spacing')
        result['profiles'] = profiles
        return result
    interior = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dx = cv2.Sobel(aligned, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(aligned, cv2.CV_32F, 0, 1, ksize=3)
    longitudinal = float(np.mean(dx[interior]**2)) if interior.any() else 0.
    transverse = float(np.mean(dy[interior]**2)) if interior.any() else 0.
    features = dict(texture_available=1., texture_width_over_pitch=width/pitch,
                    texture_area_over_pitch_length=float(support.sum())/(pitch*length),
                    texture_pitch_over_length=pitch/length,
                    texture_periodicity=float(np.median([row[1] for row in observations])),
                    texture_relative_contrast=float(np.median([row[2] for row in observations])),
                    texture_pitch_cv=pitch_cv,
                    texture_directional_coherence=transverse/max(transverse+longitudinal, 1e-9))
    if not np.isfinite([features[key] for key in TEXTURE_KEYS]).all():
        raise ValueError('Non-finite texture features')
    return dict(features, version=VERSION, texture_status='visible_periodic_texture',
                qc_flags=['visible_ridges_are_not_confirmed_strands_or_hidden_layers'],
                pitch_px=pitch, visible_length_px=length, visible_width_px=width,
                profiles=profiles)
