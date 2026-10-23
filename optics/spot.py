"""Circular-aperture spot measurement (numpy only).

    from spot import measure_spot
    s = measure_spot(img, x=640, y=512, r_ap=15, r_in=20, r_out=30)
    print(s.x, s.y, s.fwhm_radial, s.net_flux, s.ee80)

Background is the median of the annulus r_in..r_out and is subtracted from
every aperture pixel. All lengths are in pixels; x/y are 0-based pixel
centers (same as numpy indexing img[y, x]).
"""
from dataclasses import dataclass, field

import numpy as np

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class SpotResult:
    x: float                 # centroid (or the requested center if recenter=False)
    y: float
    background: float        # median per-pixel level in the annulus
    background_std: float
    net_flux: float          # sum of (pixel - background) inside the aperture
    n_pixels: int            # pixels inside the aperture
    peak: float              # brightest raw pixel inside the aperture
    n_saturated: int         # aperture pixels at or above the saturation value
    sigma_x: float           # background-subtracted second moments
    sigma_y: float
    fwhm_x: float            # 2.355 * sigma, i.e. assuming a Gaussian spot
    fwhm_y: float
    fwhm_fit: float          # circular Gaussian fit to the pixels (best for compact spots)
    fit_amplitude: float     # that Gaussian's peak, background subtracted
    fwhm_radial: float       # where the binned radial profile drops to half its center value
    ee50: float              # radius enclosing 50% / 80% of the net aperture flux
    ee80: float
    # for plotting
    pixel_r: np.ndarray = field(repr=False, default=None)      # every aperture pixel's radius
    pixel_v: np.ndarray = field(repr=False, default=None)      # ...and its background-subtracted value
    profile_r: np.ndarray = field(repr=False, default=None)    # 1-px radial bins
    profile_v: np.ndarray = field(repr=False, default=None)
    ee_r: np.ndarray = field(repr=False, default=None)         # encircled-energy curve
    ee_frac: np.ndarray = field(repr=False, default=None)


def measure_spot(img, x, y, r_ap, r_in=None, r_out=None, recenter=True, saturation=None,
                 max_iter=20) -> SpotResult:
    """Measure the spot in a circle of radius r_ap around (x, y).

    With recenter=True the center moves to the background-subtracted centroid
    inside the aperture, repeated until it settles. Pass r_in/r_out = None to
    skip background subtraction.
    """
    if r_ap <= 0:
        raise ValueError("aperture radius must be > 0")
    use_bkg = r_in is not None and r_out is not None
    if use_bkg and not r_ap <= r_in < r_out:
        raise ValueError("need aperture radius <= annulus inner < annulus outer")

    h, w = img.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("center is outside the image")
    reach = r_out if use_bkg else r_ap

    cx, cy = float(x), float(y)
    for _ in range(max_iter if recenter else 1):
        sub, xx, yy, rr = _cutout(img, cx, cy, reach)
        bkg, bkg_std = _background(sub, rr, r_in, r_out) if use_bkg else (0.0, 0.0)
        ap = rr <= r_ap
        wts = np.clip(sub[ap] - bkg, 0, None)
        if not recenter or wts.sum() <= 0:
            break
        nx, ny = (wts * xx[ap]).sum() / wts.sum(), (wts * yy[ap]).sum() / wts.sum()
        moved = np.hypot(nx - cx, ny - cy)
        cx, cy = float(nx), float(ny)
        if moved < 0.01:
            break
    # final measurement at the settled center
    sub, xx, yy, rr = _cutout(img, cx, cy, reach)
    bkg, bkg_std = _background(sub, rr, r_in, r_out) if use_bkg else (0.0, 0.0)
    ap = rr <= r_ap

    raw = sub[ap]
    val = raw - bkg
    r = rr[ap]
    net = float(val.sum())

    wts = np.clip(val, 0, None)
    if wts.sum() > 0:
        sx = float(np.sqrt((wts * (xx[ap] - cx) ** 2).sum() / wts.sum()))
        sy = float(np.sqrt((wts * (yy[ap] - cy) ** 2).sum() / wts.sum()))
    else:
        sx = sy = float("nan")

    prof_r, prof_v = _radial_profile(r, val, r_ap)
    fwhm_fit, fit_amp = _gaussian_fit(r, val)
    order = np.argsort(r)
    ee_r = r[order]
    ee_frac = np.cumsum(val[order]) / net if net > 0 else np.full(len(order), np.nan)

    return SpotResult(
        x=cx, y=cy, background=float(bkg), background_std=float(bkg_std),
        net_flux=net, n_pixels=int(ap.sum()), peak=float(raw.max()),
        n_saturated=int((raw >= saturation).sum()) if saturation is not None else 0,
        sigma_x=sx, sigma_y=sy, fwhm_x=sx * SIGMA_TO_FWHM, fwhm_y=sy * SIGMA_TO_FWHM,
        fwhm_fit=fwhm_fit, fit_amplitude=fit_amp,
        fwhm_radial=_half_max_diameter(prof_r, prof_v),
        ee50=_first_crossing(ee_r, ee_frac, 0.5), ee80=_first_crossing(ee_r, ee_frac, 0.8),
        pixel_r=r, pixel_v=val, profile_r=prof_r, profile_v=prof_v, ee_r=ee_r, ee_frac=ee_frac,
    )


def _cutout(img, cx, cy, reach):
    h, w = img.shape
    x0, x1 = max(int(np.floor(cx - reach)), 0), min(int(np.ceil(cx + reach)) + 1, w)
    y0, y1 = max(int(np.floor(cy - reach)), 0), min(int(np.ceil(cy + reach)) + 1, h)
    sub = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return sub, xx, yy, np.hypot(xx - cx, yy - cy)


def _background(sub, rr, r_in, r_out):
    ring = sub[(rr >= r_in) & (rr <= r_out)]
    if ring.size == 0:
        return 0.0, 0.0
    return float(np.median(ring)), float(ring.std())


def _radial_profile(r, val, r_ap):
    edges = np.arange(0.0, np.ceil(r_ap) + 1.0)
    idx = np.digitize(r, edges) - 1
    centers, means = [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.any():
            centers.append(r[m].mean())
            means.append(val[m].mean())
    return np.array(centers), np.array(means)


def _gaussian_fit(r, val):
    """Fit v = A exp(-r^2 / 2 sigma^2) to the pixels above 20% of the peak, as a
    straight line in (r^2, ln v), weighted by v^2 to undo the log's noise blow-up."""
    nan = (float("nan"), float("nan"))
    if len(val) < 3 or val.max() <= 0:
        return nan
    use = val > 0.2 * val.max()
    if use.sum() < 3:
        return nan
    x, y, wt = r[use] ** 2, np.log(val[use]), val[use]
    slope, intercept = np.polyfit(x, y, 1, w=wt)
    if slope >= 0:
        return nan
    return float(SIGMA_TO_FWHM * np.sqrt(-1.0 / (2.0 * slope))), float(np.exp(intercept))


def _half_max_diameter(prof_r, prof_v):
    if len(prof_v) < 2 or prof_v[0] <= 0:
        return float("nan")
    half = prof_v[0] / 2.0
    below = np.nonzero(prof_v < half)[0]
    if len(below) == 0:
        return float("nan")  # never drops to half max inside the aperture
    i = below[0]
    r0, r1, v0, v1 = prof_r[i - 1], prof_r[i], prof_v[i - 1], prof_v[i]
    return float(2.0 * (r0 + (v0 - half) * (r1 - r0) / (v0 - v1)))


def _first_crossing(r, frac, level):
    hit = np.nonzero(frac >= level)[0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(r[0])
    return float(np.interp(level, [frac[i - 1], frac[i]], [r[i - 1], r[i]]))
