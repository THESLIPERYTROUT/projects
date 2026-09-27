"""Median-combine raw darks and flats into master calibration frames.

Usage:
    python build_masters.py

Writes a master dark for every exposure found in darks/ and a normalized master
flat for every exposure found in flats/. Each flat is dark-subtracted with the
master dark of the same exposure, then divided by its mean so it's ~1.0.
"""
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

EXPOSURE_RE = re.compile(r"_(\d+\.\d+)ms_frame\d+\.fits$")


def group_by_exposure(pattern: str) -> dict:
    groups = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        m = EXPOSURE_RE.search(path)
        if m:
            groups[m.group(1)].append(path)
    return groups


def median_stack(paths):
    stack = np.stack([fits.getdata(p).astype(np.float32) for p in paths])
    return np.median(stack, axis=0), fits.getheader(paths[0])


def write_master(path, data, src_header, imagetyp, n_combined, extra=None):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    for key in ("EXPTIME", "CAMERA", "SERIALNO"):
        if key in src_header:
            hdu.header[key] = src_header[key]
    hdu.header["IMAGETYP"] = imagetyp
    hdu.header["NCOMBINE"] = (n_combined, "Number of frames median-combined")
    hdu.header["DATE"] = datetime.now(timezone.utc).isoformat()
    if extra:
        for key, value in extra.items():
            hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def main():
    script_dir = os.path.dirname(__file__)
    darks_dir = os.path.join(script_dir, "darks")
    flats_dir = os.path.join(script_dir, "flats")

    master_darks = {}
    for exposure, paths in group_by_exposure(os.path.join(darks_dir, "dark_*.fits")).items():
        dark, header = median_stack(paths)
        master_darks[exposure] = dark
        out_path = os.path.join(darks_dir, f"master_dark_{exposure}ms.fits")
        write_master(out_path, dark, header, "MASTER DARK", len(paths))
        print(f"master dark {exposure} ms: n={len(paths)}  mean {dark.mean():.3f}  max {dark.max():.1f}")

    for exposure, paths in group_by_exposure(os.path.join(flats_dir, "flat_*.fits")).items():
        if exposure not in master_darks:
            print(f"skipping flats at {exposure} ms: no master dark at that exposure")
            continue
        flat, header = median_stack(paths)
        flat -= master_darks[exposure]
        norm = float(flat.mean())
        flat /= norm
        out_path = os.path.join(flats_dir, f"master_flat_{exposure}ms.fits")
        write_master(out_path, flat, header, "MASTER FLAT", len(paths),
                     extra={"FLATNORM": (norm, "Dark-subtracted mean used to normalize")})
        print(f"master flat {exposure} ms: n={len(paths)}  norm {norm:.2f}  "
              f"min {flat.min():.3f}  max {flat.max():.3f}  std {flat.std():.4f}")


if __name__ == "__main__":
    main()
