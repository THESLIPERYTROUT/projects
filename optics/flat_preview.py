"""Snap one frame and show its image, histogram, and center row/column
intensity profiles -- for judging how uniform an improvised flat-field
diffuser setup looks before committing to a real flat capture.

Usage:
    python flat_preview.py [exposure_ms]

Default exposure is 1.0 ms if not given. Each run also saves the raw frame
(FITS) and the diagnostic figure (PNG) to flats/preview/ so you can compare
successive diffuser adjustments.
"""
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ueye_cam import UEyeCamera


def make_figure(frame: np.ndarray, exposure_ms: float, vmax: int):
    h, w = frame.shape
    fig, ((ax_img, ax_hist), (ax_row, ax_col)) = plt.subplots(2, 2, figsize=(10, 8))

    ax_img.imshow(frame, cmap="gray", vmin=0, vmax=vmax)
    ax_img.set_title("frame")
    ax_img.axis("off")

    ax_hist.hist(frame.ravel(), bins=64, range=(0, vmax), color="steelblue")
    ax_hist.set_title("pixel histogram")
    ax_hist.set_xlabel("pixel value")
    ax_hist.set_ylabel("count")
    ax_hist.set_xlim(0, vmax)

    ax_row.plot(frame[h // 2, :])
    ax_row.set_ylim(0, vmax)
    ax_row.set_title("center row profile")
    ax_row.set_xlabel("x (px)")
    ax_row.set_ylabel("pixel value")

    ax_col.plot(frame[:, w // 2])
    ax_col.set_ylim(0, vmax)
    ax_col.set_title("center column profile")
    ax_col.set_xlabel("y (px)")
    ax_col.set_ylabel("pixel value")

    mean = frame.mean()
    std = frame.std()
    flatness_pct = (std / mean * 100) if mean > 0 else float("inf")
    sat_px = int((frame >= vmax).sum())
    center = int(frame[h // 2, w // 2])
    corner = int(min(frame[0, 0], frame[0, -1], frame[-1, 0], frame[-1, -1]))

    fig.suptitle(
        f"exposure={exposure_ms:.4f}ms  mean={mean:.1f}  std={std:.1f} ({flatness_pct:.1f}% of mean)\n"
        f"min={int(frame.min())}  max={int(frame.max())}  saturated={sat_px}px  "
        f"center={center}  corner(min)={corner}"
    )
    fig.tight_layout()
    return fig


def main():
    exposure_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    script_dir = os.path.dirname(__file__)
    preview_dir = os.path.join(script_dir, "flats", "preview")
    os.makedirs(preview_dir, exist_ok=True)

    with UEyeCamera() as cam:
        # Warns (ExposureClampedWarning) if the camera can't do the requested
        # exposure -- note the argument is in ms, e.g. 1000 for 1 s.
        applied = cam.set_exposure_ms(exposure_ms)
        print(f"Requested {exposure_ms} ms, applied {applied:.6f} ms")
        frame = cam.snap()

    data = frame.data
    mean = data.mean()
    std = data.std()
    flatness_pct = (std / mean * 100) if mean > 0 else float("inf")
    print(f"mean={mean:.1f}  std={std:.1f}  flatness={flatness_pct:.1f}% of mean  "
          f"min={int(data.min())}  max={int(data.max())}  saturated={frame.n_saturated}px")
    print("(lower flatness % = more uniform; compare center vs corner in the plot title "
          "to spot vignetting/falloff)")

    fig = make_figure(data, applied, frame.saturation_value)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame.save_fits(os.path.join(preview_dir, f"preview_{stamp}.fits"), imagetyp="FLAT-PREVIEW")
    fig.savefig(os.path.join(preview_dir, f"preview_{stamp}.png"), dpi=120)
    print(f"saved preview_{stamp}.fits / .png to {preview_dir}")

    plt.show()


if __name__ == "__main__":
    main()
