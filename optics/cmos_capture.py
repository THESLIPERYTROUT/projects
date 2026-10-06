import os
import sys
import time

from ueye_cam import UEyeCamera

N_DARK_FRAMES = 20
N_FLAT_FRAMES = 20
FLAT_EXPOSURE_MS = 0.5  # ~half full-scale with the diffuser; matches a dark exposure

EXPOSURES_MS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]


def capture_light_sweep(cam, exposures, out_dir):
    print(f"saving light frames to {out_dir}")
    for exposure_ms in exposures:
        applied = cam.set_exposure_ms(exposure_ms)
        print(f"Requested {exposure_ms} ms, applied {applied:.6f} ms")

        frame = cam.snap()
        frame.save_fits(os.path.join(out_dir, f"exposure_{exposure_ms:.3f}ms.fits"), imagetyp="LIGHT")
        print("  max pixel:", int(frame.data.max()), " saturated px:", frame.n_saturated)

        time.sleep(1)


def capture_series(cam, exposure_ms, n_frames, out_dir, prefix, imagetyp):
    """n frames at one exposure, named <prefix>_<exp>ms_frameNN.fits (the
    pattern build_masters.py groups on)."""
    applied = cam.set_exposure_ms(exposure_ms)
    print(f"{imagetyp} exposure requested {exposure_ms} ms, applied {applied:.6f} ms")

    for frame in cam.burst(n_frames):
        filename = f"{prefix}_{exposure_ms:.3f}ms_frame{frame.index:02d}.fits"
        frame.save_fits(os.path.join(out_dir, filename), imagetyp=imagetyp)
        print(f"  frame {frame.index:02d}: mean {frame.data.mean():.1f}  max {int(frame.data.max())}  "
              f"saturated px {frame.n_saturated}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "light"
    if mode not in ("light", "darks", "flats"):
        print(f'Unknown mode "{mode}". Use "light", "darks", or "flats [exposure_ms]".')
        sys.exit(1)

    script_dir = os.path.dirname(__file__)

    with UEyeCamera() as cam:
        if mode == "darks":
            darks_dir = os.path.join(script_dir, "darks")
            os.makedirs(darks_dir, exist_ok=True)
            print(f"saving {N_DARK_FRAMES} dark frames per exposure to {darks_dir}")
            for exposure_ms in EXPOSURES_MS:
                capture_series(cam, exposure_ms, N_DARK_FRAMES, darks_dir, "dark", "DARK")
        elif mode == "flats":
            exposure_ms = float(sys.argv[2]) if len(sys.argv) > 2 else FLAT_EXPOSURE_MS
            flats_dir = os.path.join(script_dir, "flats")
            os.makedirs(flats_dir, exist_ok=True)
            print(f"saving {N_FLAT_FRAMES} flat frames to {flats_dir}")
            capture_series(cam, exposure_ms, N_FLAT_FRAMES, flats_dir, "flat", "FLAT")
        else:
            out_dir = os.path.join(script_dir, "output")
            os.makedirs(out_dir, exist_ok=True)
            capture_light_sweep(cam, EXPOSURES_MS, out_dir)


if __name__ == "__main__":
    main()
