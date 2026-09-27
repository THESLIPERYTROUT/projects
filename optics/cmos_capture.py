import os
import sys
import time

from ueye_common import open_camera, save_fits

N_DARK_FRAMES = 20
N_FLAT_FRAMES = 20
FLAT_EXPOSURE_MS = 0.5  # ~half full-scale with the diffuser; matches a dark exposure

EXPOSURES_MS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]


def capture_light_sweep(cam, exposures, out_dir):
    print(f"saving light frames to {out_dir}")
    for exposure_ms in exposures:
        cam.set_exposure_ms(exposure_ms)
        applied = cam.get_exposure_ms()
        print(f"Requested {exposure_ms} ms, applied {applied:.6f} ms")

        cam.grab()  # flush
        frame = cam.grab()

        filename = f"exposure_{exposure_ms:.3f}ms.fits"
        out_path = os.path.join(out_dir, filename)
        save_fits(out_path, frame, applied, "LIGHT", cam.info)

        mx = int(frame.max())
        sat = int((frame == 255).sum())
        print("  max pixel:", mx, " saturated px:", sat)

        time.sleep(1)


def capture_darks(cam, exposures, darks_dir, n_frames):
    print(f"saving {n_frames} dark frames per exposure to {darks_dir}")
    for exposure_ms in exposures:
        cam.set_exposure_ms(exposure_ms)
        applied = cam.get_exposure_ms()
        print(f"Dark exposure requested {exposure_ms} ms, applied {applied:.6f} ms")

        cam.grab()  # flush

        for frame_idx in range(n_frames):
            frame = cam.grab()
            filename = f"dark_{exposure_ms:.3f}ms_frame{frame_idx:02d}.fits"
            out_path = os.path.join(darks_dir, filename)
            save_fits(out_path, frame, applied, "DARK", cam.info, extra={"FRAMENUM": frame_idx})

        print(f"  saved {n_frames} frames")


def capture_flats(cam, exposure_ms, flats_dir, n_frames):
    print(f"saving {n_frames} flat frames to {flats_dir}")
    cam.set_exposure_ms(exposure_ms)
    applied = cam.get_exposure_ms()
    print(f"Flat exposure requested {exposure_ms} ms, applied {applied:.6f} ms")

    cam.grab()  # flush

    for frame_idx in range(n_frames):
        frame = cam.grab()
        filename = f"flat_{exposure_ms:.3f}ms_frame{frame_idx:02d}.fits"
        out_path = os.path.join(flats_dir, filename)
        save_fits(out_path, frame, applied, "FLAT", cam.info, extra={"FRAMENUM": frame_idx})
        print(f"  frame {frame_idx:02d}: mean {frame.mean():.1f}  max {int(frame.max())}  "
              f"saturated px {int((frame == 255).sum())}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "light"
    if mode not in ("light", "darks", "flats"):
        print(f'Unknown mode "{mode}". Use "light", "darks", or "flats [exposure_ms]".')
        sys.exit(1)

    script_dir = os.path.dirname(__file__)

    with open_camera() as cam:
        if mode == "darks":
            darks_dir = os.path.join(script_dir, "darks")
            os.makedirs(darks_dir, exist_ok=True)
            capture_darks(cam, EXPOSURES_MS, darks_dir, N_DARK_FRAMES)
        elif mode == "flats":
            exposure_ms = float(sys.argv[2]) if len(sys.argv) > 2 else FLAT_EXPOSURE_MS
            flats_dir = os.path.join(script_dir, "flats")
            os.makedirs(flats_dir, exist_ok=True)
            capture_flats(cam, exposure_ms, flats_dir, N_FLAT_FRAMES)
        else:
            out_dir = os.path.join(script_dir, "output")
            os.makedirs(out_dir, exist_ok=True)
            capture_light_sweep(cam, EXPOSURES_MS, out_dir)


if __name__ == "__main__":
    main()
