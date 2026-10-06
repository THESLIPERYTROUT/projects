"""Command-line capture tool.

    python -m ueye_cam info
    python -m ueye_cam snap -e 1.0 -o light.fits
    python -m ueye_cam snap -e 0.5 -n 20 -o flats/flat.fits --imagetyp FLAT
    python -m ueye_cam snap -e 0.5 -n 20 -o flat_cube.fits --cube
    python -m ueye_cam stream -e 10 -n 100
"""
import argparse
import os
import time

from . import COLOR_MODES, UEyeCamera, save_cube


def frame_stats(frame) -> str:
    d = frame.data
    return (f"mean {d.mean():.1f}  std {d.std():.1f}  min {int(d.min())}  max {int(d.max())}  "
            f"saturated px {frame.n_saturated}")


def cmd_info(cam, args):
    i = cam.info
    print(f"model       {i.model}\nserial      {i.serial}\nsize        {i.width} x {i.height}\n"
          f"pixel size  {i.pixel_size_um} um\nbit depth   {cam.bit_depth}\n"
          f"exposure    {cam.exposure_ms:.4f} ms\ngain        {cam.gain}")


def cmd_snap(cam, args):
    frames = cam.burst(args.n)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.cube:
        save_cube(args.output, frames, imagetyp=args.imagetyp)
        print(f"saved {len(frames)}-frame cube to {args.output}")
    elif args.n == 1:
        frames[0].save_fits(args.output, imagetyp=args.imagetyp)
        print(f"saved {args.output}")
    else:
        stem, ext = os.path.splitext(args.output)
        for frame in frames:
            frame.save_fits(f"{stem}_frame{frame.index:02d}{ext or '.fits'}", imagetyp=args.imagetyp)
        print(f"saved {len(frames)} frames as {stem}_frameNN{ext or '.fits'}")
    for frame in frames:
        print(f"  frame {frame.index:02d}: {frame_stats(frame)}")


def cmd_stream(cam, args):
    t0 = time.perf_counter()
    count = 0
    with cam.stream(n_frames=args.n) as frames:
        try:
            for frame in frames:
                count += 1
                fps = count / (time.perf_counter() - t0)
                print(f"frame {frame.index:05d}  {fps:5.1f} fps  {frame_stats(frame)}")
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(prog="python -m ueye_cam", description="IDS uEye capture")
    parser.add_argument("--camera-id", type=int, default=0, help="0 = first available camera")
    parser.add_argument("--mode", choices=sorted(COLOR_MODES), default="mono8")
    parser.add_argument("-e", "--exposure-ms", type=float, help="exposure in ms (1000 = 1 s)")
    parser.add_argument("-g", "--gain", type=int, help="hardware master gain, 0-100")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print camera details and current settings")

    snap = sub.add_parser("snap", help="capture frames to FITS")
    snap.add_argument("-o", "--output", required=True, help="FITS path")
    snap.add_argument("-n", type=int, default=1, help="number of frames")
    snap.add_argument("--imagetyp", default="LIGHT", help="IMAGETYP header (LIGHT, DARK, FLAT, BIAS...)")
    snap.add_argument("--cube", action="store_true", help="write all frames into one 3D FITS")

    stream = sub.add_parser("stream", help="live capture, printing per-frame stats (Ctrl+C to stop)")
    stream.add_argument("-n", type=int, help="stop after this many frames")

    args = parser.parse_args()
    with UEyeCamera(camera_id=args.camera_id, color_mode=args.mode,
                    exposure_ms=args.exposure_ms, gain=args.gain) as cam:
        {"info": cmd_info, "snap": cmd_snap, "stream": cmd_stream}[args.command](cam, args)


if __name__ == "__main__":
    main()
