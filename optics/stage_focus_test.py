"""
Stage positional test using a focused laser spot on the uEye camera.

Runs in one process on Windows: drives the MicroLYNX stage over serial and
the camera through ueye_cam, so every frame is tied to a commanded position.

    python stage_focus_test.py check  --port COM5
    python stage_focus_test.py run    --port COM5 --units-per-mm 1

Phases of `run`:
  1. Coarse focus sweep around the current position (always approached from
     the same side), fits w^2(z) = w0^2 (1 + ((z - z0)/zR)^2) for x and y.
  2. Fine sweep +/- ~4 zR around the fitted waist, refit.
  3. Repeatability: park at z_op = z0 + zR (best width sensitivity), then
     alternate excursions above/below so z_op is approached from both sides.
     Width is converted back to z through the caustic fit.

Output goes to runs/<timestamp>/: CSV per phase, summary.txt, plots.
Units on the command line are mm; the stage is commanded in its own units
(MUNIT) via --units-per-mm.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import warnings

import numpy as np

# --------------------------------------------------------------------------- #
# Spot measurement (ISO 11146-style second moments)
# --------------------------------------------------------------------------- #

def background_level(img: np.ndarray, border: int = 20) -> float:
    """Median of a border strip, assumed free of laser light."""
    b = np.concatenate([img[:border].ravel(), img[-border:].ravel(),
                        img[:, :border].ravel(), img[:, -border:].ravel()])
    return float(np.median(b))


def _box_smooth(img, k=5):
    c = np.cumsum(np.cumsum(np.pad(img, ((1, 0), (1, 0))), 0), 1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return np.pad(s, ((k // 2, k - 1 - k // 2), (k // 2, k - 1 - k // 2)))


def _gauss2d(xy, amp, x0, y0, wx, wy, off):
    x, y = xy
    return off + amp * np.exp(-2 * (x - x0) ** 2 / wx ** 2 - 2 * (y - y0) ** 2 / wy ** 2)


def spot_moments(img: np.ndarray, n_iter: int = 5):
    """Centroid and 1/e^2 radii (pixels).

    Seeds with noise-thresholded second moments (robust to background), then
    refines with a 2D Gaussian fit in a ROI of +/-2.5 w when scipy is available.
    """
    img = img.astype(np.float64)
    h, w = img.shape
    border = np.concatenate([img[:10].ravel(), img[-10:].ravel(), img[:, :10].ravel(), img[:, -10:].ravel()])
    bg = float(np.median(border))
    noise = max(1.4826 * float(np.median(np.abs(border - bg))), 0.5)
    img = img - bg

    # 1) Seed: pixels clearly part of the spot (above 3 sigma of the 5x5 mean
    #    AND above 20% of the smoothed peak), so stray noise can't dominate.
    smooth = _box_smooth(img) / 25.0
    smax = float(smooth.max())
    if smax < 3 * noise / 5.0:
        return None
    sel = smooth > max(3 * noise / 5.0, 0.2 * smax)
    yy, xx = np.nonzero(sel)
    wts = np.clip(img[yy, xx], 0, None)
    if wts.sum() <= 0:
        return None
    cx = float((wts * xx).sum() / wts.sum())
    cy = float((wts * yy).sum() / wts.sum())
    # radius at which a Gaussian falls to 20% of peak is 0.897 w
    wx = max(float(xx.max() - xx.min() + 1) / 2 / 0.897, 1.0)
    wy = max(float(yy.max() - yy.min() + 1) / 2 / 0.897, 1.0)
    seed = (cx, cy, wx, wy)

    # 2) ISO 11146-style second moments in a +/-3w box, no clipping (unbiased)
    for _ in range(n_iter):
        hx, hy = max(6.0, 3.0 * wx), max(6.0, 3.0 * wy)
        x0, x1 = int(max(0, cx - hx)), int(min(w, cx + hx + 1))
        y0, y1 = int(max(0, cy - hy)), int(min(h, cy + hy + 1))
        r = img[y0:y1, x0:x1]
        tot = r.sum()
        if tot <= 0:
            return None
        gy, gx = np.mgrid[y0:y1, x0:x1]
        cx = float((r * gx).sum() / tot)
        cy = float((r * gy).sum() / tot)
        # cap growth so background noise can't run the box away on small spots
        wx = min(2 * np.sqrt(max(float((r * (gx - cx) ** 2).sum() / tot), 0.25)), 1.5 * seed[2])
        wy = min(2 * np.sqrt(max(float((r * (gy - cy) ** 2).sum() / tot), 0.25)), 1.5 * seed[3])
    half = max(seed[2], seed[3])

    peak = float(img[int(round(cy)), int(round(cx))])
    method = "moments"
    try:
        from scipy.optimize import curve_fit
        half = max(8.0, 3.0 * half)
        sx, sy = seed[2], seed[3]
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half + 1))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half + 1))
        roi = img[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        p0 = [max(roi.max(), 1.0), seed[0], seed[1], sx, sy, 0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), roi.ravel(), p0=p0, maxfev=4000)
        ok = (popt[0] > 0 and abs(popt[1] - seed[0]) < max(sx, 2) and abs(popt[2] - seed[1]) < max(sy, 2)
              and 0.5 * sx < abs(popt[3]) < 2.0 * sx and 0.5 * sy < abs(popt[4]) < 2.0 * sy)
        if ok:
            peak, cx, cy, wx, wy = popt[0], popt[1], popt[2], abs(popt[3]), abs(popt[4])
            method = "gauss"
    except ImportError:
        pass
    except (RuntimeError, ValueError):
        pass
    return dict(cx=cx, cy=cy, wx=wx, wy=wy, peak=float(peak), method=method)


def measure_frames(frames):
    """Per-frame measurement, then median and robust std (MAD) across the burst."""
    rows = [m for m in (spot_moments(f) for f in frames) if m is not None]
    if not rows:
        return None
    out = {}
    for k in ("cx", "cy", "wx", "wy", "peak"):
        v = np.array([r[k] for r in rows])
        med = float(np.median(v))
        out[k] = med
        out[k + "_std"] = float(1.4826 * np.median(np.abs(v - med))) if len(v) > 1 else 0.0
    out["n"] = len(rows)
    return out

# --------------------------------------------------------------------------- #
# Caustic fit
# --------------------------------------------------------------------------- #

def fit_caustic(z_mm, w_um, n_reject=3):
    """Fit w^2 = a z^2 + b z + c (linear least squares, iterative 3-sigma rejection).

    Returns dict z0 (mm), w0 (um), zR (mm), residual rms (um), points used.
    """
    z_all = np.asarray(z_mm, float)
    w_all = np.asarray(w_um, float)
    keep = np.isfinite(w_all)
    for _ in range(n_reject + 1):
        z, w = z_all[keep], w_all[keep]
        if len(z) < 4:
            return None
        A = np.vstack([z ** 2, z, np.ones_like(z)]).T
        (a, b, c), *_ = np.linalg.lstsq(A, w ** 2, rcond=None)
        if a <= 0:
            return None
        model_all = np.sqrt(np.clip(a * z_all ** 2 + b * z_all + c, 0, None))
        res = w_all - model_all
        sig = 1.4826 * np.median(np.abs(res[keep]))
        new_keep = np.isfinite(w_all) & (np.abs(res) < max(4 * sig, 0.3))
        if (new_keep == keep).all():
            break
        keep = new_keep
    if a <= 0:
        return None
    z0 = -b / (2 * a)
    w0sq = c - b ** 2 / (4 * a)
    if w0sq <= 0:
        return None
    w0 = np.sqrt(w0sq)
    zR = w0 / np.sqrt(a)                   # a in um^2/mm^2, w0 in um -> zR in mm
    model = np.sqrt(np.clip(a * z ** 2 + b * z + c, 0, None))
    return dict(z0=float(z0), w0=float(w0), zR=float(zR), a=float(a), b=float(b), c=float(c),
                resid_rms=float(np.sqrt(np.mean((w - model) ** 2))), n_used=int(keep.sum()),
                n_total=len(z_all))


def caustic_w(fit, z):
    return np.sqrt(fit["a"] * np.asarray(z) ** 2 + fit["b"] * np.asarray(z) + fit["c"])


def z_from_width(fit, w_um, side=+1):
    """Invert the caustic on one side of the waist."""
    ratio = np.clip((np.asarray(w_um) / fit["w0"]) ** 2 - 1.0, 0, None)
    return fit["z0"] + side * fit["zR"] * np.sqrt(ratio)


def dz_dw(fit, z):
    """Sensitivity: mm of z per um of width at position z."""
    w = caustic_w(fit, z)
    dwdz = (2 * fit["a"] * z + fit["b"]) / (2 * w)
    return 1.0 / dwdz

# --------------------------------------------------------------------------- #
# Stage (MicroLYNX over serial)
# --------------------------------------------------------------------------- #

class Stage:
    def __init__(self, port, units_per_mm, timeout_s=60.0):
        import serial
        self.ser = serial.Serial(port=port, baudrate=9600, bytesize=8, parity="N",
                                 stopbits=1, timeout=0.5)
        self.k = float(units_per_mm)
        self.timeout_s = timeout_s
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _xfer(self, cmd, wait=0.05):
        self.ser.write((cmd + "\r").encode())
        deadline = time.time() + 1.0
        buf = b""
        time.sleep(wait)
        while time.time() < deadline:
            buf += self.ser.read_all() or b""
            if buf.rstrip().endswith(b">") or b"?" in buf:
                break
            time.sleep(0.01)
        return buf.decode(errors="ignore")

    def query(self, cmd):
        resp = self._xfer(cmd).replace("\r", "").split("\n")
        for line in reversed(resp):
            line = line.strip()
            if line and line != ">" and line.upper() != cmd.strip().upper():
                return line.lstrip(">").strip()
        return ""

    def pos_units(self):
        return float(self.query("PRINT POS"))

    def pos_mm(self):
        return self.pos_units() / self.k

    def _wait_stop(self):
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.query("PRINT MVG") == "0":
                return
            time.sleep(0.05)
        raise RuntimeError("stage motion timeout")

    def move_abs_mm(self, z_mm, settle_s):
        self._xfer(f"MOVA {z_mm * self.k:.6f}")
        self._wait_stop()
        time.sleep(settle_s)
        return self.pos_mm()

    def limits(self):
        return self.query("PRINT LMTP"), self.query("PRINT LMTM")

    def status(self):
        return {r: self.query(f"PRINT {r}") for r in ("POS", "MUNIT", "VM", "VI", "ACCL", "DECL", "MVG")}

class RemoteStage(Stage):
    """Same interface as Stage, but talks to stage_server.py (running in WSL) over TCP."""

    def __init__(self, host, port, units_per_mm, timeout_s=90.0):
        import socket
        self.k = float(units_per_mm)
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.f = self.sock.makefile("rwb")

    def _call(self, **req):
        self.f.write((json.dumps(req) + "\n").encode())
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise RuntimeError("stage server closed the connection")
        out = json.loads(line)
        if not out.get("ok"):
            raise RuntimeError(f"stage server error: {out.get('error')}")
        return out

    def close(self):
        try:
            self.f.close()
        finally:
            self.sock.close()

    def query(self, cmd):
        return self._call(op="query", cmd=cmd)["resp"]

    def move_abs_mm(self, z_mm, settle_s):
        self._call(op="mova", units=z_mm * self.k)
        time.sleep(settle_s)
        return self.pos_mm()


def open_stage(args):
    if args.port.lower().startswith("tcp:"):
        _, host, port = args.port.split(":")
        return RemoteStage(host, int(port), args.units_per_mm)
    return Stage(args.port, args.units_per_mm)

# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #

def open_camera(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ueye_cam import UEyeCamera
    for mode in ([args.mode] if args.mode != "auto" else ["mono10", "mono8"]):
        try:
            cam = UEyeCamera(color_mode=mode, exposure_ms=args.exposure_ms, gain=args.gain).open()
            print(f"camera {cam.info.model}  {cam.info.width}x{cam.info.height}  "
                  f"{cam.info.pixel_size_um} um px  mode {mode}  exp {cam.exposure_ms:.4f} ms")
            return cam
        except Exception as exc:
            print(f"  could not open camera in {mode}: {exc}")
    raise SystemExit("camera did not open")


def grab(cam, n, lo=0.35, hi=0.85, max_exp_ms=50.0):
    """Burst n frames, adjusting exposure so the spot peak sits in [lo, hi] of full scale.

    Width does not depend on exposure (background is subtracted), so this only
    keeps SNR high and avoids clipping as the spot grows/shrinks through focus.
    """
    for _ in range(12):
        probe = cam.burst(1)[0]
        peak = int(probe.data.max()) / probe.saturation_value
        if peak >= 0.98:
            cam.set_exposure_ms(max(cam.exposure_ms * 0.4, 0.005))
        elif peak > hi:
            cam.set_exposure_ms(max(cam.exposure_ms * hi * 0.9 / peak, 0.005))
        elif peak < lo and cam.exposure_ms < max_exp_ms:
            cam.set_exposure_ms(min(cam.exposure_ms * 0.6 / max(peak, 0.02), max_exp_ms))
        else:
            break
    frames = cam.burst(n)
    return frames, max(int(f.data.max()) for f in frames) / frames[0].saturation_value

# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

FIELDS = ["phase", "i", "t", "z_cmd_mm", "z_read_mm", "approach", "exposure_ms", "peak_frac",
          "cx", "cy", "wx", "wy", "cx_std", "cy_std", "wx_std", "wy_std", "n"]


def record(writer, rows, phase, i, z_cmd, z_read, approach, cam, frames, frac, px_um):
    m = measure_frames([f.data for f in frames])
    if m is None:
        print(f"  [{phase} {i}] no spot found at z={z_cmd:.4f} mm")
        return None
    row = dict(phase=phase, i=i, t=time.time(), z_cmd_mm=z_cmd, z_read_mm=z_read,
               approach=approach, exposure_ms=cam.exposure_ms, peak_frac=frac, n=m["n"])
    for k in ("cx", "cy", "cx_std", "cy_std"):
        row[k] = m[k]
    for k in ("wx", "wy", "wx_std", "wy_std"):
        row[k] = m[k] * px_um
    writer.writerow(row)
    rows.append(row)
    print(f"  [{phase} {i:3d}] z={z_cmd:+.4f} mm  wx={row['wx']:6.1f} wy={row['wy']:6.1f} um  "
          f"cx={m['cx']:7.2f} cy={m['cy']:7.2f} px  peak {frac*100:4.0f}%  exp {cam.exposure_ms:.3f} ms")
    return row


def sweep(stage, cam, writer, rows, phase, z_list, args, px_um, pre_mm):
    """Visit z_list in ascending order, always approaching from below."""
    stage.move_abs_mm(z_list[0] - pre_mm, args.settle_s)
    out = []
    for i, z in enumerate(z_list):
        zr = stage.move_abs_mm(z, args.settle_s)
        frames, frac = grab(cam, args.frames)
        r = record(writer, rows, phase, i, z, zr, "+", cam, frames, frac, px_um)
        if r:
            out.append(r)
    return out


def fit_axes(rows):
    z = [r["z_cmd_mm"] for r in rows]
    fx = fit_caustic(z, [r["wx"] for r in rows])
    fy = fit_caustic(z, [r["wy"] for r in rows])
    return fx, fy


def describe_fit(name, f):
    if f is None:
        return f"{name}: fit failed (no curvature; sweep may not straddle the waist)"
    return (f"{name}: z0={f['z0']:+.4f} mm  w0={f['w0']:.1f} um  zR={f['zR']:.3f} mm  "
            f"resid rms={f['resid_rms']:.2f} um  ({f['n_used']}/{f['n_total']} pts)")


def run(args):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"output -> {out_dir}")

    stage = open_stage(args)
    cam = open_camera(args)
    px_um = cam.info.pixel_size_um
    summary = {"args": vars(args), "stage_status": stage.status()}
    print("stage:", summary["stage_status"])

    z_start = stage.pos_mm()
    lo, hi = z_start - args.max_travel, z_start + args.max_travel
    clamp = lambda z: float(min(max(z, lo), hi))
    print(f"start z = {z_start:.4f} mm; travel limited to [{lo:.3f}, {hi:.3f}] mm")
    if not args.yes:
        if input("Stage will move. Continue? [y/N] ").strip().lower() != "y":
            return

    rows = []
    fcsv = open(os.path.join(out_dir, "data.csv"), "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
    writer.writeheader()
    try:
        # ---- 1. coarse sweep ----------------------------------------------
        print("\n== coarse focus sweep ==")
        zc = [clamp(z) for z in np.arange(z_start - args.coarse_range,
                                          z_start + args.coarse_range + 1e-9, args.coarse_step)]
        coarse = sweep(stage, cam, writer, rows, "coarse", sorted(set(zc)), args, px_um, args.preload)
        fcsv.flush()
        best = min(coarse, key=lambda r: r["wx"] + r["wy"])
        fx, fy = fit_axes(coarse)
        print(describe_fit("x", fx)); print(describe_fit("y", fy))
        z_center = best["z_cmd_mm"]
        if fx and fy and abs(fx["z0"] - z_center) < args.coarse_range:
            z_center = 0.5 * (fx["z0"] + fy["z0"])

        # ---- 2. fine sweep ------------------------------------------------
        zR_guess = np.mean([f["zR"] for f in (fx, fy) if f]) if (fx or fy) else 1.0
        half = min(max(4 * zR_guess, 1.0), args.coarse_range)
        print(f"\n== fine sweep: {z_center:.4f} +/- {half:.3f} mm ==")
        zf = [clamp(z) for z in np.linspace(z_center - half, z_center + half, args.fine_points)]
        fine = sweep(stage, cam, writer, rows, "fine", zf, args, px_um, args.preload)
        fcsv.flush()
        fx, fy = fit_axes(fine)
        print(describe_fit("x", fx)); print(describe_fit("y", fy))
        summary["fit_x"], summary["fit_y"] = fx, fy
        if not (fx and fy):
            raise RuntimeError("fine caustic fit failed; check the spot and sweep range")
        astig = fx["z0"] - fy["z0"]
        print(f"astigmatic waist separation (x - y) = {astig*1000:+.1f} um "
              f"({abs(astig)/np.mean([fx['zR'], fy['zR']]):.2f} zR)")

        # Use the axis-averaged width as the z sensor
        fw = fit_caustic([r["z_cmd_mm"] for r in fine],
                         [np.sqrt(r["wx"] * r["wy"]) for r in fine])
        summary["fit_w"] = fw
        side = +1 if args.side == "far" else -1
        z_op = clamp(fw["z0"] + side * fw["zR"])
        sens = abs(dz_dw(fw, z_op))
        print(f"\noperating point z_op = {z_op:.4f} mm  (z0 {side:+d} zR); "
              f"sensitivity {sens*1000:.1f} um of z per um of width")

        # ---- 3. repeatability --------------------------------------------
        print(f"\n== repeatability: {args.cycles} cycles, +/-{args.amp:.3f} mm excursions ==")
        rep = []
        stage.move_abs_mm(clamp(z_op - args.amp), args.settle_s)
        for c in range(args.cycles):
            for approach, excursion in (("+", -args.amp), ("-", +args.amp)):
                # approach z_op from below ("+") then from above ("-")
                if approach == "-":
                    stage.move_abs_mm(clamp(z_op + args.amp), args.settle_s)
                zr = stage.move_abs_mm(z_op, args.settle_s)
                frames, frac = grab(cam, args.repeat_frames)
                r = record(writer, rows, "repeat", 2 * c + (approach == "-"), z_op, zr,
                           approach, cam, frames, frac, px_um)
                if r:
                    r["z_meas_mm"] = float(z_from_width(fw, np.sqrt(r["wx"] * r["wy"]), side))
                    rep.append(r)
                if approach == "-":
                    stage.move_abs_mm(clamp(z_op - args.amp), args.settle_s)
            fcsv.flush()

        # ---- 4. drift check: re-measure without moving --------------------
        print("\n== stationary check at z_op (no motion) ==")
        stage.move_abs_mm(z_op, args.settle_s)
        stat = []
        for i in range(args.static_points):
            frames, frac = grab(cam, args.repeat_frames)
            r = record(writer, rows, "static", i, z_op, z_op, "0", cam, frames, frac, px_um)
            if r:
                r["z_meas_mm"] = float(z_from_width(fw, np.sqrt(r["wx"] * r["wy"]), side))
                stat.append(r)

        stage.move_abs_mm(z_start, args.settle_s)
        print(f"\nreturned to start z = {stage.pos_mm():.4f} mm")
    finally:
        fcsv.close()
        cam.close()
        stage.close()

    report(out_dir, summary, rows, fine, rep, stat, fw, z_op, px_um)


def report(out_dir, summary, rows, fine, rep, stat, fw, z_op, px_um):
    lines = []
    up = np.array([r["z_meas_mm"] for r in rep if r["approach"] == "+"]) * 1000
    dn = np.array([r["z_meas_mm"] for r in rep if r["approach"] == "-"]) * 1000
    st = np.array([r["z_meas_mm"] for r in stat]) * 1000
    cxu = np.array([r["cx"] for r in rep if r["approach"] == "+"]) * px_um
    cxd = np.array([r["cx"] for r in rep if r["approach"] == "-"]) * px_um
    for k in ("fit_x", "fit_y", "fit_w"):
        lines.append(describe_fit(k, summary.get(k)))
    lines.append(f"z_op = {z_op:.4f} mm")
    if len(st) > 1:
        lines.append(f"noise floor (no motion): std {st.std(ddof=1):.2f} um of z over {len(st)} points")
    if len(up) > 1 and len(dn) > 1:
        lines.append(f"approach from below: mean {up.mean() - z_op*1000:+.2f} um, std {up.std(ddof=1):.2f} um (n={len(up)})")
        lines.append(f"approach from above: mean {dn.mean() - z_op*1000:+.2f} um, std {dn.std(ddof=1):.2f} um (n={len(dn)})")
        lines.append(f"reversal error (above - below): {dn.mean() - up.mean():+.2f} um")
        lines.append(f"lateral centroid shift on reversal (x): {cxd.mean() - cxu.mean():+.2f} um")
    lines.append("note: z here is inferred from spot width; absolute offset depends on the caustic fit, "
                 "differences between approaches are the meaningful numbers.")
    text = "\n".join(lines)
    print("\n== summary ==\n" + text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        z = np.array([r["z_cmd_mm"] for r in fine])
        zz = np.linspace(z.min(), z.max(), 300)
        for key, col, fk in (("wx", "C0", "fit_x"), ("wy", "C1", "fit_y")):
            ax[0].plot(z, [r[key] for r in fine], "o", color=col, label=key)
            if summary.get(fk):
                ax[0].plot(zz, caustic_w(summary[fk], zz), "-", color=col)
        ax[0].axvline(z_op, color="k", ls=":", label="z_op")
        ax[0].set_xlabel("commanded z (mm)"); ax[0].set_ylabel("1/e² radius (µm)")
        ax[0].set_title("caustic"); ax[0].legend()
        ax[1].plot(np.arange(len(up)), up - z_op * 1000, "o-", label="approach from below")
        ax[1].plot(np.arange(len(dn)), dn - z_op * 1000, "s-", label="approach from above")
        if len(st):
            ax[1].axhspan(st.mean() - z_op*1000 - st.std(), st.mean() - z_op*1000 + st.std(),
                          color="0.85", label="static ±1σ")
        ax[1].set_xlabel("cycle"); ax[1].set_ylabel("measured z − z_op (µm)")
        ax[1].set_title("repeatability"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "results.png"), dpi=130)
        print(f"plot -> {os.path.join(out_dir, 'results.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


def check(args):
    """No motion: report stage status and one frame's spot measurement."""
    stage = open_stage(args)
    print("stage:", stage.status(), " limits (LMTP, LMTM):", stage.limits())
    print(f"position = {stage.pos_mm():.4f} mm")
    stage.close()
    cam = open_camera(args)
    try:
        frames, frac = grab(cam, args.frames)
        m = measure_frames([f.data for f in frames])
        print(f"peak {frac*100:.0f}% of full scale, exposure {cam.exposure_ms:.4f} ms")
        if m:
            px = cam.info.pixel_size_um
            print(f"spot at ({m['cx']:.1f}, {m['cy']:.1f}) px, wx {m['wx']*px:.1f} um, wy {m['wy']*px:.1f} um "
                  f"(±{m['wx_std']*px:.2f}, ±{m['wy_std']*px:.2f} frame-to-frame)")
        frames[0].save_fits("check_frame.fits", imagetyp="LIGHT")
        print("saved check_frame.fits")
    finally:
        cam.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "run"])
    p.add_argument("--port", required=True,
                   help="tcp:localhost:5555 to use stage_server.py in WSL, or a serial port (COM5, /dev/ttyUSB0)")
    p.add_argument("--units-per-mm", type=float, required=True,
                   help="stage user units per mm (1 if MUNIT is set so 1 unit = 1 mm)")
    p.add_argument("--mode", default="auto", help="camera mode: auto, mono8, mono10, mono12")
    p.add_argument("--exposure-ms", type=float, default=0.05)
    p.add_argument("--gain", type=int, default=0)
    p.add_argument("--frames", type=int, default=20, help="frames averaged per point")
    p.add_argument("--repeat-frames", type=int, default=100,
                   help="frames averaged per point in the repeatability and static phases")
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--max-travel", type=float, default=6.0, help="mm either side of start, hard limit")
    p.add_argument("--coarse-range", type=float, default=5.0, help="mm either side of start")
    p.add_argument("--coarse-step", type=float, default=0.25, help="mm")
    p.add_argument("--fine-points", type=int, default=41)
    p.add_argument("--preload", type=float, default=0.5, help="mm pre-travel so sweeps approach from below")
    p.add_argument("--side", choices=["far", "near"], default="far", help="operating side of waist")
    p.add_argument("--amp", type=float, default=0.5, help="mm excursion for repeatability cycles")
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--static-points", type=int, default=10)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()
    (check if args.command == "check" else run)(args)


if __name__ == "__main__":
    main()
