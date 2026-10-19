"""Simple uEye capture GUI: set exposure/gain, take a picture, and view the
FITS file it wrote (image, stats, header). Can also open any existing FITS.

Usage:
    python capture_app.py [file.fits] [-e EXPOSURE_MS] [-g GAIN]

The camera is opened on the first capture and kept open until you quit or
change the color mode. Each capture is saved as
<output folder>/<prefix>_YYYYmmdd_HHMMSS.fits.
"""
import argparse
import os
import queue
import threading
import tkinter as tk
import warnings
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

STRETCHES = {
    "min / max": (0.0, 100.0),
    "99.5%": (0.25, 99.75),
    "99%": (0.5, 99.5),
    "95%": (2.5, 97.5),
    "full range": None,  # 0 .. 2^BITDEPTH - 1
}
COLORMAPS = ["gray", "viridis", "inferno", "magma", "cividis"]
IMAGETYPS = ["LIGHT", "DARK", "FLAT", "BIAS"]
MODES = ["mono8", "mono10", "mono12", "mono16"]


class CameraWorker:
    """Runs all camera calls on one background thread so the GUI never blocks.
    Results come back through `results` as (kind, payload) tuples."""

    def __init__(self):
        self.results = queue.Queue()
        self._jobs = queue.Queue()
        self._cam = None
        self._mode = None
        threading.Thread(target=self._run, daemon=True).start()

    def capture(self, mode, exposure_ms, gain, imagetyp, path):
        self._jobs.put(("capture", (mode, exposure_ms, gain, imagetyp, path)))

    def shutdown(self):
        self._jobs.put(("close", None))

    def _run(self):
        while True:
            kind, args = self._jobs.get()
            if kind == "close":
                self._close()
                self.results.put(("closed", None))
                return
            try:
                self.results.put(("captured", self._capture(*args)))
            except Exception as exc:  # report anything, keep the worker alive
                self._close()
                self.results.put(("error", f"{type(exc).__name__}: {exc}"))

    def _capture(self, mode, exposure_ms, gain, imagetyp, path):
        from ueye_cam import UEyeCamera  # imported here so viewing works without pyueye

        if self._cam is None or mode != self._mode:
            self._close()
            self.results.put(("status", "opening camera..."))
            self._cam = UEyeCamera(color_mode=mode).open()
            self._mode = mode

        notes = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if abs(self._cam.exposure_ms - exposure_ms) > 1e-6:
                self._cam.set_exposure_ms(exposure_ms)
            if self._cam.gain != gain:
                self._cam.set_gain(gain)
        notes += [str(w.message) for w in caught]

        self.results.put(("status", f"exposing {self._cam.exposure_ms:.3f} ms..."))
        frame = self._cam.snap()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frame.save_fits(path, imagetyp=imagetyp)
        return path, notes

    def _close(self):
        if self._cam is not None:
            try:
                self._cam.close()
            finally:
                self._cam = None


class CaptureApp:
    def __init__(self, root: tk.Tk, initial_file: str = None, exposure_ms: float = 1.0, gain: int = 0):
        self.root = root
        self.root.title("uEye Capture")
        self.root.geometry("1250x820")
        self.worker = CameraWorker()

        self.data = None        # 2D or 3D array of the open FITS
        self.header = None
        self.path = None
        self._image = None
        self._colorbar = None

        self._build_controls(exposure_ms, gain)
        self._build_viewer()
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self.root.bind("<space>", lambda e: self.capture())
        self.root.after(100, self._poll_worker)

        if initial_file:
            self.load_fits(initial_file)

    # -- layout --------------------------------------------------------------

    def _build_controls(self, exposure_ms, gain):
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        cam = ttk.LabelFrame(panel, text="Camera", padding=8)
        cam.pack(fill=tk.X)
        self.exposure = tk.DoubleVar(value=exposure_ms)
        self.gain = tk.IntVar(value=gain)
        self.mode = tk.StringVar(value="mono12")
        self.imagetyp = tk.StringVar(value="LIGHT")
        rows = [
            ("Exposure (ms)", ttk.Entry(cam, textvariable=self.exposure, width=12)),
            ("Gain (0-100)", ttk.Spinbox(cam, from_=0, to=100, textvariable=self.gain, width=10)),
            ("Mode", ttk.Combobox(cam, textvariable=self.mode, values=MODES, state="readonly", width=10)),
            ("Image type", ttk.Combobox(cam, textvariable=self.imagetyp, values=IMAGETYPS, width=10)),
        ]
        for r, (label, widget) in enumerate(rows):
            ttk.Label(cam, text=label).grid(row=r, column=0, sticky=tk.W, pady=2)
            widget.grid(row=r, column=1, sticky=tk.EW, pady=2)

        out = ttk.LabelFrame(panel, text="Save to", padding=8)
        out.pack(fill=tk.X, pady=(10, 0))
        self.out_dir = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures"))
        self.prefix = tk.StringVar(value="frame")
        ttk.Entry(out, textvariable=self.out_dir, width=28).grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        ttk.Button(out, text="Browse...", command=self._pick_dir).grid(row=1, column=1, sticky=tk.E, pady=2)
        ttk.Label(out, text="Prefix").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(out, textvariable=self.prefix, width=14).grid(row=2, column=1, sticky=tk.EW)

        self.capture_btn = ttk.Button(panel, text="Capture  (space)", command=self.capture)
        self.capture_btn.pack(fill=tk.X, pady=(14, 4), ipady=8)
        ttk.Button(panel, text="Open FITS...", command=self._open_dialog).pack(fill=tk.X)

        disp = ttk.LabelFrame(panel, text="Display", padding=8)
        disp.pack(fill=tk.X, pady=(14, 0))
        self.stretch = tk.StringVar(value="99.5%")
        self.cmap = tk.StringVar(value="gray")
        self.frame_idx = tk.IntVar(value=0)
        ttk.Label(disp, text="Stretch").grid(row=0, column=0, sticky=tk.W)
        cb = ttk.Combobox(disp, textvariable=self.stretch, values=list(STRETCHES), state="readonly", width=10)
        cb.grid(row=0, column=1, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(disp, text="Colormap").grid(row=1, column=0, sticky=tk.W)
        cb = ttk.Combobox(disp, textvariable=self.cmap, values=COLORMAPS, state="readonly", width=10)
        cb.grid(row=1, column=1, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(disp, text="Cube frame").grid(row=2, column=0, sticky=tk.W)
        self.frame_spin = ttk.Spinbox(disp, from_=0, to=0, textvariable=self.frame_idx, width=8,
                                      command=self.redraw, state=tk.DISABLED)
        self.frame_spin.grid(row=2, column=1, pady=2)

        self.stats = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.stats, justify=tk.LEFT, font=("Consolas", 9)).pack(
            fill=tk.X, pady=(14, 0))

        self.status = tk.StringVar(value="Ready. Press Capture or open a FITS file.")
        ttk.Label(panel, textvariable=self.status, wraplength=230, foreground="#555").pack(
            side=tk.BOTTOM, fill=tk.X)

    def _build_viewer(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        image_tab = ttk.Frame(tabs)
        tabs.add(image_tab, text="Image")
        self.fig = Figure(figsize=(8, 7), layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_tab)
        NavigationToolbar2Tk(self.canvas, image_tab).update()  # zoom / pan / save
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.hover = tk.StringVar(value="")
        ttk.Label(image_tab, textvariable=self.hover, font=("Consolas", 10)).pack(fill=tk.X)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        header_tab = ttk.Frame(tabs)
        tabs.add(header_tab, text="Header")
        self.header_text = tk.Text(header_tab, font=("Consolas", 10), wrap=tk.NONE)
        self.header_text.pack(fill=tk.BOTH, expand=True)

    # -- actions -------------------------------------------------------------

    def capture(self):
        if str(self.capture_btn["state"]) == tk.DISABLED:
            return
        try:
            exposure = float(self.exposure.get())
            gain = int(self.gain.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid setting", "Exposure and gain must be numbers.")
            return
        if exposure <= 0 or not 0 <= gain <= 100:
            messagebox.showerror("Invalid setting", "Exposure must be > 0 and gain 0-100.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir.get(), f"{self.prefix.get() or 'frame'}_{stamp}.fits")
        self.capture_btn.state(["disabled"])
        self.status.set("capturing...")
        self.worker.capture(self.mode.get(), exposure, gain, self.imagetyp.get().strip().upper(), path)

    def load_fits(self, path):
        try:
            with fits.open(path) as hdul:
                hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
                if hdu is None:
                    raise ValueError("no image data in file")
                self.data = np.asarray(hdu.data)
                if self.data.ndim > 3:  # flatten extra axes into one frame axis
                    self.data = self.data.reshape(-1, *self.data.shape[-2:])
                self.header = hdu.header.copy()
        except Exception as exc:
            messagebox.showerror("Can't open FITS", f"{path}\n\n{exc}")
            return
        self.path = path
        n = self.data.shape[0] if self.data.ndim == 3 else 1
        self.frame_idx.set(0)
        self.frame_spin.configure(to=n - 1, state=tk.NORMAL if n > 1 else tk.DISABLED)
        self.header_text.delete("1.0", tk.END)
        self.header_text.insert(tk.END, self.header.tostring(sep="\n").rstrip())
        self.root.title(f"uEye Capture - {os.path.basename(path)}")
        self.redraw(reset_view=True)

    def redraw(self, reset_view=False):
        if self.data is None:
            return
        img = self._current_image()
        stretch = STRETCHES[self.stretch.get()]
        if stretch is None:
            bits = int(self.header.get("BITDEPTH", 16 if img.dtype.itemsize > 1 else 8))
            vmin, vmax = 0, (1 << bits) - 1
        else:
            vmin, vmax = np.percentile(img, stretch)
            if vmax <= vmin:
                vmax = vmin + 1

        if self._image is None or reset_view:
            self.ax.clear()
            self._image = self.ax.imshow(img, cmap=self.cmap.get(), vmin=vmin, vmax=vmax,
                                         origin="upper", interpolation="nearest")
            if self._colorbar is None:
                self._colorbar = self.fig.colorbar(self._image, ax=self.ax, shrink=0.85)
            else:
                self._colorbar.update_normal(self._image)
        else:
            self._image.set_data(img)
            self._image.set_cmap(self.cmap.get())
            self._image.set_clim(vmin, vmax)
        self.ax.set_title(os.path.basename(self.path), fontsize=10)
        self.canvas.draw_idle()
        self._update_stats(img)

    # -- helpers -------------------------------------------------------------

    def _current_image(self):
        if self.data.ndim == 2:
            return self.data
        idx = min(max(int(self.frame_idx.get() or 0), 0), self.data.shape[0] - 1)
        return self.data[idx]

    def _update_stats(self, img):
        h = self.header
        bits = h.get("BITDEPTH")
        sat = int((img >= (1 << bits) - 1).sum()) if bits else None
        exptime = h.get("EXPTIME")
        lines = [
            f"size    {img.shape[1]} x {img.shape[0]}",
            f"exp     {exptime * 1000:.3f} ms" if exptime is not None else "exp     ?",
            f"gain    {h.get('GAIN', '?')}",
            f"type    {h.get('IMAGETYP', '?')}",
            f"mean    {img.mean():.1f}",
            f"median  {np.median(img):.1f}",
            f"std     {img.std():.1f}",
            f"min     {img.min()}",
            f"max     {img.max()}",
        ]
        if sat is not None:
            lines.append(f"sat px  {sat}")
        self.stats.set("\n".join(lines))

    def _on_hover(self, event):
        if self.data is None or event.inaxes is not self.ax or event.xdata is None:
            self.hover.set("")
            return
        img = self._current_image()
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            self.hover.set(f"  x={x}  y={y}  value={img[y, x]}")

    def _pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def _open_dialog(self):
        initial = self.out_dir.get() if os.path.isdir(self.out_dir.get()) else os.getcwd()
        path = filedialog.askopenfilename(initialdir=initial,
                                          filetypes=[("FITS", "*.fits *.fit *.fts"), ("All files", "*.*")])
        if path:
            self.load_fits(path)

    def _poll_worker(self):
        try:
            while True:
                kind, payload = self.worker.results.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "captured":
                    path, notes = payload
                    self.capture_btn.state(["!disabled"])
                    self.status.set(f"saved {path}" + "".join(f"\n{n}" for n in notes))
                    self.load_fits(path)
                elif kind == "error":
                    self.capture_btn.state(["!disabled"])
                    self.status.set("capture failed")
                    messagebox.showerror("Capture failed", payload)
                elif kind == "closed":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def _on_quit(self):
        self.status.set("closing camera...")
        self.worker.shutdown()
        self.root.after(3000, self.root.destroy)  # don't hang if the driver does


def main():
    parser = argparse.ArgumentParser(description="uEye capture + FITS viewer")
    parser.add_argument("file", nargs="?", help="FITS file to open at startup")
    parser.add_argument("-e", "--exposure-ms", type=float, default=1.0, help="exposure in ms (default 1.0)")
    parser.add_argument("-g", "--gain", type=int, default=0, help="hardware master gain, 0-100 (default 0)")
    args = parser.parse_args()

    root = tk.Tk()
    CaptureApp(root, args.file, exposure_ms=args.exposure_ms, gain=args.gain)
    root.mainloop()


if __name__ == "__main__":
    main()
