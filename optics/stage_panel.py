"""Stage tab for capture_app: connect to stage_server.py, jog the stage, and
run a repeatability test that measures the spot after every move.

Repeatability test, per cycle:
    move away to  test -/+ excursion   (which side depends on the approach mode)
    move back to  test
    wait settle time, read back POS
    capture N frames, measure the spot in the app's aperture

Every run writes to <output folder>/stage_runs/<timestamp>/: the FITS frames
(with STAGECMD/STAGEPOS in the header), results.csv after every measurement,
and summary.txt + report.png when the run ends. Positions are in the stage's
own units (MUNIT).
"""
import csv
import json
import os
import queue
import socket
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

APPROACHES = {
    "from below": [+1],          # always arrive moving in +direction
    "from above": [-1],
    "alternate both sides": [+1, -1],
}
CSV_FIELDS = ["n", "time", "cycle", "frame", "approach", "cmd", "pos", "pos_err",
              "fwhm_px", "fwhm_x_px", "fwhm_y_px", "ee80_px", "x", "y", "net_flux", "peak",
              "n_saturated", "file"]


class StageClient:
    """Line-delimited JSON client for stage_server.py (one request, one reply)."""

    def __init__(self, host, port, connect_timeout_s=5.0, move_timeout_s=90.0):
        self.sock = socket.create_connection((host, port), timeout=connect_timeout_s)
        self.f = self.sock.makefile("rwb")
        # The server serves one client at a time; a second client connects but
        # gets no reply, so fail fast here instead of on the first move.
        try:
            self.munit = self.query("PRINT MUNIT")
        except socket.timeout:
            self.close()
            raise RuntimeError("stage server didn't answer -- is another client "
                               "(e.g. stage_focus_test.py) still connected?") from None
        self.sock.settimeout(move_timeout_s)

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

    def query(self, cmd):
        return self._call(op="query", cmd=cmd)["resp"]

    def position(self) -> float:
        return float(self.query("PRINT POS"))

    def move_abs(self, units: float, tol: float = 0.0005):
        """Blocks until the server reports the move finished. Returns (POS,
        reached); reached is False if the stage stopped more than `tol` away
        (limit switch, stall). Older servers don't report it, so assume True."""
        out = self._call(op="mova", units=units, tol=tol)
        return float(out["pos"]), bool(out.get("reached", True))

    def close(self):
        try:
            self.f.close()
        finally:
            self.sock.close()


class StageWorker:
    """Runs stage calls on a background thread (moves block for seconds).
    Results arrive on `results` as ("ok", op, value) or ("error", op, message);
    a socket-level failure also drops the connection and reports op "lost"."""

    def __init__(self):
        self.results = queue.Queue()
        self._jobs = queue.Queue()
        self._client = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, op, *args):
        self._jobs.put((op, args))

    def _run(self):
        while True:
            op, args = self._jobs.get()
            try:
                self.results.put(("ok", op, self._do(op, *args)))
            except Exception as exc:
                self.results.put(("error", op, f"{type(exc).__name__}: {exc}"))
                if op != "connect" and self._client is not None and (
                        isinstance(exc, OSError) or "closed the connection" in str(exc)):
                    self._disconnect()
                    self.results.put(("ok", "lost", None))
            if op == "quit":
                return

    def _do(self, op, *args):
        if op == "connect":
            self._disconnect()
            self._client = StageClient(*args)
            return {"munit": self._client.munit, "pos": self._client.position()}
        if op in ("disconnect", "quit"):
            self._disconnect()
            return None
        if self._client is None:
            raise RuntimeError("not connected to the stage server")
        if op == "move":
            return self._client.move_abs(*args)
        if op == "position":
            return self._client.position()
        raise ValueError(f"unknown stage op {op!r}")

    def _disconnect(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


def summarize(rows, meta) -> str:
    """Plain-text repeatability report for a list of measurement rows."""
    px_um = meta.get("pixel_um")

    def both(values, fmt="{:.3f}"):
        s = fmt.format(values)
        return s + (f" px  ({values * px_um:.3f} um)" if px_um else " px")

    def stats_block(label, v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return [f"{label:<22} no valid values"]
        return [f"{label:<22} mean {v.mean():.3f}   std {v.std(ddof=1) if len(v) > 1 else 0:.3f}   "
                f"min {v.min():.3f}   max {v.max():.3f}   p-p {np.ptp(v):.3f}"]

    lines = [
        "Stage repeatability report",
        "=" * 26,
        f"date            {meta.get('started', '')}",
        f"stage server    {meta.get('server', '')}   units (MUNIT) {meta.get('munit', '?')}",
        f"test position   {meta.get('test', '')}   excursion {meta.get('excursion', '')}   "
        f"approach {meta.get('approach', '')}",
        f"cycles          {meta.get('cycles', '')}   frames/visit {meta.get('frames', '')}   "
        f"settle {meta.get('settle', '')} s",
        f"camera          exposure {meta.get('exposure_ms', '?')} ms   gain {meta.get('gain', '?')}   "
        f"mode {meta.get('mode', '?')}",
        f"aperture        r {meta.get('r_ap', '?')}  annulus {meta.get('r_in', '?')}-{meta.get('r_out', '?')} px"
        f"   pixel {px_um or '?'} um",
        f"measurements    {len(rows)}   valid spot {sum(np.isfinite(r['fwhm_px']) for r in rows)}",
        "",
        "All values in pixels unless noted.",
    ]
    if not rows:
        return "\n".join(lines + ["", "no measurements"])

    col = {k: np.array([r[k] for r in rows], float) for k in
           ("fwhm_px", "fwhm_x_px", "fwhm_y_px", "ee80_px", "x", "y", "pos_err", "peak")}
    lines += [""]
    lines += stats_block("FWHM (Gaussian fit)", col["fwhm_px"])
    lines += stats_block("FWHM x (moments)", col["fwhm_x_px"])
    lines += stats_block("FWHM y (moments)", col["fwhm_y_px"])
    lines += stats_block("EE80 radius", col["ee80_px"])
    lines += stats_block("centroid x", col["x"])
    lines += stats_block("centroid y", col["y"])
    lines += stats_block("peak (counts)", col["peak"])
    lines += stats_block("pos - cmd (units)", col["pos_err"])

    fw = col["fwhm_px"][np.isfinite(col["fwhm_px"])]
    if len(fw) > 1:
        lines += ["", f"FWHM repeatability (1 sigma)      {both(fw.std(ddof=1))}",
                  f"FWHM repeatability (peak-peak)    {both(np.ptp(fw))}",
                  f"FWHM relative scatter             {100 * fw.std(ddof=1) / fw.mean():.2f} %"]
    ok = np.isfinite(col["x"]) & np.isfinite(col["y"])
    if ok.sum() > 1:
        dx, dy = col["x"][ok] - col["x"][ok].mean(), col["y"][ok] - col["y"][ok].mean()
        lines += [f"centroid rms radial scatter       {both(float(np.sqrt(np.mean(dx ** 2 + dy ** 2))))}",
                  f"centroid max radial excursion     {both(float(np.max(np.hypot(dx, dy))))}"]

    approaches = sorted({r["approach"] for r in rows})
    if len(approaches) > 1:
        lines += ["", "Per approach direction (a difference here points at backlash / hysteresis):"]
        means = {}
        for a in approaches:
            sel = np.array([r["approach"] == a for r in rows])
            f, x, y = col["fwhm_px"][sel], col["x"][sel], col["y"][sel]
            means[a] = (np.nanmean(f), np.nanmean(x), np.nanmean(y))
            lines.append(f"  {a:<12} n={sel.sum():<4} FWHM {np.nanmean(f):.3f} +/- "
                         f"{np.nanstd(f, ddof=1) if sel.sum() > 1 else 0:.3f}   "
                         f"centroid ({np.nanmean(x):.3f}, {np.nanmean(y):.3f})")
        if {"from below", "from above"} <= set(means):
            b, a = means["from below"], means["from above"]
            lines.append(f"  above - below   dFWHM {a[0] - b[0]:+.3f}   dx {a[1] - b[1]:+.3f}   "
                         f"dy {a[2] - b[2]:+.3f}")
    return "\n".join(lines)


class StagePanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.worker = StageWorker()
        self.connected = False
        self.munit = "?"
        self.rows = []
        self.meta = {}
        self.run_dir = None
        self._steps = None          # generator driving the current run
        self._stop_requested = False
        self._capture_meta = None   # set while waiting for a run capture
        self._build()
        self.after(100, self._poll)

    @property
    def running(self):
        return self._steps is not None

    # -- layout --------------------------------------------------------------

    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        conn = ttk.LabelFrame(left, text="Stage server", padding=8)
        conn.pack(fill=tk.X)
        self.host = tk.StringVar(value="localhost")
        self.port = tk.IntVar(value=5555)
        ttk.Label(conn, text="Host").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(conn, textvariable=self.host, width=16).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(conn, text="Port").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(conn, textvariable=self.port, width=8).grid(row=1, column=1, sticky=tk.W)
        self.connect_btn = ttk.Button(conn, text="Connect", command=self._toggle_connect)
        self.connect_btn.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0))
        self.pos_text = tk.StringVar(value="not connected")
        ttk.Label(conn, textvariable=self.pos_text, font=("Consolas", 10)).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        jog = ttk.LabelFrame(left, text="Move", padding=8)
        jog.pack(fill=tk.X, pady=(10, 0))
        self.goto = tk.DoubleVar(value=0.0)
        self.step = tk.DoubleVar(value=0.1)
        self.tol = tk.DoubleVar(value=0.0005)
        ttk.Label(jog, text="Position").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(jog, textvariable=self.goto, width=10).grid(row=0, column=1)
        ttk.Button(jog, text="Go", width=5, command=lambda: self._move(self.goto.get())).grid(row=0, column=2)
        ttk.Label(jog, text="Step").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(jog, textvariable=self.step, width=10).grid(row=1, column=1)
        steps = ttk.Frame(jog)
        steps.grid(row=1, column=2)
        ttk.Button(steps, text="-", width=2, command=lambda: self._jog(-1)).pack(side=tk.LEFT)
        ttk.Button(steps, text="+", width=2, command=lambda: self._jog(+1)).pack(side=tk.LEFT)
        ttk.Label(jog, text="In-position tol").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(jog, textvariable=self.tol, width=10).grid(row=2, column=1)
        ttk.Button(jog, text="Read position", command=lambda: self._stage("position")).grid(
            row=3, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))

        rep = ttk.LabelFrame(left, text="Repeatability test", padding=8)
        rep.pack(fill=tk.X, pady=(10, 0))
        self.test_pos = tk.DoubleVar(value=0.0)
        self.excursion = tk.DoubleVar(value=1.0)
        self.cycles = tk.IntVar(value=10)
        self.frames = tk.IntVar(value=1)
        self.settle = tk.DoubleVar(value=0.5)
        self.approach = tk.StringVar(value="from below")
        fields = [("Test position", self.test_pos), ("Excursion", self.excursion),
                  ("Cycles", self.cycles), ("Frames / visit", self.frames), ("Settle (s)", self.settle)]
        for r, (label, var) in enumerate(fields):
            ttk.Label(rep, text=label).grid(row=r, column=0, sticky=tk.W)
            ttk.Entry(rep, textvariable=var, width=10).grid(row=r, column=1, pady=1)
        ttk.Button(rep, text="Use current", command=self._use_current).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(rep, text="Approach").grid(row=len(fields), column=0, sticky=tk.W)
        ttk.Combobox(rep, textvariable=self.approach, values=list(APPROACHES), state="readonly",
                     width=18).grid(row=len(fields), column=1, columnspan=2, sticky=tk.W, pady=1)
        btns = ttk.Frame(rep)
        btns.grid(row=len(fields) + 1, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        self.start_btn = ttk.Button(btns, text="Start run", command=self.start_run)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.progress = tk.StringVar(value="")
        ttk.Label(rep, textvariable=self.progress, wraplength=230).grid(
            row=len(fields) + 2, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        res = ttk.LabelFrame(left, text="Results", padding=8)
        res.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(res, text="Measure here (capture + log)", command=self.measure_here).pack(fill=tk.X)
        ttk.Button(res, text="Save report", command=self.save_report).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(res, text="Clear results", command=self.clear_results).pack(fill=tk.X, pady=(4, 0))

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 5), layout="constrained")
        self.ax_fwhm, self.ax_xy, self.ax_pos = self.fig.subplots(3, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cols = ("n", "cycle", "approach", "cmd", "pos", "fwhm", "x", "y", "peak", "sat")
        widths = (40, 50, 90, 80, 80, 70, 80, 80, 60, 40)
        table = ttk.Frame(right)
        table.pack(fill=tk.X)
        self.table = ttk.Treeview(table, columns=cols, show="headings", height=8)
        for c, w in zip(cols, widths):
            self.table.heading(c, text="FWHM px" if c == "fwhm" else c)
            self.table.column(c, width=w, anchor=tk.E)
        sb = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._draw_plots()

    # -- manual control ------------------------------------------------------

    def _toggle_connect(self):
        if self.connected:
            if self.running:
                messagebox.showwarning("Stage", "Stop the run first.")
                return
            self._stage("disconnect")
        else:
            try:
                host, port = self.host.get().strip(), int(self.port.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Stage", "Port must be a number.")
                return
            self.pos_text.set(f"connecting to {host}:{port}...")
            self._stage("connect", host, port)

    def _move(self, target):
        if self.running:
            return
        try:
            self._stage("move", float(target))
        except (tk.TclError, ValueError):
            messagebox.showerror("Stage", "Position must be a number.")

    def _jog(self, sign):
        try:
            self._move(self._last_pos + sign * float(self.step.get()))
        except (AttributeError, tk.TclError, ValueError):
            messagebox.showerror("Stage", "Read the position first, and make the step a number.")

    def _use_current(self):
        if hasattr(self, "_last_pos"):
            self.test_pos.set(round(self._last_pos, 6))

    def _stage(self, op, *args):
        if op not in ("connect",) and not self.connected:
            messagebox.showerror("Stage", "Connect to the stage server first.")
            return False
        if op == "move":
            try:
                tol = abs(float(self.tol.get()))
            except (tk.TclError, ValueError):
                messagebox.showerror("Stage", "In-position tol must be a number.")
                if self.running:
                    self._abort("invalid in-position tolerance")
                return False
            self._move_target = args[0]
            args = (args[0], tol)
            self.pos_text.set(f"moving to {args[0]:g}...")
        self.worker.submit(op, *args)
        return True

    # -- repeatability run ---------------------------------------------------

    def start_run(self):
        if self.running:
            return
        if not self.connected:
            messagebox.showerror("Stage", "Connect to the stage server first.")
            return
        if self.app.spot_xy is None:
            messagebox.showerror("Stage", "Click the spot on the Image tab first so there's an aperture "
                                          "to measure.")
            return
        try:
            p = dict(test=float(self.test_pos.get()), excursion=float(self.excursion.get()),
                     cycles=int(self.cycles.get()), frames=int(self.frames.get()),
                     settle=float(self.settle.get()), approach=self.approach.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Stage", "Run settings must be numbers.")
            return
        if p["cycles"] < 1 or p["frames"] < 1 or p["settle"] < 0:
            messagebox.showerror("Stage", "Cycles and frames must be >= 1, settle >= 0.")
            return
        if p["excursion"] == 0 and p["approach"] != "from below":
            messagebox.showerror("Stage", "Approach direction needs a non-zero excursion.")
            return
        if self.rows and not messagebox.askyesno("Stage", "Start a new run? The current results table "
                                                          "will be cleared (its files stay on disk)."):
            return
        self.clear_results()
        self._new_run_dir("run")
        self.meta.update(p)
        self.meta.update(self.app.run_settings())
        self._stop_requested = False
        self._steps = self._run_steps(p)
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self._advance(None)

    def stop_run(self):
        if self.running:
            self._stop_requested = True
            self.progress.set("stopping after the current step...")

    def _run_steps(self, p):
        sides = APPROACHES[p["approach"]]
        for cycle in range(1, p["cycles"] + 1):
            side = sides[(cycle - 1) % len(sides)]
            label = "from below" if side > 0 else "from above"
            self.progress.set(f"cycle {cycle}/{p['cycles']} ({label})")
            if p["excursion"]:
                yield ("move", p["test"] - side * p["excursion"])
            yield ("move", p["test"])
            if p["settle"]:
                yield ("wait", p["settle"])
            pos = yield ("position",)
            for frame in range(p["frames"]):
                yield ("capture", dict(cycle=cycle, frame=frame, approach=label, cmd=p["test"], pos=pos))

    def _advance(self, value):
        """Feed the last step's result to the run and start its next step."""
        if self._stop_requested:
            self._finish_run("stopped")
            return
        try:
            step = self._steps.send(value)
        except StopIteration:
            self._finish_run("finished")
            return
        kind = step[0]
        if kind in ("move", "position"):
            self._stage(*step)
        elif kind == "wait":
            self.after(int(step[1] * 1000), lambda: self._advance(None))
        elif kind == "capture":
            m = step[1]
            self._capture_meta = m
            path = os.path.join(self.run_dir, f"c{m['cycle']:03d}_f{m['frame']}.fits")
            extra = {"STAGECMD": (m["cmd"], "commanded stage position (MUNIT)"),
                     "STAGEPOS": (m["pos"], "read-back stage position (MUNIT)"),
                     "CYCLE": m["cycle"], "APPROACH": m["approach"]}
            if not self.app.request_capture(path, extra):
                self._abort("camera is busy")

    def _finish_run(self, how):
        self._steps = None
        self._capture_meta = None
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        n = len(self.rows)
        try:
            self.save_report(quiet=True)
            self.progress.set(f"run {how}: {n} measurements\nreport saved to {self.run_dir}")
        except OSError as exc:
            self.progress.set(f"run {how}, but saving the report failed: {exc}")

    def _abort(self, msg):
        if self.running:
            self._finish_run("aborted")
            messagebox.showerror("Stage run aborted", msg)

    # -- hooks called by the app ---------------------------------------------

    def on_captured(self, path):
        """The app has loaded and measured a new frame."""
        if self._capture_meta is None:
            return
        m, self._capture_meta = self._capture_meta, None
        self._add_row(m, path)
        if self.running:
            self._advance(None)

    def on_capture_failed(self, msg):
        self._capture_meta = None
        self._abort(f"capture failed: {msg}")  # no-op unless a run is going

    def shutdown(self):
        self._steps = None
        self.worker.submit("quit")

    # -- manual measurement --------------------------------------------------

    def measure_here(self):
        if self.running or self._capture_meta is not None:
            return
        if self.app.spot_xy is None:
            messagebox.showerror("Stage", "Click the spot on the Image tab first.")
            return
        if self.run_dir is None:
            self._new_run_dir("manual")
            self.meta.update(self.app.run_settings())
            self.meta.update(approach="manual")
        pos = getattr(self, "_last_pos", float("nan")) if self.connected else float("nan")
        n = len(self.rows) + 1
        self._capture_meta = dict(cycle=n, frame=0, approach="manual", cmd=pos, pos=pos)
        path = os.path.join(self.run_dir, f"m{n:03d}.fits")
        extra = {"STAGEPOS": (pos, "last read stage position (MUNIT)")} if np.isfinite(pos) else None
        if not self.app.request_capture(path, extra):
            self._capture_meta = None
            messagebox.showerror("Stage", "Camera is busy.")

    # -- results -------------------------------------------------------------

    def _new_run_dir(self, kind):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.app.out_dir.get(), "stage_runs", f"{stamp}_{kind}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.meta = dict(started=datetime.now().isoformat(timespec="seconds"),
                         server=f"{self.host.get()}:{self.port.get()}", munit=self.munit)

    def _add_row(self, m, path):
        s = self.app.spot
        nan = float("nan")
        row = dict(
            n=len(self.rows) + 1, time=datetime.now().isoformat(timespec="seconds"),
            cycle=m["cycle"], frame=m["frame"], approach=m["approach"], cmd=m["cmd"], pos=m["pos"],
            pos_err=m["pos"] - m["cmd"],
            fwhm_px=s.fwhm_fit if s else nan, fwhm_x_px=s.fwhm_x if s else nan,
            fwhm_y_px=s.fwhm_y if s else nan, ee80_px=s.ee80 if s else nan,
            x=s.x if s else nan, y=s.y if s else nan, net_flux=s.net_flux if s else nan,
            peak=s.peak if s else nan, n_saturated=s.n_saturated if s else 0,
            file=os.path.basename(path),
        )
        if self.meta.get("pixel_um") is None:
            self.meta["pixel_um"] = self.app.header.get("XPIXSZ") if self.app.header is not None else None
        self.rows.append(row)
        self.table.insert("", tk.END, values=(
            row["n"], row["cycle"], row["approach"], f"{row['cmd']:g}", f"{row['pos']:g}",
            f"{row['fwhm_px']:.3f}", f"{row['x']:.2f}", f"{row['y']:.2f}", f"{row['peak']:.0f}",
            row["n_saturated"]))
        self.table.yview_moveto(1.0)
        self._write_csv()
        self._draw_plots()

    def _write_csv(self):
        with open(os.path.join(self.run_dir, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def save_report(self, quiet=False):
        if not self.rows or self.run_dir is None:
            if not quiet:
                messagebox.showinfo("Stage", "No measurements to save yet.")
            return
        self._write_csv()
        text = summarize(self.rows, self.meta)
        with open(os.path.join(self.run_dir, "summary.txt"), "w") as fh:
            fh.write(text + "\n")
        self.fig.savefig(os.path.join(self.run_dir, "report.png"), dpi=120)
        if not quiet:
            messagebox.showinfo("Stage", f"Report saved to\n{self.run_dir}\n\n{text}")

    def clear_results(self):
        if self.running:
            return
        self.rows = []
        self.run_dir = None
        self.meta = {}
        self.table.delete(*self.table.get_children())
        self.progress.set("")
        self._draw_plots()

    def _draw_plots(self):
        for ax in (self.ax_fwhm, self.ax_xy, self.ax_pos):
            ax.clear()
        if self.rows:
            n = np.array([r["n"] for r in self.rows])
            fw = np.array([r["fwhm_px"] for r in self.rows], float)
            x = np.array([r["x"] for r in self.rows], float)
            y = np.array([r["y"] for r in self.rows], float)
            err = np.array([r["pos_err"] for r in self.rows], float)
            for i, a in enumerate(sorted({r["approach"] for r in self.rows})):
                sel = np.array([r["approach"] == a for r in self.rows])
                self.ax_fwhm.plot(n[sel], fw[sel], "o", ms=4, color=f"C{i}", label=a)
            if np.isfinite(fw).any():
                self.ax_fwhm.axhline(np.nanmean(fw), color="0.5", lw=0.8, ls="--")
            self.ax_fwhm.legend(fontsize=8, loc="best")
            self.ax_xy.plot(n, x - np.nanmean(x), "o-", ms=3, lw=0.8, label="x")
            self.ax_xy.plot(n, y - np.nanmean(y), "s-", ms=3, lw=0.8, label="y")
            self.ax_xy.legend(fontsize=8, loc="best")
            self.ax_pos.plot(n, err, "o-", ms=3, lw=0.8, color="C3")
        self.ax_fwhm.set_ylabel("FWHM (px)")
        self.ax_fwhm.set_title("Spot vs. measurement", fontsize=10)
        self.ax_xy.set_ylabel("centroid - mean (px)")
        self.ax_pos.set_ylabel("pos - cmd (stage units)")
        self.ax_pos.set_xlabel("measurement #")
        self.canvas.draw_idle()

    # -- stage results -------------------------------------------------------

    def _poll(self):
        try:
            while True:
                status, op, value = self.worker.results.get_nowait()
                if status == "ok":
                    self._on_stage_ok(op, value)
                else:
                    self._on_stage_error(op, value)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _on_stage_ok(self, op, value):
        if op == "connect":
            self.connected = True
            self.munit = value["munit"] or "?"
            self._set_pos(value["pos"])
            self.connect_btn.configure(text="Disconnect")
            self._draw_plots()
        elif op in ("disconnect", "quit", "lost"):
            self.connected = False
            self.connect_btn.configure(text="Connect")
            self.pos_text.set("connection lost" if op == "lost" else "not connected")
        elif op == "move":
            pos, reached = value
            self._set_pos(pos)
            if not reached:
                msg = (f"Stage stopped at {pos:g}, commanded {self._move_target:g} "
                       f"(more than the in-position tol {self.tol.get():g} away).\n\n"
                       "Limit switch or stall? If the stage just settles a little off, "
                       "raise the in-position tol.")
                if self.running:
                    self._abort(msg)
                else:
                    messagebox.showwarning("Stage", msg)
            elif self.running:
                self._advance(pos)
        elif op == "position":
            self._set_pos(value)
            if self.running:
                self._advance(value)

    def _on_stage_error(self, op, msg):
        if op == "connect":
            self.connected = False
            self.pos_text.set("not connected")
            messagebox.showerror("Stage", f"Couldn't connect to {self.host.get()}:{self.port.get()}\n\n{msg}")
            return
        if self.running:
            self._abort(f"stage {op} failed: {msg}")
        else:
            messagebox.showerror("Stage", f"stage {op} failed:\n{msg}")

    def _set_pos(self, pos):
        self._last_pos = pos
        self.pos_text.set(f"POS {pos:g}   (MUNIT {self.munit})")
