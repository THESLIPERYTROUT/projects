"""
OurSky Resolved Imaging - desktop GUI
=====================================

A standalone Tkinter application (no web server) for setting up "resolved
imaging" runs against the OurSky admin API.

It wraps these endpoints:
    POST   /admin/v1/resolved-image-instructions   (create a run)
    GET    /admin/v1/resolved-image-instructions    (list existing)
    DELETE /admin/v1/resolved-image-instructions?id=...  (delete one)
    GET    /admin/v1/organizations                   (org picker)
    GET    /admin/v1/organization-targets?organizationId=...  (target picker)
    GET    /admin/v1/nodes?organizationId=...          (node picker)

Auth is a Bearer token with the ADMIN role.

Requirements:
    Python 3.9+ and the `requests` package (pip install requests).
    Tkinter ships with the standard Windows/macOS python.org installers.

Run:
    python resolved_imaging_gui.py
"""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from datetime import datetime, timedelta, timezone

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

try:
    import requests
except ImportError:  # pragma: no cover - friendly message instead of a stack trace
    import sys
    import tkinter.messagebox as mb

    _r = tk.Tk()
    _r.withdraw()
    mb.showerror(
        "Missing dependency",
        "This app needs the 'requests' package.\n\n"
        "Install it with:\n    pip install requests\n\nThen run the app again.",
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Known API base URLs. The base is editable in the UI, so if the admin endpoints
# live somewhere else for your tenant you can just type it in.
ENVIRONMENTS = {
    "Production": "https://api.prod.oursky.ai",
    "Development": "https://api.dev.oursky.ai",
    "Production (manage host)": "https://api-manage.prod.oursky.online",
    "Custom": "",
}

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oursky_resolved_imaging.json")

REQUEST_TIMEOUT = 30  # seconds


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

class ApiError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class OurSkyClient:
    """Thin wrapper around the OurSky admin endpoints."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # -- low level ---------------------------------------------------------- #

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        if not resp.ok:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise ApiError(resp.status_code, body)
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # -- pickers ------------------------------------------------------------ #

    def list_organizations(self):
        return self._request("GET", "/admin/v1/organizations") or []

    def list_targets(self, organization_id: str):
        return (
            self._request(
                "GET",
                "/admin/v1/organization-targets",
                params={"organizationId": organization_id},
            )
            or []
        )

    def list_nodes(self, organization_id: str):
        data = self._request(
            "GET", "/admin/v1/nodes", params={"organizationId": organization_id}
        )
        return (data or {}).get("nodes", []) if isinstance(data, dict) else []

    # -- resolved image instructions --------------------------------------- #

    def create_instructions(self, payload: dict):
        return self._request(
            "POST", "/admin/v1/resolved-image-instructions", data=json.dumps(payload)
        )

    def list_instructions(self, before: str | None = None):
        params = {"before": before} if before else None
        return self._request(
            "GET", "/admin/v1/resolved-image-instructions", params=params
        ) or []

    def delete_instruction(self, instruction_id: str):
        return self._request(
            "DELETE",
            "/admin/v1/resolved-image-instructions",
            params={"id": instruction_id},
        )

    # -- diagnostics -------------------------------------------------------- #

    def max_elevation_passes(self, payload: dict):
        """Upcoming max-elevation passes for a target (used to diagnose 400s)."""
        return self._request(
            "POST", "/admin/v1/maxElevation-for-target", data=json.dumps(payload)
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_iso(dt: datetime) -> str:
    """RFC3339 / ISO-8601 with a trailing Z, e.g. 2025-11-11T13:40:00Z."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string (accepting a trailing Z) into an aware datetime."""
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def target_label(t: dict) -> str:
    """Human-friendly label for an organization-target."""
    sat = t.get("satelliteTarget") or {}
    name = sat.get("tleName") or "(unnamed)"
    norad = sat.get("noradId")
    bits = [name.strip()]
    if norad:
        bits.append(f"NORAD {norad}")
    if t.get("resolvedImagingEnabled"):
        bits.append("RI:on")
    else:
        bits.append("RI:off")
    return "  -  ".join(bits)


def node_label(n: dict) -> str:
    node = n.get("node") or n  # nodes come wrapped in {node, location}
    name = node.get("name") or node.get("canonicalName") or "(unnamed node)"
    return name


def org_label(o: dict) -> str:
    return o.get("description") or o.get("domain") or o.get("ownerEmail") or o.get("id", "?")


# --------------------------------------------------------------------------- #
# Searchable combobox
# --------------------------------------------------------------------------- #

class SearchableCombobox(ttk.Combobox):
    """An editable combobox you can type into to filter its options.

    Backing items and a label function are stored, and the visible option list
    is narrowed case-insensitively as you type. Selection is resolved against
    the *currently visible* rows by index, so filtering never picks the wrong
    item even when two labels are identical.
    """

    _NAV_KEYS = {
        "Up", "Down", "Return", "Escape", "Left", "Right", "Tab",
        "Shift_L", "Shift_R", "Control_L", "Control_R", "Home", "End",
    }

    def __init__(self, master, label_fn, on_select=None, **kw):
        kw.setdefault("state", "normal")
        super().__init__(master, **kw)
        self._label_fn = label_fn
        self._on_select = on_select
        self._pairs = []          # full list: [(label, item), ...]
        self._visible = []        # currently shown subset, same shape
        self._selected = None     # last explicitly selected item
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<<ComboboxSelected>>", self._on_pick)
        self.bind("<Down>", self._on_down)

    def set_items(self, items, select_first=True):
        self._pairs = [(self._label_fn(it), it) for it in items]
        self._visible = list(self._pairs)
        self["values"] = [lbl for lbl, _ in self._pairs]
        if self._pairs and select_first:
            self.current(0)
            self._selected = self._pairs[0][1]
        else:
            self.set("")
            self._selected = None

    def selected_item(self):
        # Prefer the explicitly picked row (handles duplicate labels precisely);
        # otherwise resolve an exactly-typed label so typing then tabbing away
        # still counts as a selection.
        text = self.get()
        if self._selected is not None and self._label_fn(self._selected) == text:
            return self._selected
        for lbl, it in self._pairs:
            if lbl == text:
                return it
        return self._selected

    def _apply_filter(self, text):
        low = text.lower().strip()
        if low:
            self._visible = [p for p in self._pairs if low in p[0].lower()]
        else:
            self._visible = list(self._pairs)
        self["values"] = [lbl for lbl, _ in self._visible]

    def _on_key(self, event):
        if event.keysym in self._NAV_KEYS:
            return
        self._apply_filter(self.get())
        self._refresh_open_dropdown(open_if_closed=True)

    def _on_pick(self, _event=None):
        idx = self.current()
        if 0 <= idx < len(self._visible):
            self._selected = self._visible[idx][1]
        if self._on_select:
            self._on_select()

    # -- live dropdown while typing ------------------------------------- #

    def _popdown(self):
        """Tk path of the internal popdown toplevel, or None if unavailable."""
        try:
            return str(self.tk.call("ttk::combobox::PopdownWindow", self))
        except tk.TclError:
            return None

    def _refresh_open_dropdown(self, open_if_closed=False):
        """Rebuild the dropdown list in place so filtering is visible live.

        Tk only copies -values into the popdown listbox at Post time, so an
        open list goes stale when we filter; rewrite the listbox directly.
        Uses ttk internals; degrades to filter-on-next-open if they change.
        """
        popdown = self._popdown()
        if popdown is None:
            return
        try:
            is_open = bool(int(self.tk.call("winfo", "ismapped", popdown)))
            if not is_open:
                if not open_if_closed or not self._visible:
                    return
                self.tk.call("ttk::combobox::Post", self)
            lb = f"{popdown}.f.l"
            self.tk.call(lb, "delete", 0, "end")
            for lbl, _item in self._visible:
                self.tk.call(lb, "insert", "end", lbl)
            self.tk.call(lb, "selection", "clear", 0, "end")
            self.tk.call(lb, "configure", "-height",
                         max(1, min(len(self._visible), 10)))
            self.update_idletasks()
            self.tk.call("ttk::combobox::PlacePopdown", self, popdown)
            # The listbox force-grabs keyboard focus when first mapped; take it
            # back so typing continues here (grabs only affect pointer events).
            self.after_idle(self.focus_set)
        except tk.TclError:
            pass

    def _on_down(self, _event):
        """Down-arrow hands focus from the entry to the open dropdown list."""
        popdown = self._popdown()
        if popdown is None:
            return None
        try:
            if int(self.tk.call("winfo", "ismapped", popdown)):
                lb = f"{popdown}.f.l"
                self.tk.call(lb, "selection", "clear", 0, "end")
                self.tk.call(lb, "selection", "set", 0)
                self.tk.call(lb, "activate", 0)
                self.tk.call("focus", lb)
                return "break"  # already open: don't let Tk re-post
        except tk.TclError:
            pass
        return None  # closed: fall through to Tk's default open-on-Down


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

class ResolvedImagingApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        self.master = master
        self.grid(sticky="nsew")

        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)  # notebook
        self.rowconfigure(3, weight=1)  # log

        self.client: OurSkyClient | None = None
        self._ui_queue: "queue.Queue" = queue.Queue()

        # data caches
        self.orgs: list[dict] = []
        self.targets: list[dict] = []
        self.nodes: list[dict] = []
        self.instructions: list[dict] = []

        self._build_connection_bar()
        self._build_notebook()
        self._build_log()
        self._build_status_bar()

        self._load_config()
        self._poll_ui_queue()

    # ------------------------------------------------------------------ #
    # Connection bar
    # ------------------------------------------------------------------ #

    def _build_connection_bar(self):
        frame = ttk.LabelFrame(self, text="Connection", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for c in (1, 3):
            frame.columnconfigure(c, weight=1)

        ttk.Label(frame, text="Environment:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.env_var = tk.StringVar(value="Production")
        env_combo = ttk.Combobox(
            frame,
            textvariable=self.env_var,
            values=list(ENVIRONMENTS.keys()),
            state="readonly",
            width=22,
        )
        env_combo.grid(row=0, column=1, sticky="w")
        env_combo.bind("<<ComboboxSelected>>", self._on_env_changed)

        ttk.Label(frame, text="Base URL:").grid(row=0, column=2, sticky="w", padx=(12, 4))
        self.base_url_var = tk.StringVar(value=ENVIRONMENTS["Production"])
        ttk.Entry(frame, textvariable=self.base_url_var).grid(row=0, column=3, sticky="ew")

        ttk.Label(frame, text="Bearer token:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.token_var = tk.StringVar()
        self.token_entry = ttk.Entry(frame, textvariable=self.token_var, show="*")
        self.token_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))

        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        self.show_token_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btns,
            text="Show token",
            variable=self.show_token_var,
            command=self._toggle_token,
        ).pack(side="left")

        self.remember_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btns,
            text="Remember token (plaintext on disk)",
            variable=self.remember_var,
        ).pack(side="left", padx=(12, 0))

        self.connect_btn = ttk.Button(btns, text="Connect", command=self.on_connect)
        self.connect_btn.pack(side="right")

    def _toggle_token(self):
        self.token_entry.configure(show="" if self.show_token_var.get() else "*")

    def _on_env_changed(self, _event=None):
        env = self.env_var.get()
        url = ENVIRONMENTS.get(env, "")
        if env != "Custom":
            self.base_url_var.set(url)

    # ------------------------------------------------------------------ #
    # Notebook (Create / Manage)
    # ------------------------------------------------------------------ #

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        self._build_create_tab()
        self._build_manage_tab()

    def _build_create_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Create Run")
        tab.columnconfigure(1, weight=1)

        row = 0

        # Organization ------------------------------------------------- #
        ttk.Label(tab, text="Organization:").grid(row=row, column=0, sticky="w", pady=4)
        self.org_var = tk.StringVar()
        self.org_combo = SearchableCombobox(
            tab, label_fn=org_label, on_select=self._on_org_selected,
            textvariable=self.org_var,
        )
        self.org_combo.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(tab, text="Reload", command=self.load_orgs).grid(
            row=row, column=2, padx=(6, 0)
        )
        row += 1

        # Target ------------------------------------------------------- #
        ttk.Label(tab, text="Target:").grid(row=row, column=0, sticky="w", pady=4)
        self.target_var = tk.StringVar()
        self.target_combo = SearchableCombobox(
            tab, label_fn=target_label, on_select=self._show_target_id,
            textvariable=self.target_var,
        )
        self.target_combo.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(tab, text="Reload", command=self.load_targets).grid(
            row=row, column=2, padx=(6, 0)
        )
        row += 1

        ttk.Label(
            tab,
            text="Type to search - the open list filters live; ↓ moves into the list, Enter/click picks.",
            foreground="#666",
        ).grid(row=row, column=1, sticky="w")
        row += 1

        # Target ID source toggle (org-target / satellite-target / custom) #
        ttk.Label(tab, text="Use ID:").grid(row=row, column=0, sticky="w")
        idkind = ttk.Frame(tab)
        idkind.grid(row=row, column=1, columnspan=2, sticky="ew")
        self.target_id_kind = tk.StringVar(value="org")
        ttk.Radiobutton(
            idkind, text="Organization-target id", value="org",
            variable=self.target_id_kind, command=self._show_target_id,
        ).pack(side="left")
        ttk.Radiobutton(
            idkind, text="Satellite-target id", value="satellite",
            variable=self.target_id_kind, command=self._show_target_id,
        ).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(
            idkind, text="Custom:", value="custom",
            variable=self.target_id_kind, command=self._show_target_id,
        ).pack(side="left", padx=(8, 0))
        self.custom_target_id_var = tk.StringVar()
        self.custom_target_entry = ttk.Entry(
            idkind, textvariable=self.custom_target_id_var, state="disabled"
        )
        self.custom_target_entry.pack(side="left", padx=(4, 0), fill="x", expand=True)
        self.custom_target_entry.bind(
            "<KeyRelease>", lambda e: self._show_target_id()
        )
        row += 1

        self.target_id_preview = ttk.Label(tab, text="", foreground="#666")
        self.target_id_preview.grid(row=row, column=1, columnspan=2, sticky="w")
        row += 1

        # Node --------------------------------------------------------- #
        ttk.Label(tab, text="Node (optional):").grid(row=row, column=0, sticky="w", pady=4)
        self.node_var = tk.StringVar()
        self.node_combo = ttk.Combobox(tab, textvariable=self.node_var, state="readonly")
        self.node_combo.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(tab, text="Reload", command=self.load_nodes).grid(
            row=row, column=2, padx=(6, 0)
        )
        row += 1

        ttk.Separator(tab, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # Until -------------------------------------------------------- #
        ttk.Label(tab, text="Until (UTC):").grid(row=row, column=0, sticky="w", pady=4)
        until_frame = ttk.Frame(tab)
        until_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        until_frame.columnconfigure(0, weight=1)
        self.until_var = tk.StringVar(value=to_iso(now_utc() + timedelta(days=7)))
        ttk.Entry(until_frame, textvariable=self.until_var).grid(row=0, column=0, sticky="ew")
        for i, days in enumerate((1, 7, 14, 30), start=1):
            ttk.Button(
                until_frame,
                text=f"+{days}d",
                width=5,
                command=lambda d=days: self.until_var.set(to_iso(now_utc() + timedelta(days=d))),
            ).grid(row=0, column=i, padx=(4, 0))
        row += 1

        ttk.Label(
            tab,
            text="Must be within the next 30 days  (e.g. 2026-06-26T00:00:00Z)",
            foreground="#666",
        ).grid(row=row, column=1, columnspan=2, sticky="w")
        row += 1

        # Attempt limit ------------------------------------------------ #
        ttk.Label(tab, text="Attempt limit:").grid(row=row, column=0, sticky="w", pady=4)
        self.attempt_var = tk.IntVar(value=5)
        ttk.Spinbox(tab, from_=1, to=10, textvariable=self.attempt_var, width=8).grid(
            row=row, column=1, sticky="w", pady=4
        )
        ttk.Label(tab, text="Max instructions to generate (1-10).", foreground="#666").grid(
            row=row, column=1, sticky="e"
        )
        row += 1

        # Min elevation ------------------------------------------------ #
        ttk.Label(tab, text="Min elevation (deg):").grid(row=row, column=0, sticky="w", pady=4)
        self.elevation_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(
            tab, from_=0, to=90, increment=1, textvariable=self.elevation_var, width=8
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(
            tab, text="Minimum pass elevation above the horizon.", foreground="#666"
        ).grid(row=row, column=1, sticky="e")
        row += 1

        ttk.Separator(tab, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        action = ttk.Frame(tab)
        action.grid(row=row, column=0, columnspan=3, sticky="ew")
        action.columnconfigure(0, weight=1)
        ttk.Button(action, text="Check passes", command=self.on_check_passes).grid(
            row=0, column=1, padx=(0, 6)
        )
        ttk.Button(action, text="Preview JSON", command=self.preview_payload).grid(
            row=0, column=2, padx=(0, 6)
        )
        self.create_btn = ttk.Button(
            action, text="Create Instructions", command=self.on_create
        )
        self.create_btn.grid(row=0, column=3)

    def _build_manage_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Manage Instructions")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(top, text="Before (UTC, optional):").pack(side="left")
        self.before_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.before_var, width=24).pack(side="left", padx=6)
        ttk.Button(top, text="Refresh list", command=self.load_instructions).pack(side="left")
        ttk.Button(top, text="Delete selected", command=self.on_delete).pack(
            side="right"
        )

        columns = ("id", "targetId", "nodeId", "maxElevation", "timeOfMaxElevation", "createdAt")
        self.tree = ttk.Treeview(tab, columns=columns, show="headings", selectmode="browse")
        widths = {
            "id": 230,
            "targetId": 230,
            "nodeId": 230,
            "maxElevation": 90,
            "timeOfMaxElevation": 170,
            "createdAt": 170,
        }
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=yscroll.set)

    # ------------------------------------------------------------------ #
    # Log + status bar
    # ------------------------------------------------------------------ #

    def _build_log(self):
        frame = ttk.LabelFrame(self, text="Activity log", padding=4)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(frame, height=8, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Not connected.")
        bar = ttk.Frame(self)
        bar.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        bar.columnconfigure(0, weight=1)
        ttk.Label(bar, textvariable=self.status_var, anchor="w").grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------ #
    # Logging helpers (thread-safe via the queue)
    # ------------------------------------------------------------------ #

    def log_msg(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, text: str):
        self.status_var.set(text)

    def _busy(self, on: bool):
        if on:
            self.progress.start(12)
        else:
            self.progress.stop()

    # ------------------------------------------------------------------ #
    # Background-task plumbing
    # ------------------------------------------------------------------ #

    def run_async(self, fn, on_success=None, busy_msg="Working...", success_msg=None):
        """Run `fn` on a worker thread; marshal the result back to the UI thread."""
        if self.client is None:
            messagebox.showwarning("Not connected", "Click Connect first.")
            return

        self.set_status(busy_msg)
        self._busy(True)

        def worker():
            try:
                result = fn()
                self._ui_queue.put(("ok", result, on_success, success_msg))
            except ApiError as e:
                self._ui_queue.put(("err", e, None, None))
            except requests.RequestException as e:
                self._ui_queue.put(("err", e, None, None))
            except Exception as e:  # noqa: BLE001 - surface anything to the user
                self._ui_queue.put(("err", e, None, None))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload, on_success, success_msg = self._ui_queue.get_nowait()
                self._busy(False)
                if kind == "ok":
                    if on_success:
                        on_success(payload)
                    if success_msg:
                        self.set_status(success_msg)
                        self.log_msg(success_msg)
                else:  # error
                    self._handle_error(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _handle_error(self, error):
        if isinstance(error, ApiError):
            body = error.body
            if isinstance(body, (dict, list)):
                body = json.dumps(body, indent=2)
            msg = f"HTTP {error.status}\n{body}"
        else:
            msg = str(error)
        self.set_status("Error.")
        self.log_msg("ERROR: " + msg.replace("\n", " | "))
        messagebox.showerror("Request failed", msg)

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def on_connect(self):
        base = self.base_url_var.get().strip()
        token = self.token_var.get().strip()
        if not base:
            messagebox.showwarning("Missing base URL", "Enter an API base URL.")
            return
        if not token:
            messagebox.showwarning("Missing token", "Enter a Bearer token.")
            return

        self.client = OurSkyClient(base, token)
        self.log_msg(f"Connecting to {base} ...")
        self._save_config()
        self.load_orgs(after_connect=True)

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    def load_orgs(self, after_connect=False):
        def done(orgs):
            self.orgs = orgs or []
            self.org_combo.set_items(self.orgs)
            if self.orgs:
                self._on_org_selected()
            verb = "Connected" if after_connect else "Reloaded orgs"
            self.set_status(f"{verb}. {len(self.orgs)} organization(s).")
            self.log_msg(f"Loaded {len(self.orgs)} organization(s).")

        self.run_async(
            lambda: self.client.list_organizations(),
            on_success=done,
            busy_msg="Loading organizations...",
        )

    def _selected_org_id(self):
        org = self.org_combo.selected_item()
        return org.get("id") if org else None

    def _on_org_selected(self, _event=None):
        if self._selected_org_id():
            self.load_targets()
            self.load_nodes()

    def load_targets(self):
        org_id = self._selected_org_id()
        if not org_id:
            messagebox.showwarning("No organization", "Select an organization first.")
            return

        def done(targets):
            self.targets = targets or []
            self.target_combo.set_items(self.targets)
            self._show_target_id()
            self.log_msg(f"Loaded {len(self.targets)} target(s).")

        self.run_async(
            lambda: self.client.list_targets(org_id),
            on_success=done,
            busy_msg="Loading targets...",
        )

    def load_nodes(self):
        org_id = self._selected_org_id()
        if not org_id:
            return

        def done(nodes):
            self.nodes = nodes or []
            labels = ["(Any node - let the server choose)"] + [
                node_label(n) for n in self.nodes
            ]
            self.node_combo["values"] = labels
            self.node_combo.current(0)
            self.log_msg(f"Loaded {len(self.nodes)} node(s).")

        self.run_async(
            lambda: self.client.list_nodes(org_id),
            on_success=done,
            busy_msg="Loading nodes...",
        )

    # ------------------------------------------------------------------ #
    # Build + validate payload
    # ------------------------------------------------------------------ #

    def _selected_target(self):
        return self.target_combo.selected_item()

    def _selected_target_id(self):
        kind = self.target_id_kind.get()
        if kind == "custom":
            return self.custom_target_id_var.get().strip() or None
        t = self._selected_target()
        if not t:
            return None
        if kind == "satellite":
            return (t.get("satelliteTarget") or {}).get("id")
        return t.get("id")

    def _show_target_id(self):
        kind = self.target_id_kind.get()
        self.custom_target_entry.configure(
            state="normal" if kind == "custom" else "disabled"
        )
        tid = self._selected_target_id()
        self.target_id_preview.configure(text=f"{kind}: {tid or '-'}")

    def _selected_node_id(self):
        idx = self.node_combo.current()
        # index 0 is the "Any node" sentinel
        if idx <= 0 or (idx - 1) >= len(self.nodes):
            return None
        node = self.nodes[idx - 1]
        inner = node.get("node") or node
        return inner.get("id")

    def build_payload(self):
        target_id = self._selected_target_id()
        if not target_id:
            raise ValueError("Select a target (or enter a custom target ID).")
        if self.target_id_kind.get() == "custom":
            try:
                uuid.UUID(target_id)
            except ValueError:
                raise ValueError("Custom target ID must be a valid UUID.")

        # Validate `until`.
        try:
            until_dt = parse_iso(self.until_var.get())
        except ValueError:
            raise ValueError(
                "Could not parse 'Until'. Use ISO-8601, e.g. 2026-06-26T00:00:00Z"
            )
        now = now_utc()
        if until_dt <= now:
            raise ValueError("'Until' must be in the future.")
        if until_dt > now + timedelta(days=30):
            raise ValueError("'Until' must be within the next 30 days.")

        attempt = int(self.attempt_var.get())
        if not 1 <= attempt <= 10:
            raise ValueError("Attempt limit must be between 1 and 10.")

        elevation = float(self.elevation_var.get())
        if not 0 <= elevation <= 90:
            raise ValueError("Min elevation must be between 0 and 90 degrees.")

        # Always send all five keys (nodeId as null when no node is chosen).
        # Key order matches the API docs example; some generated servers 400 on
        # a missing nullable field, so we never omit nodeId.
        payload = {
            "attemptLimit": attempt,
            "minElevation": elevation,
            "nodeId": self._selected_node_id(),  # None -> JSON null
            "targetId": target_id,
            "until": to_iso(until_dt),
        }
        return payload

    def preview_payload(self):
        try:
            payload = self.build_payload()
        except ValueError as e:
            messagebox.showwarning("Cannot build request", str(e))
            return
        self.log_msg("Preview payload:\n" + json.dumps(payload, indent=2))
        messagebox.showinfo("Request payload", json.dumps(payload, indent=2))

    # ------------------------------------------------------------------ #
    # Diagnostics: check whether the target actually has qualifying passes
    # ------------------------------------------------------------------ #

    def on_check_passes(self):
        """Call /admin/v1/maxElevation-for-target with the same target/node/elevation.

        A 400 from the create endpoint is most often "no qualifying passes" or a
        target that can't be imaged. This isolates that without creating anything.
        """
        target_id = self._selected_target_id()
        if not target_id:
            messagebox.showwarning(
                "No target", "Select a target (or enter a custom target ID) first."
            )
            return
        if self.target_id_kind.get() == "custom":
            try:
                uuid.UUID(target_id)
            except ValueError:
                messagebox.showwarning(
                    "Bad target ID", "Custom target ID must be a valid UUID."
                )
                return
        try:
            until_dt = parse_iso(self.until_var.get())
            elevation = float(self.elevation_var.get())
        except ValueError:
            messagebox.showwarning("Bad input", "Check the 'Until' and 'Min elevation' fields.")
            return

        # maxElevation-for-target requires targetId, start, end; nodeId is
        # optional and NOT nullable here, so omit it rather than send null.
        payload = {
            "targetId": target_id,
            "start": to_iso(now_utc()),
            "end": to_iso(until_dt),
            "minElevation": elevation,
        }
        node_id = self._selected_node_id()
        if node_id:
            payload["nodeId"] = node_id

        self.log_msg("POST /admin/v1/maxElevation-for-target\n" + json.dumps(payload, indent=2))

        def done(result):
            count = len(result) if isinstance(result, list) else (
                len(result.get("passes", [])) if isinstance(result, dict) else 0
            )
            self.log_msg("Passes response:\n" + json.dumps(result, indent=2))
            if count:
                self.set_status(f"Found {count} qualifying pass(es).")
                messagebox.showinfo(
                    "Passes found",
                    f"Found {count} pass(es) >= {elevation} deg before the 'Until' time.\n\n"
                    "The target/node/elevation look fine, so a create 400 is likely a "
                    "different rule (e.g. resolved imaging not enabled for this target).",
                )
            else:
                self.set_status("No qualifying passes.")
                messagebox.showwarning(
                    "No passes",
                    f"No passes >= {elevation} deg before the 'Until' time for this "
                    "target/node.\n\nThis is the most likely cause of a 400 on create. "
                    "Try lowering 'Min elevation', extending 'Until', or choosing "
                    "'Any node'.",
                )

        self.run_async(
            lambda: self.client.max_elevation_passes(payload),
            on_success=done,
            busy_msg="Checking passes...",
        )

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #

    def on_create(self):
        try:
            payload = self.build_payload()
        except ValueError as e:
            messagebox.showwarning("Cannot build request", str(e))
            return

        # RI-disabled warning only applies when the ID comes from the dropdown;
        # a custom ID may not correspond to the selected row at all.
        sel = None if self.target_id_kind.get() == "custom" else self._selected_target()
        if sel is not None and sel.get("resolvedImagingEnabled") is False:
            if not messagebox.askyesno(
                "Resolved imaging disabled",
                "This target has resolvedImagingEnabled = false, which is a common "
                "cause of a 400 on create.\n\nTry anyway?",
            ):
                return

        if self.target_id_kind.get() == "custom":
            tgt = f"custom id {payload['targetId']}"
        else:
            tgt = self.target_var.get()
        node = self.node_var.get()
        if not messagebox.askyesno(
            "Confirm",
            "Create resolved imaging instructions?\n\n"
            f"Target:  {tgt}\n"
            f"Node:    {node}\n"
            f"Until:   {payload['until']}\n"
            f"Attempt limit: {payload['attemptLimit']}\n"
            f"Min elevation: {payload['minElevation']} deg",
        ):
            return

        self.log_msg("POST /admin/v1/resolved-image-instructions\n" + json.dumps(payload, indent=2))

        def done(result):
            count = len(result) if isinstance(result, list) else 1
            self.log_msg(f"Created {count} instruction(s):\n" + json.dumps(result, indent=2))
            self.set_status(f"Created {count} instruction(s).")
            messagebox.showinfo(
                "Success",
                f"Created {count} instruction(s). See the activity log for details.",
            )

        self.run_async(
            lambda: self.client.create_instructions(payload),
            on_success=done,
            busy_msg="Creating instructions...",
        )

    # ------------------------------------------------------------------ #
    # Manage (list / delete)
    # ------------------------------------------------------------------ #

    def load_instructions(self):
        before = self.before_var.get().strip() or None
        if before:
            try:
                before = to_iso(parse_iso(before))
            except ValueError:
                messagebox.showwarning("Bad date", "'Before' must be ISO-8601 or blank.")
                return

        def done(items):
            self.instructions = items or []
            self.tree.delete(*self.tree.get_children())
            for it in self.instructions:
                self.tree.insert(
                    "",
                    "end",
                    iid=it.get("id"),
                    values=(
                        it.get("id", ""),
                        it.get("targetId", ""),
                        it.get("nodeId", ""),
                        it.get("maxElevation", ""),
                        it.get("timeOfMaxElevation", ""),
                        it.get("createdAt", ""),
                    ),
                )
            self.set_status(f"Loaded {len(self.instructions)} instruction(s).")
            self.log_msg(f"Loaded {len(self.instructions)} instruction(s).")

        self.run_async(
            lambda: self.client.list_instructions(before),
            on_success=done,
            busy_msg="Loading instructions...",
        )

    def on_delete(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No selection", "Select an instruction to delete.")
            return
        instruction_id = selection[0]
        if not messagebox.askyesno(
            "Confirm delete", f"Delete instruction\n{instruction_id}?"
        ):
            return

        def done(_):
            self.tree.delete(instruction_id)
            self.set_status("Instruction deleted.")
            self.log_msg(f"Deleted instruction {instruction_id}.")

        self.run_async(
            lambda: self.client.delete_instruction(instruction_id),
            on_success=done,
            busy_msg="Deleting instruction...",
        )

    # ------------------------------------------------------------------ #
    # Config persistence (base URL, env, optional token)
    # ------------------------------------------------------------------ #

    def _load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return
        if cfg.get("env") in ENVIRONMENTS:
            self.env_var.set(cfg["env"])
        if cfg.get("base_url"):
            self.base_url_var.set(cfg["base_url"])
        if cfg.get("token"):
            self.token_var.set(cfg["token"])
            self.remember_var.set(True)

    def _save_config(self):
        cfg = {
            "env": self.env_var.get(),
            "base_url": self.base_url_var.get().strip(),
        }
        if self.remember_var.get():
            cfg["token"] = self.token_var.get().strip()
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except OSError as e:
            self.log_msg(f"Could not save config: {e}")


def main():
    root = tk.Tk()
    root.title("OurSky - Resolved Imaging")
    root.geometry("900x760")
    root.minsize(760, 640)
    ResolvedImagingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
