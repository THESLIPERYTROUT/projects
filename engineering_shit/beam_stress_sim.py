import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

#Units: SI (m, N, N·m, Pa)
#
# All case settings (beam, material, supports, geometry, loads) live in a TOML
# case file so different sim cases can be saved independently of this program.
# See beam_cases/gear_shaft.toml for a documented example.
#
#   python beam_stress_sim.py                       -> runs DEFAULT_CASE
#   python beam_stress_sim.py path/to/case.toml     -> runs that case

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE = SCRIPT_DIR / "beam_cases" / "gear_shaft.toml"

# Multipliers to convert a gear's "power" value to watts, keyed by "power_unit".
POWER_UNITS = {
    "W":  1.0,
    "kW": 1e3,
    "MW": 1e6,
    "hp": 745.69987,   # mechanical / imperial horsepower
    "PS": 735.49875,   # metric horsepower
}

def _require(table, key, section):
    if key not in table:
        raise KeyError(f"Config section [{section}] is missing required key '{key}'")
    return table[key]

def load_case(path):
    """Read a TOML case file and return a validated settings dict."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    for section in ("beam", "material", "supports", "geometry", "loads"):
        if section not in raw:
            raise KeyError(f"Config file {path} is missing required section [{section}]")

    beam = raw["beam"]
    length = float(_require(beam, "length", "beam"))
    mesh_density_factor = float(_require(beam, "mesh_density_factor", "beam"))
    if length <= 0 or mesh_density_factor <= 0:
        raise ValueError("[beam] length and mesh_density_factor must be positive")

    material = raw["material"]
    material_properties = {
        "young_modulus": float(_require(material, "young_modulus", "material")),
        "poisson_ratio": float(material.get("poisson_ratio", 0.3)),
        "density":       float(material.get("density", 7850)),
    }
    yield_strength = float(_require(material, "yield_strength", "material"))

    design = raw.get("design", {})
    target_fos = design.get("target_factor_of_safety", 2)

    supports = [dict(s) for s in raw["supports"]]
    for s in supports:
        _require(s, "position", "supports")
        _require(s, "type", "supports")

    geometry = sorted((dict(g) for g in raw["geometry"]), key=lambda g: g["start"])
    for g in geometry:
        for key in ("start", "end", "diameter"):
            _require(g, key, "geometry")
    if not np.isclose(geometry[0]["start"], 0.0):
        raise ValueError("First [[geometry]] segment must start at 0")
    if not np.isclose(geometry[-1]["end"], length):
        raise ValueError(f"Last [[geometry]] segment must end at the beam length ({length})")
    for a, b in zip(geometry, geometry[1:]):
        if not np.isclose(a["end"], b["start"]):
            raise ValueError(f"[[geometry]] segments are not contiguous at x={a['end']} / {b['start']}")

    loads = []
    for raw_load in raw["loads"]:
        load = dict(raw_load)
        _require(load, "type", "loads")
        _require(load, "position", "loads")
        if load["type"] == "driving gear":
            unit = str(load.get("power_unit", "W"))
            if unit not in POWER_UNITS:
                raise ValueError(
                    f"Load at x={load['position']}: unknown power_unit '{unit}' "
                    f"(choose from {', '.join(POWER_UNITS)})"
                )
            load["power_unit"] = unit
            load["power_input"] = float(_require(load, "power", "loads"))  # as written in the case file
            load["power"] = load["power_input"] * POWER_UNITS[unit]        # solver works in W
        # TOML arrays come in as lists; the solver expects (Fx, Fy, Fz) / (Mx, My, Mz) tuples
        load["force"]  = tuple(float(v) for v in load.get("force",  (0, 0, 0)))
        load["moment"] = tuple(float(v) for v in load.get("moment", (0, 0, 0)))
        if len(load["force"]) != 3 or len(load["moment"]) != 3:
            raise ValueError(f"Load at x={load['position']}: force/moment must have 3 components")
        loads.append(load)

    for item in supports + loads:
        if not 0 <= item["position"] <= length:
            raise ValueError(f"Position {item['position']} is outside the beam [0, {length}]")

    return {
        "case_path": path,
        "length": length,
        "mesh_density_factor": mesh_density_factor,
        "simulated_points": int(length / mesh_density_factor),
        "supports": supports,
        "material_properties": material_properties,
        "yield_strength": yield_strength,
        "target_factor_of_safety": target_fos,
        "geometry": geometry,
        "loads": loads,
    }

def prepare_geometry(geometry):
    """Pre-compute cross-section properties for each segment."""
    for seg in geometry:
        d = seg["diameter"]
        seg["I"] = np.pi * d**4 / 64
        seg["J"] = np.pi * d**4 / 32
        seg["A"] = np.pi * d**2 / 4
        seg["c"] = d / 2

def section_at(x, geometry):
    """Return the geometry segment containing position x."""
    for seg in geometry:
        if seg["start"] <= x <= seg["end"]:
            return seg
    raise ValueError(f"x={x:.4f} is outside the beam geometry [{geometry[0]['start']}, {geometry[-1]['end']}]")

def S(x, a, n):
    """
    Singularity function <x - a>^n

    n < 0  : not integrated (Dirac / doublet) — returns 1 if x == a else 0
    n >= 0 : Macaulay bracket — returns (x-a)^n if x >= a, else 0
             (right-continuous, so a load at x = a is counted at x = a)
    """
    if n < 0:
        return 1.0 if np.isclose(x, a) else 0.0
    else:
        return (x - a)**n if x >= a else 0.0

def build_load_list(loads, supports):
    """
    Returns a unified list of all loads including reactions.
    Reactions are solved here from equilibrium, then injected back
    as point loads/moments so the singularity engine treats everything uniformly.
    """

    all_loads = loads.copy()

    driving_gear = next((load for load in all_loads if load["type"] == "driving gear"), None)

    for load in all_loads:
        if load["type"] in ("driving gear", "driven gear"):
            if driving_gear is None:
                raise ValueError("A 'driven gear' load requires a 'driving gear' load to supply shaft power/speed.")

            P = driving_gear["power"]
            N = driving_gear["speed"]
            d = load["diameter"] * 1000

            Wt = (60000 * P) / (d * np.pi * N)
            Wn = Wt / np.tan(np.radians(load["tooth_angle"]))
            load["force"] = (0, Wn, Wt)

            T = (d / 2) * Wt
           
            load["moment"] = (T, 0, 0) if load["type"] == "driving gear" else (-T, 0, 0)
        elif load["type"] == "point load":
            pass
        elif load["type"] == "point moment":
            pass
        else:
            raise NotImplementedError(f"Load type '{load['type']}' not implemented")

    statics_solver_matrix = []
    unknowns = []

    for support in supports:
        if support["type"] == "bearing":
            d = support["position"]

            col_Fy = [0,  1,  0,  0,  0,  d]
            col_Fz = [0,  0,  1,  0, -d,  0]

            statics_solver_matrix.append(col_Fy)
            statics_solver_matrix.append(col_Fz)
            unknowns.append({"dof": "Fy", "position": d})
            unknowns.append({"dof": "Fz", "position": d})
        elif support["type"] == "fixed":
            d = support["position"]

            col_Fx = [1,  0,  0,  0,  0,  0]
            col_Fy = [0,  1,  0,  0,  0,  d]
            col_Fz = [0,  0,  1,  0, -d,  0]
            col_Mx = [0,  0,  0,  1,  0,  0]
            col_My = [0,  0,  0,  0,  1,  0]
            col_Mz = [0,  0,  0,  0,  0,  1]

            statics_solver_matrix.append(col_Fx)
            statics_solver_matrix.append(col_Fy)
            statics_solver_matrix.append(col_Fz)
            statics_solver_matrix.append(col_Mx)
            statics_solver_matrix.append(col_My)
            statics_solver_matrix.append(col_Mz)
            unknowns.append({"dof": "Fx", "position": d})
            unknowns.append({"dof": "Fy", "position": d})
            unknowns.append({"dof": "Fz", "position": d})
            unknowns.append({"dof": "Mx", "position": d})
            unknowns.append({"dof": "My", "position": d})
            unknowns.append({"dof": "Mz", "position": d})
        else:
            raise NotImplementedError(f"Support type '{support['type']}' not implemented")

    A = np.array(statics_solver_matrix).T  # (6, n_unknowns)

    b = np.array([
        -sum(load["force"][0]  for load in all_loads),   # ΣFx
        -sum(load["force"][1]  for load in all_loads),   # ΣFy
        -sum(load["force"][2]  for load in all_loads),   # ΣFz
        -sum(load["moment"][0] for load in all_loads),   # ΣMx
        -sum(load["moment"][1] - load["force"][2] * load["position"] for load in all_loads),  # ΣMy
        -sum(load["moment"][2] + load["force"][1] * load["position"] for load in all_loads),  # ΣMz
    ])

    '''print("\n--- Solver Debug ---")
    print(f"\nA matrix (6 x {A.shape[1]}):")
    print(A)
    print(f"\nb vector:")
    labels = ["ΣFx", "ΣFy", "ΣFz", "ΣMx", "ΣMy", "ΣMz"]
    for label, val in zip(labels, b):
        print(f"  {label} = {val:.4f}")
    print(f"\nUnknowns: {[u['dof']+'@'+str(u['position']) for u in unknowns]}")

    reactions, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    print(f"\nSolved reactions: {reactions}")
    print(f"\nResidual A@x - b: {A @ reactions - b}")'''

    reactions, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    if rank < A.shape[1]:
        raise ValueError("The system is statically indeterminate or has insufficient supports to solve for reactions.")

    if not np.allclose(A @ reactions, b, atol=1e-8, rtol=1e-8):
        raise ValueError(
            "Support configuration is unstable or incompatible with the applied loads."
        )

    # Inject reactions back as point loads/moments

    # Map back
    reaction_loads = []
    for value, unknown in zip(reactions, unknowns):
        force  = [0, 0, 0]
        moment = [0, 0, 0]

        if unknown["dof"] == "Fy": force[1] = value
        if unknown["dof"] == "Fz": force[2] = value
        if unknown["dof"] == "Fx": force[0] = value
        if unknown["dof"] == "Mx": moment[0] = value
        if unknown["dof"] == "My": moment[1] = value
        if unknown["dof"] == "Mz": moment[2] = value

        reaction_loads.append({
            "type":     "reaction",
            "position": unknown["position"],
            "force":    tuple(force),
            "moment":   tuple(moment),
        })

    all_loads.extend(reaction_loads)

    return all_loads

def compute_internal_loads(x_arr, all_loads):
    results = {
        "V_y": np.zeros(len(x_arr)),
        "V_z": np.zeros(len(x_arr)),
        "M_y": np.zeros(len(x_arr)),
        "M_z": np.zeros(len(x_arr)),
        "T":   np.zeros(len(x_arr)),
        "N":   np.zeros(len(x_arr)),
    }

    for i, x in enumerate(x_arr):
        for load in all_loads:
            a  = load["position"]
            Fx, Fy, Fz = load["force"]
            Mx, My, Mz = load["moment"]

            # shear — point force uses ⟨x-a⟩⁰
            results["V_y"][i] += Fy * S(x, a, 0)
            results["V_z"][i] += Fz * S(x, a, 0)

            # moment — point force uses ⟨x-a⟩¹, point moment uses ⟨x-a⟩⁰
            results["M_z"][i] += Fy * S(x, a, 1)   # Fy bends about z
            results["M_y"][i] += Fz * S(x, a, 1)   # Fz bends about y

            # applied couples. Sign convention matches the statics solver
            # (sum Mz = Mz + Fy*x, sum My = My - Fz*x) with M_z' = V_y and M_y' = V_z,
            # so a z-couple enters with the opposite sign to a force's moment.
            results["M_z"][i] -= Mz * S(x, a, 0)
            results["M_y"][i] += My * S(x, a, 0)

            # torsion and axial
            results["T"][i]   += Mx * S(x, a, 0)
            results["N"][i]   += Fx * S(x, a, 0)

    return results

def compute_deflection(x_arr, results, geometry, material_properties, supports):
    E = material_properties["young_modulus"]

    # build per-point EI arrays (EI varies where diameter steps)
    EI_arr = np.array([section_at(x, geometry)["I"] * E for x in x_arr])

    theta_y_raw = np.zeros(len(x_arr))
    theta_z_raw = np.zeros(len(x_arr))
    delta_y_raw = np.zeros(len(x_arr))
    delta_z_raw = np.zeros(len(x_arr))

    for i in range(len(x_arr)):
        theta_y_raw[i] = np.trapezoid(results["M_z"][:i+1] / EI_arr[:i+1], x_arr[:i+1])
        theta_z_raw[i] = np.trapezoid(results["M_y"][:i+1] / EI_arr[:i+1], x_arr[:i+1])

    for i in range(len(x_arr)):
        delta_y_raw[i] = np.trapezoid(theta_y_raw[:i+1], x_arr[:i+1])
        delta_z_raw[i] = np.trapezoid(theta_z_raw[:i+1], x_arr[:i+1])

    # Boundary conditions come from the supports:
    #   theta(x) = theta_raw(x) + C1
    #   delta(x) = delta_raw(x) + C1*x + C2
    # every support pins delta = 0 at its position; a fixed support also pins theta = 0.
    rows, b_y, b_z = [], [], []
    for support in supports:
        xs = support["position"]
        rows.append([xs, 1])
        b_y.append(-np.interp(xs, x_arr, delta_y_raw))
        b_z.append(-np.interp(xs, x_arr, delta_z_raw))
        if support["type"] == "fixed":
            rows.append([1, 0])
            b_y.append(-np.interp(xs, x_arr, theta_y_raw))
            b_z.append(-np.interp(xs, x_arr, theta_z_raw))

    A = np.array(rows, dtype=float)
    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("Supports do not constrain the deflection curve (need two bearings or one fixed support).")

    (C1_y, C2_y), *_ = np.linalg.lstsq(A, np.array(b_y), rcond=None)
    (C1_z, C2_z), *_ = np.linalg.lstsq(A, np.array(b_z), rcond=None)

    theta_y = theta_y_raw + C1_y
    theta_z = theta_z_raw + C1_z

    delta_y = delta_y_raw + C1_y * x_arr + C2_y
    delta_z = delta_z_raw + C1_z * x_arr + C2_z

    return theta_y, theta_z, delta_y, delta_z

def compute_stress(x_arr, results, geometry):
    sigma = np.zeros(len(x_arr))
    tau   = np.zeros(len(x_arr))

    for i, x in enumerate(x_arr):
        seg = section_at(x, geometry)
        I, J, A, c = seg["I"], seg["J"], seg["A"], seg["c"]

        M_total = np.sqrt(results["M_y"][i]**2 + results["M_z"][i]**2)
        sigma[i] = (M_total * c / I) + (results["N"][i] / A)
        tau[i]   = (results["T"][i] * c / J)

    sigma_vm = np.sqrt(sigma**2 + 3 * tau**2)

    return sigma, tau, sigma_vm

def plot_beam_preview(case):
    """Draw a to-scale side view of the beam: segments, supports and loads."""
    length   = case["length"]
    geometry = case["geometry"]
    supports = case["supports"]
    loads    = case["loads"]

    d_max = max(seg["diameter"] for seg in geometry)
    gear_max = max((l["diameter"] for l in loads if l["type"] in ("driving gear", "driven gear")), default=0.0)
    y_extent = max(d_max, gear_max) / 2

    fig, ax = plt.subplots(figsize=(14, 6))

    # shaft segments
    for seg in geometry:
        d = seg["diameter"]
        ax.add_patch(plt.Rectangle((seg["start"], -d / 2), seg["end"] - seg["start"], d,
                                   facecolor="lightgray", edgecolor="black", linewidth=1.2, zorder=2))
        ax.text((seg["start"] + seg["end"]) / 2, 0, f"d = {d*1e3:.1f} mm",
                ha="center", va="center", fontsize=8, zorder=5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="-.", zorder=1)

    # supports
    tri_h = 0.35 * d_max        # triangle height in y units
    tri_w = 0.015 * length      # triangle half-width in x units (independent of aspect)
    for s in supports:
        x = s["position"]
        y0 = -section_at(x, geometry)["diameter"] / 2
        if s["type"] == "bearing":
            ax.add_patch(plt.Polygon([[x, y0], [x - tri_w, y0 - tri_h], [x + tri_w, y0 - tri_h]],
                                     closed=True, facecolor="white", edgecolor="black", linewidth=1.2, zorder=3))
            ax.text(x, y0 - tri_h * 1.15, f"bearing\nx = {x:.3f} m", ha="center", va="top", fontsize=8)
        else:
            wall_w = 0.02 * length
            ax.add_patch(plt.Rectangle((x - wall_w / 2, -y_extent), wall_w, 2 * y_extent,
                                       facecolor="none", edgecolor="black", hatch="////", linewidth=1.2, zorder=3))
            ax.text(x, -y_extent * 1.05, f"{s['type']}\nx = {x:.3f} m", ha="center", va="top", fontsize=8)

    # loads
    arrow_len = 0.6 * y_extent if y_extent > 0 else 0.1 * length
    for l in loads:
        x = l["position"]
        r_shaft = section_at(x, geometry)["diameter"] / 2
        if l["type"] in ("driving gear", "driven gear"):
            r = l["diameter"] / 2
            color = "steelblue" if l["type"] == "driving gear" else "darkorange"
            ax.add_patch(plt.Circle((x, 0), r, facecolor=color, alpha=0.15, edgecolor=color,
                                    linewidth=1.5, linestyle="--", zorder=1))
            label = f"{l['type']}\nD = {l['diameter']*1e3:.0f} mm, {l['tooth_angle']:g}°"
            if l["type"] == "driving gear":
                label += f"\n{l['power_input']:g} {l['power_unit']} @ {l['speed']:g} RPM"
            ax.text(x, r + 0.04 * y_extent, label, ha="center", va="bottom", fontsize=8, color=color, zorder=5)
        elif l["type"] == "point load":
            Fx, Fy, Fz = l["force"]
            F = np.hypot(Fy, Fz)
            # arrow points in the direction of Fy (down if Fy < 0), tail away from the shaft
            tail = 1 if Fy < 0 else -1
            if F > 0:
                ax.annotate("", xy=(x, tail * r_shaft), xytext=(x, tail * (r_shaft + arrow_len)),
                            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8), zorder=4)
            ax.text(x, tail * (r_shaft + arrow_len * 1.1), f"point load\nF = ({Fx:g}, {Fy:g}, {Fz:g}) N",
                    ha="center", va="bottom" if tail > 0 else "top", fontsize=8, color="red", zorder=5)
        elif l["type"] == "point moment":
            Mx, My, Mz = l["moment"]
            ax.plot(x, r_shaft, marker="o", markersize=9, markerfacecolor="none", markeredgecolor="purple",
                    markeredgewidth=1.8, zorder=4)
            ax.text(x, r_shaft + 0.15 * y_extent, f"point moment\nM = ({Mx:g}, {My:g}, {Mz:g}) N·m",
                    ha="center", va="bottom", fontsize=8, color="purple", zorder=5)

    # station marks along the x axis for every support and load
    stations = sorted({s["position"] for s in supports} | {l["position"] for l in loads})
    ax.set_xticks(stations)
    ax.set_xticklabels([f"{s:.3f}" for s in stations], rotation=45, fontsize=8)

    ax.set_xlim(-0.05 * length, 1.05 * length)
    ax.set_ylim(-2.0 * y_extent - tri_h, 2.2 * y_extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # slender beams are unreadable at true aspect, so exaggerate y for those
    if length / (2 * y_extent) <= 6:
        ax.set_aspect("equal")
        scale_note = "drawn to scale"
    else:
        ax.set_aspect("auto")
        scale_note = "x to scale, y exaggerated"
    ax.set_title(f"Beam preview - {Path(case['case_path']).name}  (L = {length:.3f} m, {scale_note})")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_diagrams(x_arr, results, critical_x):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    load_diagrams = [
        ("V_y", "Shear V_y (N)",        axes[0, 0]),
        ("V_z", "Shear V_z (N)",        axes[0, 1]),
        ("M_y", "Bending M_y (N*m)",    axes[1, 0]),
        ("M_z", "Bending M_z (N*m)",    axes[1, 1]),
        ("T",   "Torsion T (N*m)",      axes[2, 0]),
        ("N",   "Axial N (N)",          axes[2, 1]),
    ]

    for key, title, ax in load_diagrams:
        ax.plot(x_arr, results[key], color="steelblue", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(critical_x, color="red", linewidth=1, linestyle="--", label="critical section")
        ax.fill_between(x_arr, results[key], alpha=0.15, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    deflection_diagrams = [
        ("theta_y", "Slope theta_y (rad)",    axes[0, 0]),
        ("theta_z", "Slope theta_z (rad)",    axes[0, 1]),
        ("delta_y", "Deflection delta_y (m)", axes[1, 0]),
        ("delta_z", "Deflection delta_z (m)", axes[1, 1]),
    ]

    for key, title, ax in deflection_diagrams:
        ax.plot(x_arr, results[key], color="darkorange", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(critical_x, color="red", linewidth=1, linestyle="--", label="critical section")
        ax.fill_between(x_arr, results[key], alpha=0.15, color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_mohrs_circle(critical_idx, x_arr, results, geometry):
    x_c = x_arr[critical_idx]
    seg = section_at(x_c, geometry)
    I, J, A, c = seg["I"], seg["J"], seg["A"], seg["c"]

    # internal loads at critical section
    My  = results["M_y"][critical_idx]
    Mz  = results["M_z"][critical_idx]
    T   = results["T"][critical_idx]
    N   = results["N"][critical_idx]

    # stress components at worst point (outer fiber)
    M_total = np.sqrt(My**2 + Mz**2)
    sigma_x = (M_total * c / I) + (N / A)   # normal stress
    sigma_y = 0                               # no transverse normal stress
    tau_xy  = T * c / J                       # torsional shear

    # mohr's circle parameters
    center = (sigma_x + sigma_y) / 2
    radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)

    # principal stresses
    sigma_1 = center + radius
    sigma_2 = center - radius

    # principal angle
    theta_p = 0.5 * np.degrees(np.arctan2(tau_xy, (sigma_x - sigma_y) / 2))

    # max shear
    tau_max = radius

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # draw circle
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(center + radius * np.cos(theta),
            radius * np.sin(theta),
            color="steelblue", linewidth=1.5)

    # centre point
    ax.plot(center, 0, "ko", markersize=4)

    # current stress state point A (sigma_x, -tau_xy) and B (sigma_y, +tau_xy)
    ax.plot(sigma_x,  -tau_xy, "o", color="coral",    markersize=8, label=f"Point A  (sigma={sigma_x/1e6:.1f} MPa, tau={-tau_xy/1e6:.1f} MPa)")
    ax.plot(sigma_y,  +tau_xy, "o", color="steelblue", markersize=8, label=f"Point B  (sigma={sigma_y/1e6:.1f} MPa, tau={+tau_xy/1e6:.1f} MPa)")

    # diameter line A to B
    ax.plot([sigma_x, sigma_y], [-tau_xy, tau_xy],
            color="gray", linewidth=0.8, linestyle="--")

    # principal stress points on sigma axis
    ax.plot(sigma_1, 0, "^", color="red",   markersize=9, label=f"sigma_1 = {sigma_1/1e6:.2f} MPa")
    ax.plot(sigma_2, 0, "v", color="green", markersize=9, label=f"sigma_2 = {sigma_2/1e6:.2f} MPa")

    # max shear point
    ax.plot(center, tau_max, "s", color="purple", markersize=8, label=f"tau_max = {tau_max/1e6:.2f} MPa")

    # reference lines
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # annotations
    ax.annotate(f"C = {center/1e6:.2f} MPa", xy=(center, 0),
                xytext=(center, radius * 0.15),
                ha="center", fontsize=9, color="black")

    ax.set_xlabel("Normal stress sigma (Pa)")
    ax.set_ylabel("Shear stress tau (Pa)")
    ax.set_title(f"Mohr's circle - critical section x = {x_c:.4f} m  (d = {seg['diameter']*1e3:.1f} mm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()

    # print summary
    print(f"\n--- Principal stress summary (x = {x_c:.4f} m) ---")
    print(f"  sigma_x  = {sigma_x/1e6:.2f} MPa")
    print(f"  tau_xy   = {tau_xy/1e6:.2f} MPa")
    print(f"  Center C = {center/1e6:.2f} MPa")
    print(f"  Radius R = {radius/1e6:.2f} MPa")
    print(f"  sigma_1  = {sigma_1/1e6:.2f} MPa")
    print(f"  sigma_2  = {sigma_2/1e6:.2f} MPa")
    print(f"  tau_max  = {tau_max/1e6:.2f} MPa")
    print(f"  theta_p  = {theta_p:.2f} deg  (principal angle, 2D convention)")
    print(f" theta_z = {results['theta_z'][critical_idx]:.6f} rad  (slope about z at critical section)")
    print(f"  delta_z = {results['delta_z'][critical_idx]*1e3:.2f} mm  (deflection at critical section)")

def critical_section_heatmap(critical_idx, x_arr, results, geometry):
    x_c = x_arr[critical_idx]
    seg = section_at(x_c, geometry)
    I, J, A, c = seg["I"], seg["J"], seg["A"], seg["c"]

    # internal loads at critical section
    My = results["M_y"][critical_idx]
    Mz = results["M_z"][critical_idx]
    T  = results["T"][critical_idx]
    N  = results["N"][critical_idx]

    # build a 2D grid over the cross section
    n    = 300
    coords = np.linspace(-c, c, n)
    Y, Z  = np.meshgrid(coords, coords)

    # mask to circular section
    mask = (Y**2 + Z**2) <= c**2

    # normal stress at each point (y, z)
    sigma = np.where(mask,
        N/A + (Mz * Y / I) + (My * Z / I),
        np.nan
    )

    # signed torsional shear stress at each point
    r   = np.sqrt(Y**2 + Z**2)
    tau = np.where(mask, T * r / J, np.nan)

    # von Mises
    sigma_vm = np.where(mask,
        np.sqrt(sigma**2 + 3 * tau**2),
        np.nan
    )

    return Y, Z, sigma, tau, sigma_vm

def debug_reactions(all_loads, loads):
    print("\n--- Reaction Debug ---")

    # print applied loads
    print("\nApplied loads:")
    for load in loads:
        print(f"  x={load['position']:.3f}m  F={load['force']}  M={load['moment']}")

    # print solved reactions
    print("\nSolved reactions:")
    for load in all_loads:
        if load["type"] == "reaction":
            print(f"  x={load['position']:.3f}m  F={load['force']}  M={load['moment']}")

    # verify equilibrium manually
    print("\nEquilibrium check (should all be ~0):")
    print(f"  ΣFx = {sum(l['force'][0] for l in all_loads):.4f} N")
    print(f"  ΣFy = {sum(l['force'][1] for l in all_loads):.4f} N")
    print(f"  ΣFz = {sum(l['force'][2] for l in all_loads):.4f} N")
    print(f"  ΣMx = {sum(l['moment'][0] for l in all_loads):.4f} N·m")
    print(f"  ΣMy = {sum(l['moment'][1] + l['force'][2] * l['position'] for l in all_loads):.4f} N·m")
    print(f"  ΣMz = {sum(l['moment'][2] + l['force'][1] * l['position'] for l in all_loads):.4f} N·m")

def validate_global_equilibrium(all_loads, atol=1e-6):
    residuals = {
        "sum_fx": sum(l["force"][0] for l in all_loads),
        "sum_fy": sum(l["force"][1] for l in all_loads),
        "sum_fz": sum(l["force"][2] for l in all_loads),
        "sum_mx": sum(l["moment"][0] for l in all_loads),
        "sum_my": sum(l["moment"][1] - l["force"][2] * l["position"] for l in all_loads),
        "sum_mz": sum(l["moment"][2] + l["force"][1] * l["position"] for l in all_loads),
    }

    if not all(np.isclose(value, 0.0, atol=atol) for value in residuals.values()):
        raise ValueError(f"Global equilibrium check failed: {residuals}")

    return residuals

def required_diameter_msst(M, T, Sy, n):
    """
    M  = resultant bending moment at section (N·m)
    T  = torque at section (N·m)
    Sy = yield strength (Pa)
    n  = desired factor of safety
    """
    return ((32 * n / (np.pi * Sy)) * np.sqrt(M**2 + T**2)) ** (1/3)


if __name__ == "__main__":

    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASE
    if not case_path.is_file():
        sys.exit(f"Case file not found: {case_path}")
    case = load_case(case_path)
    print(f"Loaded case: {case_path}")
    for load in case["loads"]:
        if load["type"] == "driving gear":
            print(f"  driving gear at x={load['position']} m: "
                  f"{load['power_input']:g} {load['power_unit']} = {load['power']:.2f} W")

    length              = case["length"]
    simulated_points    = case["simulated_points"]
    supports            = case["supports"]
    material_properties = case["material_properties"]
    geometry            = case["geometry"]
    loads               = case["loads"]

    prepare_geometry(geometry)

    #preview drawing of the case
    plot_beam_preview(case)

    #build mesh
    x_arr = np.linspace(0, length, simulated_points)

    #solve reactions and build unified load list
    all_loads = build_load_list(loads, supports)
    validate_global_equilibrium(all_loads)
    print(reactions := [load for load in all_loads if load["type"] == "reaction"])

    # --- compute internal load diagrams ---
    results = compute_internal_loads(x_arr, all_loads)
    #debug_reactions(all_loads, loads)
    Mz_max_idx = np.argmax(np.abs(results["M_z"]))
    print(f"Maximum bending moment M_z = {results['M_z'][Mz_max_idx]:.2f} N·m at x = {x_arr[Mz_max_idx]:.4f} m")
    My_max_idx = np.argmax(np.abs(results["M_y"]))
    print(f"Maximum bending moment M_y = {results['M_y'][My_max_idx]:.2f} N·m at x = {x_arr[My_max_idx]:.4f} m")
    M_total_max_idx = np.argmax(np.sqrt(results["M_y"]**2 + results["M_z"]**2))
    print(f"Maximum resultant bending moment M_total = {np.sqrt(results['M_y'][M_total_max_idx]**2 + results['M_z'][M_total_max_idx]**2):.2f} N·m at x = {x_arr[M_total_max_idx]:.4f} m")
    T_max_idx = np.argmax(np.abs(results["T"]))
    print(f"Maximum internal torque T = {results['T'][T_max_idx]:.2f} N·m at x = {x_arr[T_max_idx]:.4f} m")

    theta_y, theta_z, delta_y, delta_z = compute_deflection(x_arr, results, geometry, material_properties, supports)
    results["theta_y"] = theta_y
    results["theta_z"] = theta_z
    results["delta_y"] = delta_y
    results["delta_z"] = delta_z

    #stress recovery along beam centerline
    sigma, tau, sigma_vm = compute_stress(x_arr, results, geometry)

    Sy = case["yield_strength"]           # yield strength (Pa)
    n  = case["target_factor_of_safety"]  # target factor of safety
    tau_msst = np.sqrt((sigma / 2)**2 + tau**2)
    fos_msst = np.divide(
        Sy,
        2 * tau_msst,
        out=np.full_like(tau_msst, np.inf),
        where=tau_msst > 0,
    )

    #find critical section
    critical_idx = np.argmin(fos_msst)
    critical_x   = x_arr[critical_idx]
    print(f"Critical section by current geometry MSST/Tresca FOS at x = {critical_x:.4f} m")
    print(f"  sigma_vm = {sigma_vm[critical_idx]/1e6:.2f} MPa")
    print(f"  sigma    = {sigma[critical_idx]/1e6:.2f} MPa")
    print(f"  tau      = {tau[critical_idx]/1e6:.2f} MPa")
    print(f"  FOS      = {fos_msst[critical_idx]:.2f}")
    print(f"  M_y      = {results['M_y'][critical_idx]:.2f} N·m")
    print(f"  M_z      = {results['M_z'][critical_idx]:.2f} N·m")
    M_total = np.sqrt(results["M_y"][critical_idx]**2 + results["M_z"][critical_idx]**2)
    print(f"  M_total  = {M_total:.2f} N·m")
    Torque = results["T"][critical_idx]
    print(f"  T        = {Torque:.2f} N·m")

    plot_diagrams(x_arr, results, critical_x)

    plot_mohrs_circle(critical_idx, x_arr, results, geometry)

    #heatmap at critical section
    Y, Z, sigma_cs, tau_cs, sigma_vm_cs = critical_section_heatmap(
        critical_idx, x_arr, results, geometry
    )

    #plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Critical section at x = {critical_x:.4f} m")

    tau_torsion_max = np.nanmax(np.abs(tau_cs))
    tau_msst_max = np.nanmax(np.sqrt((sigma_cs / 2)**2 + tau_cs**2))
    factor_of_safety = Sy / (2 * tau_msst_max)
    print(f"Max torsional shear at section: {tau_torsion_max/1e6:.2f} MPa")
    print(f"Max shear for yielding (MSST/Tresca): {tau_msst_max/1e6:.2f} MPa")
    print(f"Estimated factor of safety against yielding (MSST/Tresca): {factor_of_safety:.2f}")

    d_required = np.zeros(len(x_arr))
    for i, x in enumerate(x_arr):
        M = np.sqrt(results["M_y"][i]**2 + results["M_z"][i]**2)
        T = results["T"][i]
        d_required[i] = required_diameter_msst(M, T, Sy, n)

    d_critical = np.max(d_required)
    required_diameter_idx = np.argmax(d_required)
    required_diameter_x = x_arr[required_diameter_idx]
    print(f"Required diameter for FOS = {n} (MSST/Tresca): {d_critical*1e3:.2f} mm at x = {required_diameter_x:.4f} m")
    print(f"  Current diameter at required-diameter section = {section_at(required_diameter_x, geometry)['diameter']*1e3:.2f} mm")
    print(f"  Current FOS at required-diameter section = {fos_msst[required_diameter_idx]:.2f}")
    print(f"  T at required-diameter section = {results['T'][required_diameter_idx]:.2f} N·m")

    print(f"\nDiameter recommendations for FOS < {n}:")
    recommendation_found = False
    for seg in geometry:
        in_segment = (seg["start"] <= x_arr) & (x_arr <= seg["end"])
        below_target = in_segment & (fos_msst < n)

        if not np.any(below_target):
            continue

        recommendation_found = True
        segment_idxs = np.flatnonzero(below_target)
        worst_idx = segment_idxs[np.argmin(fos_msst[segment_idxs])]
        required_idx = segment_idxs[np.argmax(d_required[segment_idxs])]
        current_diameter = seg["diameter"]
        recommended_diameter = d_required[required_idx]
        increase = recommended_diameter - current_diameter

        print(
            f"  x={x_arr[segment_idxs[0]]:.4f}-{x_arr[segment_idxs[-1]]:.4f} m: "
            f"current d={current_diameter*1e3:.2f} mm, "
            f"recommend d>={recommended_diameter*1e3:.2f} mm "
            f"(increase {increase*1e3:.2f} mm), "
            f"worst FOS={fos_msst[worst_idx]:.2f} at x={x_arr[worst_idx]:.4f} m"
        )

    if not recommendation_found:
        print(f"  All sections meet FOS >= {n}.")

    for ax, data, title in zip(
        axes,
        [sigma_cs, tau_cs, sigma_vm_cs],
        ["Normal stress σ (Pa)", "Shear stress τ (Pa)", "Von Mises σ_vm (Pa)"]
    ):
        im = ax.contourf(Y, Z, data, levels=100, cmap="RdBu_r")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("z (m)")

    plt.tight_layout()
    plt.show()
