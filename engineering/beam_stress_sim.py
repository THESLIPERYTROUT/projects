import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Inputs: [units] system = "SI" (default) or "Imperial" in the case file.
# Solver uses SI internally; reports and plots follow the selected unit system.
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

# Multipliers from case input units to SI. Imperial density is mass density,
# in lbm/in^3, not weight density. Angles (deg), speed (RPM), Poisson's ratio
# and factor of safety have the same conventions in both systems.
INCH_TO_M = 0.0254
LBM_TO_KG = 0.45359237
LBF_TO_N = LBM_TO_KG * 9.80665
# Unit vectors from the modeled shaft toward the mating external spur gear.
# +x runs along the shaft. Viewed from the +x end toward the origin, with
# +y up, +z points left: top = +y, bottom = -y, left = +z, right = -z.
MESH_DIRECTIONS = {
    "top": (1.0, 0.0), "right": (0.0, -1.0),
    "bottom": (-1.0, 0.0), "left": (0.0, 1.0),
}
INPUT_UNITS = {
    "SI": {
        "length": 1.0, "force": 1.0, "moment": 1.0,
        "stress": 1.0, "density": 1.0, "power_unit": "W",
    },
    "Imperial": {
        "length": INCH_TO_M,
        "force": LBF_TO_N,
        "moment": LBF_TO_N * INCH_TO_M,
        "stress": LBF_TO_N / INCH_TO_M**2,
        "density": LBM_TO_KG / INCH_TO_M**3,
        "power_unit": "hp",
    },
}

def _require(table, key, section):
    if key not in table:
        raise KeyError(f"Config section [{section}] is missing required key '{key}'")
    return table[key]

def load_case(path):
    """Read a TOML case file, convert all dimensional inputs to SI, and validate."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    units_config = raw.get("units", {})
    if not isinstance(units_config, dict):
        raise ValueError('Use [units] with system = "SI" or "Imperial"')
    system_input = str(units_config.get("system", "SI")).strip().casefold()
    unit_system = {"si": "SI", "imperial": "Imperial"}.get(system_input)
    if unit_system is None:
        raise ValueError('[units] system must be "SI" or "Imperial"')
    units = INPUT_UNITS[unit_system]

    for section in ("beam", "material", "supports", "geometry", "loads"):
        if section not in raw:
            raise KeyError(f"Config file {path} is missing required section [{section}]")

    beam = raw["beam"]
    length = float(_require(beam, "length", "beam")) * units["length"]
    mesh_density_factor = float(_require(beam, "mesh_density_factor", "beam")) * units["length"]
    if length <= 0 or mesh_density_factor <= 0:
        raise ValueError("[beam] length and mesh_density_factor must be positive")

    material = raw["material"]
    material_properties = {
        "young_modulus": float(_require(material, "young_modulus", "material")) * units["stress"],
        "poisson_ratio": float(material.get("poisson_ratio", 0.3)),
        "density":       float(material["density"]) * units["density"] if "density" in material else 7850.0,
    }
    yield_strength = float(_require(material, "yield_strength", "material")) * units["stress"]

    design = raw.get("design", {})
    target_fos = design.get("target_factor_of_safety", 2)

    supports = [dict(s) for s in raw["supports"]]
    for s in supports:
        s["position"] = float(_require(s, "position", "supports")) * units["length"]
        _require(s, "type", "supports")

    geometry = sorted((dict(g) for g in raw["geometry"]), key=lambda g: g["start"])
    for g in geometry:
        for key in ("start", "end", "diameter"):
            g[key] = float(_require(g, key, "geometry")) * units["length"]
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
        load["position"] = float(_require(load, "position", "loads")) * units["length"]
        if load["type"] in ("driving gear", "driven gear"):
            load["diameter"] = float(_require(load, "diameter", "loads")) * units["length"]
            orientation = str(_require(load, "meshing orientation", "loads")).strip().casefold()
            if orientation not in MESH_DIRECTIONS:
                raise ValueError("[loads] meshing orientation must be top, right, bottom, or left")
            load["meshing orientation"] = orientation
            load["tooth_angle"] = float(_require(load, "tooth_angle", "loads"))
            if not np.isfinite(load["diameter"]) or load["diameter"] <= 0:
                raise ValueError("[loads] gear diameter must be positive and finite")
            if not 0 < load["tooth_angle"] < 90:
                raise ValueError("[loads] tooth_angle is the pressure angle and must be between 0 and 90 degrees")
        if load["type"] == "driving gear":
            unit = str(load.get("power_unit", units["power_unit"]))
            if unit not in POWER_UNITS:
                raise ValueError(
                    f"Load at x={load['position']}: unknown power_unit '{unit}' "
                    f"(choose from {', '.join(POWER_UNITS)})"
                )
            load["power_unit"] = unit
            load["power_input"] = float(_require(load, "power", "loads"))  # as written in the case file
            load["power"] = load["power_input"] * POWER_UNITS[unit]        # solver works in W
            load["speed"] = float(_require(load, "speed", "loads"))
            if not np.isfinite(load["power"]) or load["power"] < 0:
                raise ValueError("[loads] gear power must be nonnegative and finite")
            if not np.isfinite(load["speed"]) or load["speed"] <= 0:
                raise ValueError("[loads] gear speed must be positive and finite (RPM magnitude)")
        # TOML arrays come in as lists; the solver expects (Fx, Fy, Fz) / (Mx, My, Mz) tuples
        load["force"]  = tuple(float(v) * units["force"] for v in load.get("force",  (0, 0, 0)))
        load["moment"] = tuple(float(v) * units["moment"] for v in load.get("moment", (0, 0, 0)))
        if len(load["force"]) != 3 or len(load["moment"]) != 3:
            raise ValueError(f"Load at x={load['position']}: force/moment must have 3 components")
        loads.append(load)

    for item in supports + loads:
        if not 0 <= item["position"] <= length:
            raise ValueError(f"Position {item['position']} is outside the beam [0, {length}]")

    return {
        "case_path": path,
        "unit_system": unit_system,
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
            d = load["diameter"]  # meters, matching the rest of the solver

            Wt = (60 * P) / (d * np.pi * N)
            Wr = Wt * np.tan(np.radians(load["tooth_angle"]))
            ny, nz = MESH_DIRECTIONS[load["meshing orientation"]]
            # Preserve the solver's torque convention: driving gear adds +Mx,
            # driven gear removes it (-Mx). For an external mesh the radial
            # force is -Wr*n, and the +Mx tangential direction is x_hat cross n.
            sign = 1 if load["type"] == "driving gear" else -1
            load["force"] = (0, -Wr * ny - sign * Wt * nz,
                                -Wr * nz + sign * Wt * ny)
            # Transfer the contact force to the shaft centerline with its
            # equivalent couple: Mx = r_y*Fz - r_z*Fy = sign * radius * Wt.
            load["moment"] = (sign * (d / 2) * Wt, 0, 0)
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
    units = ReportUnits(case["unit_system"])
    scale = units.scales["length"]
    length = case["length"] / scale
    # Display copies keep all solver geometry and loads in SI.
    geometry = [{key: seg[key] / scale for key in ("start", "end", "diameter")}
                for seg in case["geometry"]]
    supports = [dict(support, position=support["position"] / scale) for support in case["supports"]]
    loads = []
    for source in case["loads"]:
        load = dict(source, position=source["position"] / scale)
        if "diameter" in load:
            load["diameter"] /= scale
        for quantity in ("force", "moment"):
            load[quantity] = tuple(value / units.scales[quantity] for value in source[quantity])
        loads.append(load)

    d_max = max(seg["diameter"] for seg in geometry)
    gear_max = max((l["diameter"] for l in loads if l["type"] in ("driving gear", "driven gear")), default=0.0)
    y_extent = max(d_max, gear_max) / 2

    fig, ax = plt.subplots(figsize=(14, 6))

    # shaft segments
    for seg in geometry:
        d = seg["diameter"]
        ax.add_patch(plt.Rectangle((seg["start"], -d / 2), seg["end"] - seg["start"], d,
                                   facecolor="lightgray", edgecolor="black", linewidth=1.2, zorder=2))
        ax.text((seg["start"] + seg["end"]) / 2, 0, f"d = {units.number(d * scale, 'diameter')} {units.labels['diameter']}",
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
            ax.text(x, y0 - tri_h * 1.15, f"bearing\nx = {format_number(x)} {units.labels['length']}", ha="center", va="top", fontsize=8)
        else:
            wall_w = 0.02 * length
            ax.add_patch(plt.Rectangle((x - wall_w / 2, -y_extent), wall_w, 2 * y_extent,
                                       facecolor="none", edgecolor="black", hatch="////", linewidth=1.2, zorder=3))
            ax.text(x, -y_extent * 1.05, f"{s['type']}\nx = {format_number(x)} {units.labels['length']}", ha="center", va="top", fontsize=8)

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
            label = f"{l['type']}\nD = {units.number(l['diameter'] * scale, 'diameter')} {units.labels['diameter']}, {l['tooth_angle']:g}°"
            if l["type"] == "driving gear":
                label += f"\n{units.number(l['power'], 'power')} {units.labels['power']} @ {l['speed']:g} RPM"
            ax.text(x, r + 0.04 * y_extent, label, ha="center", va="bottom", fontsize=8, color=color, zorder=5)
        elif l["type"] == "point load":
            Fx, Fy, Fz = l["force"]
            F = np.hypot(Fy, Fz)
            # arrow points in the direction of Fy (down if Fy < 0), tail away from the shaft
            tail = 1 if Fy < 0 else -1
            if F > 0:
                ax.annotate("", xy=(x, tail * r_shaft), xytext=(x, tail * (r_shaft + arrow_len)),
                            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8), zorder=4)
            ax.text(x, tail * (r_shaft + arrow_len * 1.1), f"point load\nF = ({format_number(Fx)}, {format_number(Fy)}, {format_number(Fz)}) {units.labels['force']}",
                    ha="center", va="bottom" if tail > 0 else "top", fontsize=8, color="red", zorder=5)
        elif l["type"] == "point moment":
            Mx, My, Mz = l["moment"]
            ax.plot(x, r_shaft, marker="o", markersize=9, markerfacecolor="none", markeredgecolor="purple",
                    markeredgewidth=1.8, zorder=4)
            ax.text(x, r_shaft + 0.15 * y_extent, f"point moment\nM = ({format_number(Mx)}, {format_number(My)}, {format_number(Mz)}) {units.labels['moment']}",
                    ha="center", va="bottom", fontsize=8, color="purple", zorder=5)

    # station marks along the x axis for every support and load
    stations = sorted({s["position"] for s in supports} | {l["position"] for l in loads})
    ax.set_xticks(stations)
    ax.set_xticklabels([format_number(s) for s in stations], rotation=45, fontsize=8)

    ax.set_xlim(-0.05 * length, 1.05 * length)
    ax.set_ylim(-2.0 * y_extent - tri_h, 2.2 * y_extent)
    ax.set_xlabel(f"x ({units.labels['length']})")
    ax.set_ylabel(f"y ({units.labels['length']})")
    # slender beams are unreadable at true aspect, so exaggerate y for those
    if length / (2 * y_extent) <= 6:
        ax.set_aspect("equal")
        scale_note = "drawn to scale"
    else:
        ax.set_aspect("auto")
        scale_note = "x to scale, y exaggerated"
    ax.set_title(f"Beam preview - {Path(case['case_path']).name}  (L = {format_number(length)} {units.labels['length']}, {scale_note})")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_diagrams(x_arr, results, critical_x, unit_system="SI"):
    units = ReportUnits(unit_system)
    x_plot = x_arr / units.scales["length"]
    critical_plot = critical_x / units.scales["length"]
    groups = [
        ((3, 2), (16, 10), "steelblue", [
            ("V_y", "Shear V_y", "force"), ("V_z", "Shear V_z", "force"),
            ("M_y", "Bending M_y", "moment"), ("M_z", "Bending M_z", "moment"),
            ("T", "Torsion T", "moment"), ("N", "Axial N", "force"),
        ]),
        ((2, 2), (14, 8), "darkorange", [
            ("theta_y", "Slope theta_y", "slope"), ("theta_z", "Slope theta_z", "slope"),
            ("delta_y", "Deflection delta_y", "deflection"),
            ("delta_z", "Deflection delta_z", "deflection"),
        ]),
    ]
    for shape, size, color, diagrams in groups:
        fig, axes = plt.subplots(*shape, figsize=size)
        for ax, (key, title, quantity) in zip(axes.flat, diagrams):
            values = results[key] / units.scales[quantity]
            ax.plot(x_plot, values, color=color, linewidth=1.5)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.axvline(critical_plot, color="red", linewidth=1, linestyle="--", label="critical section")
            ax.fill_between(x_plot, values, alpha=0.15, color=color)
            ax.set_title(f"{title} ({units.labels[quantity]})")
            ax.set_xlabel(f"x ({units.labels['length']})")
            ax.set_ylabel(units.labels[quantity])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def principal_stresses(sigma_x, tau_xy):
    """Plane stress properties, in Pa and degrees, for sigma_y = 0."""
    center = sigma_x / 2
    radius = np.hypot(center, tau_xy)
    return {
        "center": center,
        "radius": radius,
        "sigma_1": center + radius,
        "sigma_2": center - radius,
        "theta_p": 0.5 * np.degrees(np.arctan2(tau_xy, center)),
    }

def plot_mohrs_circle(critical_idx, x_arr, results, geometry, unit_system="SI"):
    units = ReportUnits(unit_system)
    stress_unit = units.labels["stress"]
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

    sigma_x /= units.scales["stress"]
    tau_xy /= units.scales["stress"]
    principal = principal_stresses(sigma_x, tau_xy)
    center, radius = principal["center"], principal["radius"]
    sigma_1, sigma_2 = principal["sigma_1"], principal["sigma_2"]

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
    ax.plot(sigma_x,  -tau_xy, "o", color="coral",    markersize=8, label=f"Point A  (sigma={format_number(sigma_x)} {stress_unit}, tau={format_number(-tau_xy)} {stress_unit})")
    ax.plot(sigma_y,  +tau_xy, "o", color="steelblue", markersize=8, label=f"Point B  (sigma={format_number(sigma_y)} {stress_unit}, tau={format_number(+tau_xy)} {stress_unit})")

    # diameter line A to B
    ax.plot([sigma_x, sigma_y], [-tau_xy, tau_xy],
            color="gray", linewidth=0.8, linestyle="--")

    # principal stress points on sigma axis
    ax.plot(sigma_1, 0, "^", color="red",   markersize=9, label=f"sigma_1 = {format_number(sigma_1)} {stress_unit}")
    ax.plot(sigma_2, 0, "v", color="green", markersize=9, label=f"sigma_2 = {format_number(sigma_2)} {stress_unit}")

    # max shear point
    ax.plot(center, tau_max, "s", color="purple", markersize=8, label=f"tau_max = {format_number(tau_max)} {stress_unit}")

    # reference lines
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    # annotations
    ax.annotate(f"C = {format_number(center)} {stress_unit}", xy=(center, 0),
                xytext=(center, radius * 0.15),
                ha="center", fontsize=9, color="black")

    ax.set_xlabel(f"Normal stress sigma ({stress_unit})")
    ax.set_ylabel(f"Shear stress tau ({stress_unit})")
    ax.set_title(f"Mohr's circle - critical section x = {units.number(x_c, 'length')} {units.labels['length']}  (d = {units.number(seg['diameter'], 'diameter')} {units.labels['diameter']})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()

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

def debug_reactions(all_loads, loads, unit_system="SI"):
    units = ReportUnits(unit_system)
    print_load_table("APPLIED LOADS", loads, units)
    print_load_table("REACTION COMPONENTS", [l for l in all_loads if l["type"] == "reaction"], units)
    residuals = validate_global_equilibrium(all_loads)
    print_table("EQUILIBRIUM RESIDUALS (expected near zero)", ("Quantity", "Value", "Unit"), [
        units.row(name, value, "force" if name.startswith("sum_f") else "moment")
        for name, value in residuals.items()
    ])

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

def format_number(value):
    """Compact terminal numbers without hiding small, nonzero results."""
    value = float(value)
    if np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1e7:
        return f"{value:.4e}"
    decimals = max(0, min(4, 5 - int(np.floor(np.log10(abs(value))))))
    formatted = f"{value:,.{decimals}f}"
    return formatted.rstrip("0").rstrip(".") if decimals else formatted

class ReportUnits:
    """Convert SI results only for display; never mutate solver data."""

    def __init__(self, system="SI"):
        units = INPUT_UNITS[system]
        imperial = system == "Imperial"
        self.scales = {
            "length": units["length"],
            "diameter": units["length"] if imperial else 0.001,
            "deflection": units["length"] if imperial else 0.001,
            "force": units["force"],
            "moment": units["moment"],
            "stress": units["stress"] if imperial else 1e6,
            "power": POWER_UNITS[units["power_unit"]],
            "angle": 1.0, "slope": 1.0, "ratio": 1.0,
        }
        self.labels = {
            "length": "in" if imperial else "m",
            "diameter": "in" if imperial else "mm",
            "deflection": "in" if imperial else "mm",
            "force": "lbf" if imperial else "N",
            "moment": "lbf*in" if imperial else "N*m",
            "stress": "psi" if imperial else "MPa",
            "power": units["power_unit"],
            "angle": "deg", "slope": "rad", "ratio": "-",
        }

    def number(self, value, quantity):
        return format_number(value / self.scales[quantity])

    def row(self, name, value, quantity):
        return (name, self.number(value, quantity), self.labels[quantity])

def print_table(title, headers, rows, left_columns=(0,)):
    """Print aligned ASCII tables that also work in redirected output."""
    rows = [tuple(str(cell) for cell in row) for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in rows)])
              for i, header in enumerate(headers)]

    def line(cells):
        return "  " + "  ".join(
            cell.ljust(width) if i in left_columns else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        )

    print(f"\n{title}")
    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rows:
        print(line(row))

def print_load_table(title, loads, units):
    headers = ("Type", f"x [{units.labels['length']}]", *(
        f"{axis} [{units.labels[quantity]}]"
        for quantity, axes in (("force", ("Fx", "Fy", "Fz")),
                               ("moment", ("Mx", "My", "Mz")))
        for axis in axes
    ))
    rows = [
        (load["type"], units.number(load["position"], "length"),
         *(units.number(value, quantity)
           for quantity in ("force", "moment") for value in load[quantity]))
        for load in loads
    ]
    print_table(title, headers, rows)

def print_report(case, x_arr, all_loads, results, stresses, fos_msst,
                 critical_idx, section_stresses):
    """Print all results together, in the case's selected unit system."""
    units = ReportUnits(case["unit_system"])
    row = units.row
    headers = ("Quantity", "Value", "Unit")
    geometry = case["geometry"]
    sigma, tau, sigma_vm = stresses
    sigma_cs, tau_cs, _ = section_stresses
    idx = critical_idx
    critical_x = x_arr[idx]
    target = case["target_factor_of_safety"]
    bending = np.hypot(results["M_y"], results["M_z"])

    print("\n" + "=" * 78)
    print("BEAM STRESS ANALYSIS")
    print("=" * 78)
    print(f"  Case   : {case['case_path']}")
    print(f"  Units  : {case['unit_system']} (terminal report)")
    print(f"  Plots  : {case['unit_system']}, as labeled")
    print("  Status : Global equilibrium verified")
    print_table("CASE SUMMARY", headers, [
        row("Beam length", case["length"], "length"),
        row("Mesh spacing setting", case["mesh_density_factor"], "length"),
        ("Mesh points", f"{len(x_arr):,}", "-"),
        row("Young's modulus", case["material_properties"]["young_modulus"], "stress"),
        row("Yield strength", case["yield_strength"], "stress"),
        row("Target factor of safety", target, "ratio"),
    ])
    gears = [load for load in case["loads"] if load["type"] == "driving gear"]
    if gears:
        print_table("DRIVING GEARS", (
            f"x [{units.labels['length']}]", f"Power [{units.labels['power']}]", "Speed [RPM]"
        ), [(units.number(g["position"], "length"), units.number(g["power"], "power"),
             format_number(g["speed"])) for g in gears], left_columns=())

    print_load_table("APPLIED LOADS", case["loads"], units)
    # Reactions are stored by degree of freedom; combine into one row per support.
    reactions = {}
    for load in all_loads:
        if load["type"] != "reaction":
            continue
        reaction = reactions.setdefault(load["position"], {
            "type": "reaction", "position": load["position"],
            "force": np.zeros(3), "moment": np.zeros(3),
        })
        reaction["force"] += load["force"]
        reaction["moment"] += load["moment"]
    print_load_table("SUPPORT REACTIONS", list(reactions.values()), units)

    peaks = []
    for name, values in (("Bending M_y", results["M_y"]), ("Bending M_z", results["M_z"]),
                         ("Resultant bending", bending), ("Torque T", results["T"])):
        peak = np.argmax(np.abs(values))
        peaks.append((*row(name, values[peak], "moment"), units.number(x_arr[peak], "length")))
    print_table("PEAK INTERNAL MOMENTS (signed values at maximum magnitude)",
                (*headers, f"x [{units.labels['length']}]"), peaks)

    print_table("CRITICAL SECTION (minimum MSST/Tresca factor of safety)", headers, [
        row("Position x", critical_x, "length"),
        row("Current diameter", section_at(critical_x, geometry)["diameter"], "diameter"),
        row("Von Mises stress", sigma_vm[idx], "stress"),
        row("Normal stress sigma_x", sigma[idx], "stress"),
        row("Torsional shear tau_xy", tau[idx], "stress"),
        row("Factor of safety", fos_msst[idx], "ratio"),
        row("Bending M_y", results["M_y"][idx], "moment"),
        row("Bending M_z", results["M_z"][idx], "moment"),
        row("Resultant bending", bending[idx], "moment"),
        row("Torque T", results["T"][idx], "moment"),
        row("Deflection y", results["delta_y"][idx], "deflection"),
        row("Deflection z", results["delta_z"][idx], "deflection"),
        row("Slope about y", results["theta_y"][idx], "slope"),
        row("Slope about z", results["theta_z"][idx], "slope"),
    ])
    principal = principal_stresses(sigma[idx], tau[idx])
    print_table("PRINCIPAL STRESSES (critical section, plane stress)", headers, [
        row("Mohr circle center", principal["center"], "stress"),
        row("Mohr circle radius / max shear", principal["radius"], "stress"),
        row("Principal stress sigma_1", principal["sigma_1"], "stress"),
        row("Principal stress sigma_2", principal["sigma_2"], "stress"),
        row("Principal angle (2D)", principal["theta_p"], "angle"),
    ])
    tau_torsion_max = np.nanmax(np.abs(tau_cs))
    tau_msst_max = np.nanmax(np.hypot(sigma_cs / 2, tau_cs))
    section_fos = case["yield_strength"] / (2 * tau_msst_max) if tau_msst_max > 0 else np.inf
    print_table("CROSS-SECTION GRID CHECK (critical section)", headers, [
        row("Maximum torsional shear", tau_torsion_max, "stress"),
        row("Maximum MSST/Tresca shear", tau_msst_max, "stress"),
        row("Estimated factor of safety", section_fos, "ratio"),
    ])

    d_required = required_diameter_msst(bending, results["T"], case["yield_strength"], target)
    required_idx = np.argmax(d_required)
    required_x = x_arr[required_idx]
    print_table("DIAMETER SIZING (MSST/Tresca)", headers, [
        row("Target factor of safety", target, "ratio"),
        row("Governing position x", required_x, "length"),
        row("Required diameter", d_required[required_idx], "diameter"),
        row("Current diameter", section_at(required_x, geometry)["diameter"], "diameter"),
        row("Current factor of safety", fos_msst[required_idx], "ratio"),
        row("Torque at sizing section", results["T"][required_idx], "moment"),
    ])
    recommendations = []
    for seg in geometry:
        below_target = (seg["start"] <= x_arr) & (x_arr <= seg["end"]) & (fos_msst < target)
        if not np.any(below_target):
            continue
        indices = np.flatnonzero(below_target)
        worst = indices[np.argmin(fos_msst[indices])]
        recommended = np.max(d_required[indices])
        recommendations.append((
            units.number(x_arr[indices[0]], "length"), units.number(x_arr[indices[-1]], "length"),
            units.number(seg["diameter"], "diameter"), units.number(recommended, "diameter"),
            units.number(recommended - seg["diameter"], "diameter"),
            format_number(fos_msst[worst]), units.number(x_arr[worst], "length"),
        ))
    if recommendations:
        length_unit, diameter_unit = units.labels["length"], units.labels["diameter"]
        print_table("DIAMETER RECOMMENDATIONS (sampled ranges below target FOS)", (
            f"From [{length_unit}]", f"To [{length_unit}]", f"Current [{diameter_unit}]",
            f"Required [{diameter_unit}]", f"Change [{diameter_unit}]", "Min FOS", f"At x [{length_unit}]"
        ), recommendations, left_columns=())
    else:
        print(f"\n  All sections meet the target factor of safety ({format_number(target)}).")
    print("=" * 78 + "\n")

def run_case(case, show_plots=True):
    """Solve in SI, print the complete report, then optionally display plots."""
    geometry = case["geometry"]
    prepare_geometry(geometry)
    x_arr = np.linspace(0, case["length"], case["simulated_points"])
    all_loads = build_load_list(case["loads"], case["supports"])
    validate_global_equilibrium(all_loads)
    results = compute_internal_loads(x_arr, all_loads)
    deflection = compute_deflection(
        x_arr, results, geometry, case["material_properties"], case["supports"]
    )
    results.update(zip(("theta_y", "theta_z", "delta_y", "delta_z"), deflection))
    stresses = compute_stress(x_arr, results, geometry)
    sigma, tau, _ = stresses
    tau_msst = np.hypot(sigma / 2, tau)
    fos_msst = np.divide(
        case["yield_strength"], 2 * tau_msst,
        out=np.full_like(tau_msst, np.inf), where=tau_msst > 0,
    )
    critical_idx = np.argmin(fos_msst)
    critical_x = x_arr[critical_idx]
    Y, Z, *section_stresses = critical_section_heatmap(critical_idx, x_arr, results, geometry)
    print_report(case, x_arr, all_loads, results, stresses, fos_msst,
                 critical_idx, section_stresses)

    if not show_plots:
        return
    plot_beam_preview(case)
    plot_diagrams(x_arr, results, critical_x, case["unit_system"])
    plot_mohrs_circle(critical_idx, x_arr, results, geometry, case["unit_system"])
    plot_section_heatmap(Y, Z, section_stresses, critical_x, case["unit_system"])


def plot_section_heatmap(Y, Z, section_stresses, critical_x, unit_system="SI"):
    units = ReportUnits(unit_system)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Critical section at x = {units.number(critical_x, 'length')} {units.labels['length']}")
    for ax, data, title in zip(
        axes, section_stresses,
        ["Normal stress sigma", "Shear stress tau", "Von Mises sigma_vm"]
    ):
        im = ax.contourf(Y / units.scales["diameter"], Z / units.scales["diameter"],
                         data / units.scales["stress"], levels=100, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label=units.labels["stress"])
        ax.set_title(f"{title} ({units.labels['stress']})")
        ax.set_aspect("equal")
        ax.set_xlabel(f"y ({units.labels['diameter']})")
        ax.set_ylabel(f"z ({units.labels['diameter']})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASE
    if not case_path.is_file():
        sys.exit(f"Case file not found: {case_path}")
    run_case(load_case(case_path))
