import sys
import tomllib
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

# Inputs: [units] system = "SI" (default) or "Imperial" in the case file.
# Solver uses SI internally; reports and plots follow the selected unit system.
#
# All case settings (beam, material, supports, geometry, loads) live in a TOML
# case file so different sim cases can be saved independently of this program.
# See beam_cases/gear_shaft_stepped_shaft.toml for a documented example.
#
#   python beam_stress_sim.py                       -> runs DEFAULT_CASE
#   python beam_stress_sim.py path/to/case.toml     -> runs that case

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE = SCRIPT_DIR / "beam_cases" / "gear_shaft_stepped_shaft.toml"

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

FATIGUE_CRITERIA = {
    "goodman": "Modified Goodman", "gerber": "Gerber",
    "asme-elliptic": "ASME elliptic", "soderberg": "Soderberg",
}
# Marin surface factor ka = a * Sut^b with Sut in MPa (Shigley Table 6-2).
SURFACE_FINISH = {
    "ground":     (1.58, -0.085),
    "machined":   (4.51, -0.265),
    "cold-drawn": (4.51, -0.265),
    "hot-rolled": (57.7, -0.718),
    "as-forged":  (272.0, -0.995),
}
# Shoulder-fillet stress concentration of a stepped round shaft,
# Kt = A * (r/d)^b, tabulated by D/d. Power-law fits to Peterson's charts
# (Norton, Machine Design, App. C; cf. Shigley Figs A-15-8 / A-15-9).
KT_BENDING = (  # (D/d, A, b), ascending D/d
    (1.01, 0.91938, -0.17032), (1.02, 0.96048, -0.17711), (1.03, 0.98061, -0.18381),
    (1.05, 0.98137, -0.19653), (1.07, 0.97527, -0.20958), (1.10, 0.95120, -0.23757),
    (1.20, 0.97098, -0.21796), (1.50, 0.93836, -0.25759), (2.00, 0.90879, -0.28598),
    (3.00, 0.89334, -0.30860), (6.00, 0.87868, -0.33243),
)
KT_TORSION = (
    (1.09, 0.90337, -0.12692), (1.20, 0.83425, -0.21649),
    (1.33, 0.84897, -0.23161), (2.00, 0.86331, -0.23865),
)
# Fillet radius assumed for a shoulder with no fillet_radius / kf given:
# a sharp fillet, r/d = 0.02 (Shigley Table 7-1 first-iteration estimate).
DEFAULT_FILLET_RATIO = 0.02
# r/d range covered by the Peterson charts behind the Kt fits above.
KT_FIT_RD_RANGE = (0.01, 0.3)
# Fatigue diagram: an operating point up to this many times beyond the
# envelope axes widens the axes; farther out it is drawn as an edge arrow.
OFF_SCALE_LIMIT = 3.0
CRITERION_COLORS = {
    "goodman": "steelblue", "gerber": "seagreen",
    "asme-elliptic": "mediumvioletred", "soderberg": "saddlebrown",
}
FATIGUE_KEYS = {"criterion", "surface_finish", "reliability", "temperature",
                "endurance_limit", "torque_alternating_fraction"}

def _require(table, key, section):
    if key not in table:
        raise KeyError(f"Config section [{section}] is missing required key '{key}'")
    return table[key]

def load_fatigue_settings(table, unit_system, ultimate_strength):
    """Validate the optional [fatigue] section; stresses to SI, temperature to deg F."""
    if not isinstance(table, dict):
        raise ValueError("Use a [fatigue] table for fatigue settings")
    unknown = set(table) - FATIGUE_KEYS
    if unknown:
        raise KeyError(f"[fatigue] has unknown key(s): {', '.join(sorted(unknown))} "
                       f"(expected {', '.join(sorted(FATIGUE_KEYS))})")
    if ultimate_strength is None:
        raise KeyError("[fatigue] requires [material] ultimate_strength")

    criterion = str(table.get("criterion", "goodman")).strip().casefold()
    if criterion not in FATIGUE_CRITERIA:
        raise ValueError(f"[fatigue] criterion must be one of {', '.join(FATIGUE_CRITERIA)}")
    finish = str(table.get("surface_finish", "machined")).strip().casefold()
    if finish not in SURFACE_FINISH:
        raise ValueError(f"[fatigue] surface_finish must be one of {', '.join(SURFACE_FINISH)}")
    reliability = float(table.get("reliability", 0.5))
    if not 0.5 <= reliability < 1:
        raise ValueError("[fatigue] reliability must be in [0.5, 1)")

    temperature = table.get("temperature")
    temperature_f = None
    if temperature is not None:
        temperature = float(temperature)
        temperature_f = temperature if unit_system == "Imperial" else temperature * 9 / 5 + 32
        if not temperature_f <= 1000:
            raise ValueError("[fatigue] temperature must be at most 1000 F / 538 C "
                             "(limit of the Marin temperature factor fit)")

    endurance_limit = None
    if "endurance_limit" in table:
        endurance_limit = float(table["endurance_limit"]) * INPUT_UNITS[unit_system]["stress"]
        if not endurance_limit > 0:
            raise ValueError("[fatigue] endurance_limit must be positive")

    torque_fraction = float(table.get("torque_alternating_fraction", 0.0))
    if not torque_fraction >= 0:
        raise ValueError("[fatigue] torque_alternating_fraction must be nonnegative")

    return {
        "criterion": criterion,
        "surface_finish": finish,
        "reliability": reliability,
        "temperature": temperature,      # as written, deg F (Imperial) or deg C (SI)
        "temperature_f": temperature_f,
        "endurance_limit": endurance_limit,
        "torque_alternating_fraction": torque_fraction,
    }

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
    ultimate_strength = None
    if "ultimate_strength" in material:
        ultimate_strength = float(material["ultimate_strength"]) * units["stress"]
        if not ultimate_strength >= yield_strength:
            raise ValueError("[material] ultimate_strength must be at least yield_strength")

    design = raw.get("design", {})
    target_fos = design.get("target_factor_of_safety", 2)

    fatigue = None
    if "fatigue" in raw:
        fatigue = load_fatigue_settings(raw["fatigue"], unit_system, ultimate_strength)

    supports = [dict(s) for s in raw["supports"]]
    for s in supports:
        s["position"] = float(_require(s, "position", "supports")) * units["length"]
        _require(s, "type", "supports")

    geometry = sorted((dict(g) for g in raw["geometry"]), key=lambda g: g["start"])
    for g in geometry:
        for key in ("start", "end", "diameter"):
            g[key] = float(_require(g, key, "geometry")) * units["length"]
        # Optional shoulder details at this segment's start (fatigue only).
        if "fillet_radius" in g:
            g["fillet_radius"] = float(g["fillet_radius"]) * units["length"]
            if not g["fillet_radius"] > 0:
                raise ValueError("[[geometry]] fillet_radius must be positive")
        for key in ("kf", "kfs"):
            if key in g:
                g[key] = float(g[key])
                if not g[key] >= 1:
                    raise ValueError(f"[[geometry]] {key} must be at least 1")
    if not np.isclose(geometry[0]["start"], 0.0):
        raise ValueError("First [[geometry]] segment must start at 0")
    if not np.isclose(geometry[-1]["end"], length):
        raise ValueError(f"Last [[geometry]] segment must end at the beam length ({length})")
    for a, b in zip(geometry, geometry[1:]):
        if not np.isclose(a["end"], b["start"]):
            raise ValueError(f"[[geometry]] segments are not contiguous at x={a['end']} / {b['start']}")
    if any(key in geometry[0] for key in ("fillet_radius", "kf", "kfs")):
        raise ValueError("[[geometry]] fillet_radius / kf / kfs describe the shoulder at a segment's "
                         "start, so they cannot be set on the first segment")

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
        "ultimate_strength": ultimate_strength,
        "target_factor_of_safety": target_fos,
        "fatigue": fatigue,
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

def section_indices(x_arr, geometry):
    """
    Index of the geometry segment containing each position in x_arr.

    At a shoulder (x exactly on a segment boundary) the smaller-diameter
    segment is chosen, since it carries the higher stress.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    starts = np.array([seg["start"] for seg in geometry])
    ends = np.array([seg["end"] for seg in geometry])
    diameters = np.array([seg["diameter"] for seg in geometry])
    outside = (x_arr < starts[0]) | (x_arr > ends[-1])
    if np.any(outside):
        x = x_arr[outside][0]
        raise ValueError(f"x={x:.4f} is outside the beam geometry [{starts[0]}, {ends[-1]}]")
    idx = np.searchsorted(ends, x_arr, side="left")
    on_boundary = (idx + 1 < len(geometry)) & (x_arr == ends[np.minimum(idx, len(ends) - 1)])
    next_idx = np.minimum(idx + 1, len(geometry) - 1)
    return np.where(on_boundary & (diameters[next_idx] < diameters[idx]), next_idx, idx)

def section_at(x, geometry):
    """Return the geometry segment containing position x (smaller side at a shoulder)."""
    return geometry[section_indices([x], geometry)[0]]

def section_arrays(x_arr, geometry):
    """Per-station cross-section properties (I, J, A, c, diameter) as arrays."""
    idx = section_indices(x_arr, geometry)
    return {key: np.array([seg[key] for seg in geometry])[idx]
            for key in ("I", "J", "A", "c", "diameter")}

def is_stress_raiser(prev_seg, seg):
    """A segment boundary is a stress raiser if the diameter steps or kf/kfs/fillet is given."""
    return (not np.isclose(prev_seg["diameter"], seg["diameter"])
            or any(key in seg for key in ("fillet_radius", "kf", "kfs")))

def stress_raiser_positions(geometry):
    """x positions of shoulders and other marked stress raisers between segments."""
    return [b["start"] for a, b in zip(geometry, geometry[1:]) if is_stress_raiser(a, b)]

def build_stations(length, n_points, geometry):
    """Evenly spaced stations plus every stress raiser, so steps are always evaluated."""
    return np.union1d(np.linspace(0, length, n_points), stress_raiser_positions(geometry))

def S(x, a, n):
    """
    Singularity function <x - a>^n, evaluated elementwise over x.

    n < 0  : not integrated (Dirac / doublet) — returns 1 if x == a else 0
    n >= 0 : Macaulay bracket — returns (x-a)^n if x >= a, else 0
             (right-continuous, so a load at x = a is counted at x = a)
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        value = np.isclose(x, a).astype(float)
    else:
        value = np.where(x >= a, np.maximum(x - a, 0.0)**n, 0.0)
    return float(value) if value.ndim == 0 else value

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

    for load in all_loads:
        a  = load["position"]
        Fx, Fy, Fz = load["force"]
        Mx, My, Mz = load["moment"]
        step = S(x_arr, a, 0)
        ramp = S(x_arr, a, 1)

        # shear — point force uses ⟨x-a⟩⁰
        results["V_y"] += Fy * step
        results["V_z"] += Fz * step

        # moment — point force uses ⟨x-a⟩¹, point moment uses ⟨x-a⟩⁰
        results["M_z"] += Fy * ramp   # Fy bends about z
        results["M_y"] += Fz * ramp   # Fz bends about y

        # applied couples. Sign convention matches the statics solver
        # (sum Mz = Mz + Fy*x, sum My = My - Fz*x) with M_z' = V_y and M_y' = V_z,
        # so a z-couple enters with the opposite sign to a force's moment.
        results["M_z"] -= Mz * step
        results["M_y"] += My * step

        # torsion and axial
        results["T"]   += Mx * step
        results["N"]   += Fx * step

    return results

def cumulative_trapezoid(y, x):
    """Running trapezoid integral of y over x, starting at 0 (same length as x)."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))

def compute_deflection(x_arr, results, geometry, material_properties, supports):
    E = material_properties["young_modulus"]

    # build per-point EI arrays (EI varies where diameter steps)
    EI_arr = section_arrays(x_arr, geometry)["I"] * E

    theta_y_raw = cumulative_trapezoid(results["M_z"] / EI_arr, x_arr)
    theta_z_raw = cumulative_trapezoid(results["M_y"] / EI_arr, x_arr)
    delta_y_raw = cumulative_trapezoid(theta_y_raw, x_arr)
    delta_z_raw = cumulative_trapezoid(theta_z_raw, x_arr)

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
    sec = section_arrays(x_arr, geometry)
    M_total = np.hypot(results["M_y"], results["M_z"])
    sigma = (M_total * sec["c"] / sec["I"]) + (results["N"] / sec["A"])
    tau   = results["T"] * sec["c"] / sec["J"]

    sigma_vm = np.sqrt(sigma**2 + 3 * tau**2)

    return sigma, tau, sigma_vm

# ---------------------------------------------------------------------------
# Fatigue (infinite life, rotating shaft) — Shigley ch. 6-7 approach:
#   bending is fully reversed by rotation, torque and axial load are mean,
#   with an optional alternating torque fraction. Kf/Kfs act at stress raisers.
# ---------------------------------------------------------------------------

def shoulder_kt(D, d, r, table):
    """Geometric stress concentration Kt for a shoulder fillet (stepped round shaft)."""
    ratios, A, b = (np.array(column) for column in zip(*table))
    ratio = D / d
    # Inside the table interpolate the fit coefficients; above it use the
    # largest D/d row; below it blend linearly to Kt = 1 at D/d = 1.
    kt = np.interp(ratio, ratios, A) * (r / d) ** np.interp(ratio, ratios, b)
    if ratio < ratios[0]:
        kt = 1 + (kt - 1) * max(ratio - 1, 0.0) / (ratios[0] - 1)
    return max(float(kt), 1.0)

def notch_sensitivity(r, ultimate_strength, torsion=False):
    """Neuber notch sensitivity q (Shigley eqs 6-34, 6-35a/b), r in m, Sut in Pa."""
    # The Neuber constant fit is valid for 50-250 kpsi. Clamping Sut low keeps
    # sqrt(a) from turning negative; clamping it high raises q (conservative).
    sut = np.clip(ultimate_strength / (1e3 * INPUT_UNITS["Imperial"]["stress"]), 50, 250)
    first = 0.190 - 2.51e-3 * sut if torsion else 0.246 - 3.08e-3 * sut
    sqrt_a = first + (1.35e-5 if torsion else 1.51e-5) * sut**2 - 2.67e-8 * sut**3  # sqrt(in)
    return 1 / (1 + sqrt_a / np.sqrt(r / INCH_TO_M))

def stress_raisers(geometry, ultimate_strength):
    """Kt, q and Kf (bending) / Kts, qs and Kfs (torsion) at each stress raiser."""
    raisers = []
    for prev, seg in zip(geometry, geometry[1:]):
        if not is_stress_raiser(prev, seg):
            continue
        d, D = sorted((prev["diameter"], seg["diameter"]))
        r = seg.get("fillet_radius", DEFAULT_FILLET_RATIO * d)
        kt = shoulder_kt(D, d, r, KT_BENDING)
        kts = shoulder_kt(D, d, r, KT_TORSION)
        q = notch_sensitivity(r, ultimate_strength)
        qs = notch_sensitivity(r, ultimate_strength, torsion=True)
        warnings = []
        # Outside the charted r/d range the power-law fit is extrapolated; for
        # large r/d it drops toward Kt = 1 and the shoulder silently stops
        # counting, which is how a units slip (in vs mm) would show up.
        low, high = KT_FIT_RD_RANGE
        if not ("kf" in seg and "kfs" in seg) and not low <= r / d <= high:
            warnings.append(f"r/d = {format_number(r / d)} is outside the Kt fit range "
                            f"{low:g}-{high:g}; check fillet_radius units or set kf/kfs")
        raisers.append({
            "position": seg["start"], "d": d, "D": D, "r": r,
            "r_assumed": "fillet_radius" not in seg,
            "Kt": kt, "Kts": kts, "q": q, "qs": qs,
            "Kf": seg.get("kf", 1 + q * (kt - 1)),
            "Kfs": seg.get("kfs", 1 + qs * (kts - 1)),
            "override": "kf" in seg or "kfs" in seg,
            "warnings": warnings,
        })
    return raisers

def marin_factors(d, ultimate_strength, fatigue):
    """Marin factors and the endurance limit Se (Pa) for diameter(s) d (m)."""
    sut_mpa = ultimate_strength / 1e6
    a, b = SURFACE_FINISH[fatigue["surface_finish"]]
    ka = a * sut_mpa**b
    # Size factor for rotating round sections; outside 2.79-254 mm the fit is
    # held at its end values.
    d_mm = np.clip(np.asarray(d, dtype=float) * 1e3, 2.79, 254)
    kb = np.where(d_mm <= 51, 1.24 * d_mm**-0.107, 1.51 * d_mm**-0.157)
    kc = 1.0  # bending; torsion enters through von Mises, not kc
    T = fatigue["temperature_f"]
    if T is None or T <= 70:
        kd = 1.0
    else:
        kd = 0.975 + 0.432e-3 * T - 0.115e-5 * T**2 + 0.104e-8 * T**3 - 0.595e-12 * T**4
    z_a = NormalDist().inv_cdf(fatigue["reliability"])
    ke = 1 - 0.08 * z_a
    se_prime = fatigue["endurance_limit"]
    if se_prime is None:
        se_prime = 0.5 * ultimate_strength if sut_mpa <= 1400 else 700e6
    return {"ka": ka, "kb": kb, "kc": kc, "kd": kd, "ke": ke,
            "Se_prime": se_prime, "Se": ka * kb * kc * kd * ke * se_prime}

def fatigue_stress_state(d, M, T, N, Kf, Kfs, torque_fraction):
    """Alternating/mean stress components (Pa) for a rotating round shaft."""
    sigma_a = Kf * 32 * M / (np.pi * d**3)        # fully reversed bending
    sigma_m = Kf * 4 * np.abs(N) / (np.pi * d**2)  # steady axial (sign ignored: conservative)
    tau_m = Kfs * 16 * np.abs(T) / (np.pi * d**3)
    tau_a = torque_fraction * tau_m
    return {
        "sigma_a": sigma_a, "sigma_m": sigma_m, "tau_a": tau_a, "tau_m": tau_m,
        "vm_a": np.sqrt(sigma_a**2 + 3 * tau_a**2),
        "vm_m": np.sqrt(sigma_m**2 + 3 * tau_m**2),
        "vm_max": np.sqrt((sigma_a + sigma_m)**2 + 3 * (tau_a + tau_m)**2),
    }

def fatigue_fos(vm_a, vm_m, Se, Sut, Sy, criterion):
    """Infinite-life factor of safety from von Mises alternating/mean stress."""
    vm_a, vm_m = np.broadcast_arrays(np.asarray(vm_a, float), np.asarray(vm_m, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        if criterion == "gerber":
            # n = (Sut^2 sa / 2 sm^2 Se)(-1 + sqrt(1 + (2 sm Se / Sut sa)^2)), written
            # in a form that stays finite for sa -> 0 (n -> Sut/sm) and sm -> 0 (n -> Se/sa).
            denominator = vm_a + np.sqrt(vm_a**2 + (2 * vm_m * Se / Sut)**2)
            return np.where(denominator > 0, 2 * Se / denominator, np.inf)
        if criterion == "goodman":
            inverse = vm_a / Se + vm_m / Sut
        elif criterion == "soderberg":
            inverse = vm_a / Se + vm_m / Sy
        elif criterion == "asme-elliptic":
            inverse = np.hypot(vm_a / Se, vm_m / Sy)
        else:
            raise ValueError(f"Unknown fatigue criterion '{criterion}'")
        return np.where(inverse > 0, 1 / inverse, np.inf)

def _fatigue_and_yield_fos(d, M, T, N, Kf, Kfs, case, criterion=None):
    fatigue = case["fatigue"]
    state = fatigue_stress_state(d, M, T, N, Kf, Kfs, fatigue["torque_alternating_fraction"])
    marin = marin_factors(d, case["ultimate_strength"], fatigue)
    n_f = fatigue_fos(state["vm_a"], state["vm_m"], marin["Se"],
                      case["ultimate_strength"], case["yield_strength"],
                      criterion or fatigue["criterion"])
    with np.errstate(divide="ignore"):
        n_y = np.where(state["vm_max"] > 0, case["yield_strength"] / state["vm_max"], np.inf)
    return state, marin, n_f, n_y

def required_diameter_fatigue(M, T, N, Kf, Kfs, case, target, criterion=None, iterations=60):
    """
    Smallest diameter at each station meeting the target FOS for both fatigue
    and first-cycle (Langer) yield. Kf/Kfs are held fixed; the size factor kb
    is re-evaluated at every trial diameter, so this is solved by bisection.
    """
    def governing(d):
        _, _, n_f, n_y = _fatigue_and_yield_fos(d, M, T, N, Kf, Kfs, case, criterion)
        return np.minimum(n_f, n_y)

    lo = np.full(np.shape(M), 1e-6)
    hi = np.full(np.shape(M), 1e-3)
    for _ in range(40):  # grow the bracket until every station passes
        short = governing(hi) < target
        if not np.any(short):
            break
        hi = np.where(short, hi * 2, hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        passes = governing(mid) >= target
        hi = np.where(passes, mid, hi)
        lo = np.where(passes, lo, mid)
    unloaded = (M == 0) & (T == 0) & (N == 0)
    return np.where(unloaded, 0.0, hi)

def compute_fatigue(x_arr, results, geometry, case):
    """
    Fatigue and Langer-yield factors of safety along the shaft.

    Every criterion in FATIGUE_CRITERIA is evaluated (fos_by_criterion,
    d_required_by_criterion); the case's selected criterion drives
    fos_fatigue, the critical section and d_required.
    """
    sec = section_arrays(x_arr, geometry)
    d = sec["diameter"]
    raisers = stress_raisers(geometry, case["ultimate_strength"])
    Kf = np.ones(len(x_arr))
    Kfs = np.ones(len(x_arr))
    raiser_idx = []
    for raiser in raisers:
        hits = np.flatnonzero(x_arr == raiser["position"])
        if len(hits) == 0:
            raise ValueError(f"Station list is missing the stress raiser at x={raiser['position']}; "
                             "build x_arr with build_stations()")
        Kf[hits] = raiser["Kf"]
        Kfs[hits] = raiser["Kfs"]
        raiser_idx.extend(hits)

    M = np.hypot(results["M_y"], results["M_z"])
    T, N = results["T"], results["N"]
    state, marin, _, n_y = _fatigue_and_yield_fos(d, M, T, N, Kf, Kfs, case)
    target = case["target_factor_of_safety"]
    selected = case["fatigue"]["criterion"]
    fos_by_criterion = {
        criterion: fatigue_fos(state["vm_a"], state["vm_m"], marin["Se"],
                               case["ultimate_strength"], case["yield_strength"], criterion)
        for criterion in FATIGUE_CRITERIA
    }
    d_required_by_criterion = {
        criterion: required_diameter_fatigue(M, T, N, Kf, Kfs, case, target, criterion)
        for criterion in FATIGUE_CRITERIA
    }
    n_f = fos_by_criterion[selected]
    return {
        **state, **marin,
        "Kf": Kf, "Kfs": Kfs, "diameter": d,
        "M": M, "T": T, "N": N,
        "fos_fatigue": n_f, "fos_yield": n_y,
        "fos_by_criterion": fos_by_criterion,
        "d_required_by_criterion": d_required_by_criterion,
        "critical_idx": int(np.argmin(n_f)),
        "yield_critical_idx": int(np.argmin(n_y)),
        "raisers": raisers, "raiser_idx": sorted(raiser_idx),
        "d_required": d_required_by_criterion[selected],
    }

def plot_beam_preview(case):
    """
    Side view of the beam: shaft segments, supports and loads.

    x is to scale. The y layout is built from the largest shaft radius h, so
    the shaft stays readable; gears are schematic symbols (not to scale),
    sized relative to each other, with the mesh side marked.
    """
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

    h = max(seg["diameter"] for seg in geometry) / 2   # layout unit: largest shaft radius
    gears = [l for l in loads if l["type"] in ("driving gear", "driven gear")]
    gear_max = max((g["diameter"] for g in gears), default=0.0)
    text_box = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)

    fig, ax = plt.subplots(figsize=(14, 6))

    # shaft segments (true relative diameters)
    for seg in geometry:
        d = seg["diameter"]
        ax.add_patch(plt.Rectangle((seg["start"], -d / 2), seg["end"] - seg["start"], d,
                                   facecolor="lightgray", edgecolor="black", linewidth=1.2, zorder=3))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="-.", zorder=4)

    # diameter labels in a band below the supports, staggered when segments are
    # narrow, each with a leader line to its segment
    label_rows = (-4.6 * h, -5.4 * h)
    crowded = any(seg["end"] - seg["start"] < 0.08 * length for seg in geometry)
    for i, seg in enumerate(geometry):
        x_mid = (seg["start"] + seg["end"]) / 2
        y_label = label_rows[i % 2] if crowded else label_rows[0]
        ax.annotate(f"Ø{units.number(seg['diameter'] * scale, 'diameter')}", xy=(x_mid, -seg["diameter"] / 2),
                    xytext=(x_mid, y_label), ha="center", va="top", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.5), zorder=2)
    ax.text(-0.04 * length, label_rows[0], f"shaft Ø\n[{units.labels['diameter']}]",
            ha="right", va="top", fontsize=7.5, color="dimgray")

    # supports
    tri_h = 0.8 * h
    tri_w = 0.012 * length      # half-width in x units (independent of aspect)
    for s in supports:
        x = s["position"]
        y0 = -section_at(x, geometry)["diameter"] / 2
        label = f"{s['type']}\nx = {format_number(x)} {units.labels['length']}"
        if s["type"] == "bearing":
            ax.add_patch(plt.Polygon([[x, y0], [x - tri_w, y0 - tri_h], [x + tri_w, y0 - tri_h]],
                                     closed=True, facecolor="white", edgecolor="black", linewidth=1.2, zorder=5))
            ax.text(x, y0 - tri_h - 0.2 * h, label, ha=edge_align(x, length), va="top", fontsize=8, bbox=text_box, zorder=6)
        else:
            wall_w = 0.02 * length
            ax.add_patch(plt.Rectangle((x - wall_w / 2, -2.2 * h), wall_w, 4.4 * h,
                                       facecolor="none", edgecolor="black", hatch="////", linewidth=1.2, zorder=5))
            ax.text(x, -2.3 * h, label, ha=edge_align(x, length), va="top", fontsize=8, bbox=text_box, zorder=6)

    # loads
    arrow_len = 1.5 * h
    for l in loads:
        x = l["position"]
        r_shaft = section_at(x, geometry)["diameter"] / 2
        if l["type"] in ("driving gear", "driven gear"):
            draw_gear_symbol(ax, l, h, gear_max, length, units, text_box)
        elif l["type"] == "point load":
            Fx, Fy, Fz = l["force"]
            F = np.hypot(Fy, Fz)
            # arrow points in the direction of Fy (down if Fy < 0), tail away from the shaft
            tail = 1 if Fy < 0 else -1
            if F > 0:
                ax.annotate("", xy=(x, tail * r_shaft), xytext=(x, tail * (r_shaft + arrow_len)),
                            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8), zorder=6)
            ax.text(x, tail * (r_shaft + arrow_len * 1.1), f"point load\nF = ({format_number(Fx)}, {format_number(Fy)}, {format_number(Fz)}) {units.labels['force']}",
                    ha=edge_align(x, length), va="bottom" if tail > 0 else "top", fontsize=8, color="red", bbox=text_box, zorder=7)
        elif l["type"] == "point moment":
            Mx, My, Mz = l["moment"]
            ax.plot(x, r_shaft, marker="o", markersize=9, markerfacecolor="none", markeredgecolor="purple",
                    markeredgewidth=1.8, zorder=6)
            ax.text(x, r_shaft + 0.6 * h, f"point moment\nM = ({format_number(Mx)}, {format_number(My)}, {format_number(Mz)}) {units.labels['moment']}",
                    ha=edge_align(x, length), va="bottom", fontsize=8, color="purple", bbox=text_box, zorder=7)

    # station marks along the x axis for every support and load
    stations = sorted({s["position"] for s in supports} | {l["position"] for l in loads})
    ax.set_xticks(stations)
    ax.set_xticklabels([format_number(s) for s in stations], rotation=45, fontsize=8)

    ax.set_xlim(-0.08 * length, 1.05 * length)
    ax.set_ylim(-6.4 * h, 7.0 * h)
    ax.set_xlabel(f"x ({units.labels['length']})")
    ax.set_ylabel(f"y ({units.labels['length']}, shaft only)")
    # slender beams are unreadable at true aspect, so exaggerate y for those
    if length / (2 * h) <= 6:
        ax.set_aspect("equal")
        scale_note = "shaft to scale"
    else:
        ax.set_aspect("auto")
        scale_note = "x to scale, y exaggerated"
    if gears:
        scale_note += "; gears schematic, not to scale"
    ax.set_title(f"Beam preview - {Path(case['case_path']).name}  (L = {format_number(length)} {units.labels['length']}, {scale_note})")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

def edge_align(x, length):
    """Text alignment that keeps a label near either beam end inside the axes."""
    if x > 0.9 * length:
        return "right"
    if x < 0.1 * length:
        return "left"
    return "center"

def draw_gear_symbol(ax, gear, h, gear_max, length, units, text_box):
    """
    Schematic side view of a spur gear: a blank with tooth bands top and bottom.
    Height grows with pitch diameter (relative to the largest gear) but stays
    within a few shaft radii, so it never dwarfs the shaft.
    """
    x = gear["position"]
    half = h * (1.8 + 1.4 * gear["diameter"] / gear_max)
    width = 0.022 * length
    tooth = 0.22 * half
    color = "steelblue" if gear["type"] == "driving gear" else "darkorange"
    ax.add_patch(plt.Rectangle((x - width / 2, -half), width, 2 * half, facecolor=color, alpha=0.25,
                               edgecolor=color, linewidth=1.5, zorder=1))
    for y in (half - tooth, -half):
        ax.add_patch(plt.Rectangle((x - width / 2, y), width, tooth, facecolor="white",
                                   edgecolor=color, hatch="||||", linewidth=1.0, zorder=2))
    ax.plot([x, x], [-half, half], color=color, linewidth=0.8, linestyle="-.", zorder=2)

    # Mesh side. +x right, +y up, so +z points out of the page:
    # top/bottom (+-y) get an arrow toward the mate; left (+z) is out of the
    # page (dot), right (-z) is into the page (cross).
    orientation = gear["meshing orientation"]
    mesh_note = {"top": "+y", "bottom": "-y", "left": "+z, out of page", "right": "-z, into page"}[orientation]
    if orientation in ("top", "bottom"):
        sign = 1 if orientation == "top" else -1
        ax.annotate("", xy=(x, sign * (half + 0.9 * h)), xytext=(x, sign * (half + 0.1 * h)),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=6)
    else:
        marker_x = x + width / 2 + 0.012 * length
        ax.plot(marker_x, 0.55 * half, marker="o", markersize=11, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5, zorder=6)
        ax.plot(marker_x, 0.55 * half, marker="." if orientation == "left" else "x",
                markersize=8 if orientation == "left" else 6, color=color, markeredgewidth=1.5, zorder=7)

    label = (f"{gear['type']}\nD = {units.number(gear['diameter'] * units.scales['length'], 'diameter')} "
             f"{units.labels['diameter']}, {gear['tooth_angle']:g}°\nmesh: {orientation} ({mesh_note})")
    if gear["type"] == "driving gear":
        label += f"\n{units.number(gear['power'], 'power')} {units.labels['power']} @ {gear['speed']:g} RPM"
    top = half + (1.0 * h if orientation == "top" else 0.15 * h)
    ax.text(x, top, label, ha=edge_align(x, length), va="bottom", fontsize=8, color=color, bbox=text_box, zorder=7)

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
                 critical_idx, section_stresses, fatigue=None):
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
    print_recommendations("DIAMETER RECOMMENDATIONS (sampled ranges below target FOS)",
                          x_arr, geometry, fos_msst, d_required, target, units)
    if fatigue is not None:
        print_fatigue_report(case, x_arr, fatigue, units)
    print("=" * 78 + "\n")

def print_recommendations(title, x_arr, geometry, fos, d_required, target, units):
    """Per-segment diameter changes where the sampled FOS is below the target."""
    sec_idx = section_indices(x_arr, geometry)
    recommendations = []
    for i, seg in enumerate(geometry):
        below_target = (sec_idx == i) & (fos < target)
        if not np.any(below_target):
            continue
        indices = np.flatnonzero(below_target)
        worst = indices[np.argmin(fos[indices])]
        recommended = np.max(d_required[indices])
        recommendations.append((
            units.number(x_arr[indices[0]], "length"), units.number(x_arr[indices[-1]], "length"),
            units.number(seg["diameter"], "diameter"), units.number(recommended, "diameter"),
            units.number(recommended - seg["diameter"], "diameter"),
            format_number(fos[worst]), units.number(x_arr[worst], "length"),
        ))
    if recommendations:
        length_unit, diameter_unit = units.labels["length"], units.labels["diameter"]
        print_table(title, (
            f"From [{length_unit}]", f"To [{length_unit}]", f"Current [{diameter_unit}]",
            f"Required [{diameter_unit}]", f"Change [{diameter_unit}]", "Min FOS", f"At x [{length_unit}]"
        ), recommendations, left_columns=())
    else:
        print(f"\n  All sections meet the target factor of safety ({format_number(target)}).")

def print_fatigue_report(case, x_arr, fatigue, units):
    """Fatigue block of the report (infinite life, rotating shaft)."""
    settings = case["fatigue"]
    row = units.row
    headers = ("Quantity", "Value", "Unit")
    target = case["target_factor_of_safety"]
    idx = fatigue["critical_idx"]
    y_idx = fatigue["yield_critical_idx"]
    temperature = settings["temperature"]
    temperature_unit = "F" if case["unit_system"] == "Imperial" else "C"

    print("\n" + "-" * 78)
    print(f"FATIGUE ANALYSIS - {FATIGUE_CRITERIA[settings['criterion']]}, infinite life, rotating shaft")
    print("  Bending is fully reversed; torque and axial load are steady"
          + (f" (alternating torque = {format_number(settings['torque_alternating_fraction'])} x steady)."
             if settings["torque_alternating_fraction"] else "."))
    print("-" * 78)
    print_table("ENDURANCE LIMIT (at fatigue critical section)", headers, [
        row("Ultimate strength Sut", case["ultimate_strength"], "stress"),
        row("Unmodified endurance limit Se'", fatigue["Se_prime"], "stress"),
        (f"Surface factor ka ({settings['surface_finish']})", format_number(fatigue["ka"]), "-"),
        row("Size factor kb", fatigue["kb"][idx], "ratio"),
        row("Load factor kc", fatigue["kc"], "ratio"),
        ("Temperature factor kd", format_number(fatigue["kd"]),
         "-" if temperature is None else f"- (at {format_number(temperature)} {temperature_unit})"),
        (f"Reliability factor ke ({format_number(settings['reliability'])})", format_number(fatigue["ke"]), "-"),
        row("Endurance limit Se", fatigue["Se"][idx], "stress"),
    ])

    if fatigue["raisers"]:
        diameter_unit = units.labels["diameter"]
        print_table("STRESS RAISERS (Kf, Kfs applied at these stations)", (
            f"x [{units.labels['length']}]", f"d [{diameter_unit}]", f"D [{diameter_unit}]",
            f"r [{diameter_unit}]", "Kt", "Kts", "q", "qs", "Kf", "Kfs", "Note",
        ), [(
            units.number(r["position"], "length"), units.number(r["d"], "diameter"),
            units.number(r["D"], "diameter"), units.number(r["r"], "diameter"),
            format_number(r["Kt"]), format_number(r["Kts"]), format_number(r["q"]),
            format_number(r["qs"]), format_number(r["Kf"]), format_number(r["Kfs"]),
            "kf/kfs override" if r["override"]
            else f"assumed sharp, r = {DEFAULT_FILLET_RATIO:g} d" if r["r_assumed"]
            else "WARNING: r/d out of range" if r["warnings"] else "",
        ) for r in fatigue["raisers"]], left_columns=(10,))
        if any(r["r_assumed"] and not r["override"] for r in fatigue["raisers"]):
            print("  Note: set fillet_radius (or kf/kfs) on [[geometry]] segments to replace assumed sharp fillets.")
        for raiser in fatigue["raisers"]:
            for warning in raiser["warnings"]:
                print(f"  WARNING at x = {units.number(raiser['position'], 'length')} "
                      f"{units.labels['length']}: {warning}")

    print_table("FATIGUE CRITICAL SECTION (minimum fatigue factor of safety)", headers, [
        row("Fatigue position x", x_arr[idx], "length"),
        row("Fatigue section diameter", fatigue["diameter"][idx], "diameter"),
        row("Fatigue Kf (bending)", fatigue["Kf"][idx], "ratio"),
        row("Fatigue Kfs (torsion)", fatigue["Kfs"][idx], "ratio"),
        row("Alternating bending sigma_a", fatigue["sigma_a"][idx], "stress"),
        row("Mean axial sigma_m", fatigue["sigma_m"][idx], "stress"),
        row("Alternating torsion tau_a", fatigue["tau_a"][idx], "stress"),
        row("Mean torsion tau_m", fatigue["tau_m"][idx], "stress"),
        row("Von Mises alternating sigma_a'", fatigue["vm_a"][idx], "stress"),
        row("Von Mises mean sigma_m'", fatigue["vm_m"][idx], "stress"),
        row("Fatigue factor of safety", fatigue["fos_fatigue"][idx], "ratio"),
        row("Langer yield FOS at this section", fatigue["fos_yield"][idx], "ratio"),
        row("Minimum Langer yield FOS", fatigue["fos_yield"][y_idx], "ratio"),
        row("Minimum Langer yield FOS at x", x_arr[y_idx], "length"),
    ])

    length_unit, diameter_unit = units.labels["length"], units.labels["diameter"]
    comparison = []
    for criterion, name in FATIGUE_CRITERIA.items():
        fos = fatigue["fos_by_criterion"][criterion]
        d_req = fatigue["d_required_by_criterion"][criterion]
        worst, sizing = int(np.argmin(fos)), int(np.argmax(d_req))
        comparison.append((
            name + (" (selected)" if criterion == settings["criterion"] else ""),
            format_number(fos[idx]), format_number(fos[worst]),
            units.number(x_arr[worst], "length"),
            units.number(d_req[sizing], "diameter"), units.number(x_arr[sizing], "length"),
        ))
    print_table("FATIGUE CRITERIA COMPARISON (Langer yield included in required diameter)", (
        "Criterion", "FOS at critical x", "Min FOS", f"Min at x [{length_unit}]",
        f"Required d [{diameter_unit}]", f"Governing x [{length_unit}]",
    ), comparison)
    print("  Gerber and ASME-elliptic fit test data closely; Goodman is linear and conservative;\n"
          "  Soderberg (yield-based) is the most conservative.")

    governing = np.minimum(fatigue["fos_fatigue"], fatigue["fos_yield"])
    d_required = fatigue["d_required"]
    required_idx = int(np.argmax(d_required))
    print_table("FATIGUE DIAMETER SIZING (fatigue and Langer yield, Kf held fixed)", headers, [
        row("Fatigue target factor of safety", target, "ratio"),
        row("Fatigue governing position x", x_arr[required_idx], "length"),
        row("Fatigue required diameter", d_required[required_idx], "diameter"),
        row("Fatigue current diameter", fatigue["diameter"][required_idx], "diameter"),
    ])
    print_recommendations("FATIGUE DIAMETER RECOMMENDATIONS (sampled ranges below target FOS)",
                          x_arr, case["geometry"], governing, d_required, target, units)

def run_case(case, show_plots=True):
    """Solve in SI, print the complete report, then optionally display plots."""
    geometry = case["geometry"]
    prepare_geometry(geometry)
    x_arr = build_stations(case["length"], case["simulated_points"], geometry)
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
    fatigue = None
    if case.get("fatigue") is not None:
        fatigue = compute_fatigue(x_arr, results, geometry, case)
    print_report(case, x_arr, all_loads, results, stresses, fos_msst,
                 critical_idx, section_stresses, fatigue)

    if not show_plots:
        return
    plot_beam_preview(case)
    plot_diagrams(x_arr, results, critical_x, case["unit_system"])
    plot_mohrs_circle(critical_idx, x_arr, results, geometry, case["unit_system"])
    plot_section_heatmap(Y, Z, section_stresses, critical_x, case["unit_system"])
    if fatigue is not None:
        plot_fatigue_fos(x_arr, fos_msst, fatigue, case)
        plot_fatigue_diagram(x_arr, fatigue, case)

def plot_fatigue_fos(x_arr, fos_static, fatigue, case):
    """Factors of safety along the shaft: static MSST, fatigue and Langer yield."""
    units = ReportUnits(case["unit_system"])
    x_plot = x_arr / units.scales["length"]
    target = case["target_factor_of_safety"]
    # Unloaded stations have infinite FOS; cap the axis so the interesting range shows.
    finite = np.concatenate([f[np.isfinite(f)] for f in
                             (fos_static, fatigue["fos_fatigue"], fatigue["fos_yield"])])
    cap = max(4 * target, 1.2 * np.min(finite)) if finite.size else 4 * target

    fig, ax = plt.subplots(figsize=(14, 6))
    for values, label, color in ((fos_static, "static yield (MSST, nominal)", "gray"),
                                 (fatigue["fos_yield"], "first-cycle yield (Langer, with Kf)", "darkorange"),
                                 (fatigue["fos_fatigue"], f"fatigue ({FATIGUE_CRITERIA[case['fatigue']['criterion']]})", "steelblue")):
        ax.plot(x_plot, np.minimum(values, cap), label=label, color=color, linewidth=1.5)
    # Other criteria for comparison, thin and dashed.
    for criterion, values in fatigue["fos_by_criterion"].items():
        if criterion != case["fatigue"]["criterion"]:
            ax.plot(x_plot, np.minimum(values, cap), label=f"fatigue ({FATIGUE_CRITERIA[criterion]})",
                    color=CRITERION_COLORS[criterion], linewidth=0.9, linestyle="--")
    ax.axhline(target, color="red", linestyle="--", linewidth=1, label=f"target FOS = {format_number(target)}")
    for i, raiser in enumerate(fatigue["raisers"]):
        ax.axvline(raiser["position"] / units.scales["length"], color="purple", linestyle=":",
                   linewidth=1, label="stress raiser" if i == 0 else None)
    idx = fatigue["critical_idx"]
    ax.plot(x_plot[idx], min(fatigue["fos_fatigue"][idx], cap), "o", color="steelblue")
    ax.set_ylim(0, cap)
    ax.set_xlabel(f"x ({units.labels['length']})")
    ax.set_ylabel(f"factor of safety (capped at {format_number(cap)})")
    ax.set_title(f"Factor of safety along the shaft - {Path(case['case_path']).name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def criterion_envelope(sm, Se, Sut, Sy, criterion):
    """Alternating strength on the failure line at mean stress sm (NaN past its end)."""
    with np.errstate(invalid="ignore"):
        return {
            "goodman": Se * (1 - sm / Sut),
            "gerber": Se * (1 - (sm / Sut)**2),
            "asme-elliptic": Se * np.sqrt(1 - (sm / Sy)**2),
            "soderberg": Se * (1 - sm / Sy),
        }[criterion]

def plot_fatigue_diagram(x_arr, fatigue, case):
    """Mean-alternating (Haigh) diagram at the fatigue critical section, all criteria."""
    units = ReportUnits(case["unit_system"])
    scale = units.scales["stress"]
    settings = case["fatigue"]
    idx = fatigue["critical_idx"]
    Se, Sut, Sy = fatigue["Se"][idx], case["ultimate_strength"], case["yield_strength"]
    sm = np.linspace(0, Sut, 400)
    selected = settings["criterion"]
    envelopes = {criterion: criterion_envelope(sm, Se, Sut, Sy, criterion)
                 for criterion in ("goodman", "gerber", "asme-elliptic", "soderberg")}

    fig, ax = plt.subplots(figsize=(9, 8))
    for criterion, envelope in envelopes.items():
        keep = envelope >= 0
        n_c = fatigue["fos_by_criterion"][criterion][idx]
        is_selected = criterion == selected
        ax.plot(sm[keep] / scale, envelope[keep] / scale, color=CRITERION_COLORS[criterion],
                linewidth=2.2 if is_selected else 1.0, linestyle="-" if is_selected else "--",
                label=f"{FATIGUE_CRITERIA[criterion]}{' (selected)' if is_selected else ''}, "
                      f"n = {format_number(n_c)}")
    ax.plot([0, Sy / scale], [Sy / scale, 0], color="darkorange", linestyle="--", linewidth=1.2,
            label="Langer first-cycle yield line")
    vm_a, vm_m = fatigue["vm_a"][idx], fatigue["vm_m"][idx]
    n = fatigue["fos_fatigue"][idx]
    x_max = 1.05 * max(Sut, Sy) / scale
    y_max = 1.05 * max(Se, Sy) / scale
    px, py = vm_m / scale, vm_a / scale
    # A point a little outside the envelopes widens the axes; one far outside
    # (n well below 1) would shrink the envelopes to nothing, so it is drawn
    # as an arrow on the plot edge, pointing along the load line instead.
    overshoot = max(px / x_max, py / y_max)
    if 1 < overshoot <= OFF_SCALE_LIMIT:
        x_max, y_max = x_max * overshoot * 1.05, y_max * overshoot * 1.05
        overshoot = 1.0
    point_label = (f"operating point (n = {format_number(n)}): sigma_m' = {format_number(px)}, "
                   f"sigma_a' = {format_number(py)} {units.labels['stress']}")
    if overshoot <= 1:
        ax.plot(px, py, "o", color="red", markersize=8, label=point_label)
        line_end = (n * px, n * py) if np.isfinite(n) else None
    else:
        edge = 0.97 / overshoot   # fraction of the way to the point that lands just inside the axes
        ex, ey = edge * px, edge * py
        ax.annotate("", xy=(ex, ey), xytext=(0.8 * ex, 0.8 * ey),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
        ax.plot([], [], marker=r"$\rightarrow$", color="red", linestyle="none", markersize=10,
                label=f"{point_label}\n  [off scale, {format_number(overshoot)}x beyond axes]")
        ax.text(ex - 0.06 * x_max, ey, "operating point\noff scale", color="red", fontsize=8,
                ha="right", va="top")
        line_end = (ex, ey)
    if line_end is not None:
        ax.plot([0, line_end[0]], [0, line_end[1]], color="red", linewidth=0.8,
                linestyle=":", label="load line")

    # Design point: this section's stresses at the diameter that just meets the
    # target FOS (selected criterion and Langer yield, Kf held fixed). Se changes
    # with diameter through kb, so the design line uses Se at that diameter.
    d_req = fatigue["d_required"][idx]
    if d_req > 0:
        target = case["target_factor_of_safety"]
        design = fatigue_stress_state(d_req, fatigue["M"][idx], fatigue["T"][idx], fatigue["N"][idx],
                                      fatigue["Kf"][idx], fatigue["Kfs"][idx],
                                      settings["torque_alternating_fraction"])
        Se_req = float(marin_factors(d_req, Sut, settings)["Se"])
        # A constant-n line is the failure line shrunk toward the origin by 1/n on both axes.
        design_line = criterion_envelope(sm, Se_req, Sut, Sy, selected) / target
        keep = design_line >= 0
        ax.plot(sm[keep] / target / scale, design_line[keep] / scale, color=CRITERION_COLORS[selected],
                linewidth=1.0, linestyle=":",
                label=f"{FATIGUE_CRITERIA[selected]} design line, n = {format_number(target)}")
        dx, dy = design["vm_m"] / scale, design["vm_a"] / scale
        d_label = f"d = {units.number(d_req, 'diameter')} {units.labels['diameter']}"
        ax.plot(dx, dy, "*", color="green", markersize=14, markeredgecolor="black", zorder=5,
                label=f"target FOS = {format_number(target)} point: required {d_label} "
                      f"(current {units.number(fatigue['diameter'][idx], 'diameter')} "
                      f"{units.labels['diameter']})")
        ax.annotate(f"required {d_label}\n(n = {format_number(target)})", xy=(dx, dy),
                    xytext=(10, 10), textcoords="offset points", fontsize=8, color="green",
                    fontweight="bold")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(f"von Mises mean stress sigma_m' ({units.labels['stress']})")
    ax.set_ylabel(f"von Mises alternating stress sigma_a' ({units.labels['stress']})")
    ax.set_title(f"Fatigue diagram - critical section x = {units.number(x_arr[idx], 'length')} "
                 f"{units.labels['length']}  (Se = {units.number(Se, 'stress')} {units.labels['stress']})")
    # Below the axes, so it never hides the edge arrow or the envelopes.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
