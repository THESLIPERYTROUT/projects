import matplotlib.pyplot as plt
import numpy as np

#Units: SI (m, N, N·m, Pa)

length = 1.05
mesh_density_factor = 0.01
simulated_points = int(length / mesh_density_factor)

supports = [
    {"position": 0.0,    "type": "bearing"},
    {"position": length, "type": "bearing"},
]

material_properties = {
    "young_modulus": 200e9,
    "poisson_ratio": 0.3,
    "density": 7850,
}

# Each segment defines a shaft section with a given diameter.
# Segments must be contiguous and together span the full beam length.
geometry = [
    {"start": 0.00, "end": 1.05, "diameter": 0.05},
]

loads = [
    {
        "type": "point_load",
        "position": 0.40,
        "force":  (0, -3760, 10340),
        "moment": (-3100, 0, 0),
    },
    {
        "type": "point_load",
        "position": 0.75,
        "force":  (0, -9640, -20660),
        "moment": (3100, 0, 0),
    },
]

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
    n >= 0 : Macaulay bracket — returns (x-a)^n if x > a, else 0
    """
    if n < 0:
        return 1.0 if np.isclose(x, a) else 0.0
    else:
        return (x - a)**n if x > a else 0.0

def build_load_list(loads, supports):
    """
    Returns a unified list of all loads including reactions.
    Reactions are solved here from equilibrium, then injected back
    as point loads/moments so the singularity engine treats everything uniformly.
    """

    all_loads = loads.copy()

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

            statics_solver_matrix.append(col_Fx)
            statics_solver_matrix.append(col_Fy)
            statics_solver_matrix.append(col_Fz)
            statics_solver_matrix.append(col_Mx)
            unknowns.append({"dof": "Fx", "position": d})
            unknowns.append({"dof": "Fy", "position": d})
            unknowns.append({"dof": "Fz", "position": d})
            unknowns.append({"dof": "Mx", "position": d})
        else:
            raise NotImplementedError(f"Support type '{support['type']}' not implemented")

    A = np.array(statics_solver_matrix).T  # (6, n_unknowns)

    b = np.array([
        -sum(load["force"][0]  for load in loads),   # ΣFx
        -sum(load["force"][1]  for load in loads),   # ΣFy
        -sum(load["force"][2]  for load in loads),   # ΣFz
        -sum(load["moment"][0] for load in loads),   # ΣMx
        -sum(load["moment"][1] - load["force"][2] * load["position"] for load in loads),  # ΣMy
        -sum(load["moment"][2] + load["force"][1] * load["position"] for load in loads),  # ΣMz
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

            # applied moments
            results["M_z"][i] += Mz * S(x, a, 0)
            results["M_y"][i] += My * S(x, a, 0)

            # torsion and axial
            results["T"][i]   += Mx * S(x, a, 0)
            results["N"][i]   += Fx * S(x, a, 0)

    return results

def compute_deflection(x_arr, results, geometry, material_properties):
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

    x_left = x_arr[0]
    x_right = x_arr[-1]

    # Solve:
    # delta(left)  = delta_raw(left)  + C1*x_left  + C2 = 0
    # delta(right) = delta_raw(right) + C1*x_right + C2 = 0
    A = np.array([
        [x_left,  1],
        [x_right, 1],
    ])

    b_y = np.array([
        -delta_y_raw[0],
        -delta_y_raw[-1],
    ])

    b_z = np.array([
        -delta_z_raw[0],
        -delta_z_raw[-1],
    ])

    C1_y, C2_y = np.linalg.solve(A, b_y)
    C1_z, C2_z = np.linalg.solve(A, b_z)

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

    # torsional shear stress magnitude at each point
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
    print(f"  ΣMz = {sum(l['moment'][2] - l['force'][1] * l['position'] for l in all_loads):.4f} N·m")

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

if __name__ == "__main__":

    prepare_geometry(geometry)

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

    theta_y, theta_z, delta_y, delta_z = compute_deflection(x_arr, results, geometry, material_properties)
    results["theta_y"] = theta_y
    results["theta_z"] = theta_z
    results["delta_y"] = delta_y
    results["delta_z"] = delta_z

    #stress recovery along beam centerline
    sigma, tau, sigma_vm = compute_stress(x_arr, results, geometry)

    #find critical section
    critical_idx = np.argmax(sigma_vm)
    critical_x   = x_arr[critical_idx]
    print(f"Critical section at x = {critical_x:.4f} m")
    print(f"  sigma_vm = {sigma_vm[critical_idx]/1e6:.2f} MPa")
    print(f"  sigma    = {sigma[critical_idx]/1e6:.2f} MPa")
    print(f"  tau      = {tau[critical_idx]/1e6:.2f} MPa")
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

    Sy = 200e6 # yield strength (Pa)
    tau_max = np.nanmax(tau_cs)
    factor_of_safety = Sy / (2 * tau_max)
    print(f"Estimated factor of safety against yielding (using max shear): {factor_of_safety:.2f}")


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
