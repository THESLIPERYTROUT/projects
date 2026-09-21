import math

import matplotlib.pyplot as plt

# all lengths in inches
lid_weight: float = 71.32  # pounds-force

# Coordinate system:
# x = hinge axis of rotation
# y = toward the front face of the lid
# z = up
lid_cg_y: float = 14.76
lid_cg_z: float = -7.33

# SolidWorks mass moment of inertia about the hinge axis at the output coordinate system.
lid_axis_inertia_lbm_in2: float = 28861.83

gas_strut_compression_force: int = 157  # lbf
gas_strut_extension_force: int = 157
gas_strut_stroke: float = 9.84
gas_strut_force_per_length: float = (
    gas_strut_extension_force - gas_strut_compression_force
) / gas_strut_stroke
gas_strut_base_to_axis: float = 20.79
gas_strut_top_to_axis: float = 3.575
hinge_friction_torque_lb_in: float = 15.0
applied_force_y: float = 23.56
applied_force_z: float = -7.33
equilibrium_force_plot_limit: float = 200.0

DOMAIN_DEGREES = 75
GRAVITY_IN_PER_S2 = 386.089
INITIAL_ANGULAR_VELOCITY = 0.0  # rad/s


def build_lid_response(
    lid_weight,
    lid_cg_y,
    lid_cg_z,
    lid_axis_inertia_lbm_in2,
    gas_strut_force_per_length,
    gas_strut_base_to_axis,
    gas_strut_top_to_axis,
    hinge_friction_torque_lb_in,
    initial_angular_velocity=0.0,
    domain=DOMAIN_DEGREES,
):
    angles = list(range(domain + 1))
    weight_torques = []
    strut_torques = []
    raw_net_torques = []
    net_torques = []
    equilibrium_forces = []

    for degree in angles:
        weight_torque_value = weight_torque(degree, lid_weight, lid_cg_y, lid_cg_z)
        strut_torque_value = strut_torque(
            degree,
            gas_strut_force_per_length,
            gas_strut_base_to_axis,
            gas_strut_top_to_axis,
            gas_strut_compression_force,
        )
        raw_net_torque = strut_torque_value - weight_torque_value
        net_torque = apply_hinge_friction(raw_net_torque, hinge_friction_torque_lb_in)

        weight_torques.append(-weight_torque_value)
        strut_torques.append(strut_torque_value)
        raw_net_torques.append(raw_net_torque)
        net_torques.append(net_torque)
        equilibrium_forces.append(
            equilibrium_force(degree, net_torque, applied_force_y, applied_force_z)
        )

    angular_velocity_rad_s, time_s = integrate_lid_dynamics(
        angles,
        net_torques,
        lid_axis_inertia_lbm_in2,
        initial_angular_velocity,
    )

    angular_velocity_deg_s = [math.degrees(value) for value in angular_velocity_rad_s]

    return {
        "angles": angles,
        "weight_torques": weight_torques,
        "strut_torques": strut_torques,
        "raw_net_torques": raw_net_torques,
        "net_torques": net_torques,
        "equilibrium_forces": equilibrium_forces,
        "angular_velocity_rad_s": angular_velocity_rad_s,
        "angular_velocity_deg_s": angular_velocity_deg_s,
        "time_s": time_s,
        "moment_of_inertia": lid_moment_of_inertia(lid_axis_inertia_lbm_in2),
        "hinge_friction_torque_lb_in": hinge_friction_torque_lb_in,
    }


def plot_lid_response(results):
    fig, (torque_ax, velocity_ax, force_ax) = plt.subplots(3, 1, figsize=(10, 13), sharex=True)

    torque_ax.plot(results["angles"], results["net_torques"], color="blue", label="Net Torque")
    torque_ax.plot(
        results["angles"],
        results["raw_net_torques"],
        color="cyan",
        linestyle="--",
        label="Net Torque (No Hinge Friction)",
    )
    torque_ax.plot(results["angles"], results["weight_torques"], color="red", label="Weight Torque")
    torque_ax.plot(results["angles"], results["strut_torques"], color="green", label="Strut Torque")
    torque_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    torque_ax.set_ylabel("Torque (lb-in)")
    torque_ax.set_title(
        f"Lid Torque and Dynamics {gas_strut_compression_force} lbf, "
        f"{results['hinge_friction_torque_lb_in']:.1f} lb-in hinge friction"
    )
    torque_ax.grid(True, alpha=0.3)
    torque_ax.legend(loc="best")

    velocity_ax.plot(
        results["angles"],
        results["angular_velocity_deg_s"],
        color="purple",
        label="Angular Velocity",
    )
    velocity_ax.set_xlabel("Lid Angle (degrees)")
    velocity_ax.set_ylabel("Angular Velocity (deg/s)")
    velocity_ax.grid(True, alpha=0.3)
    velocity_ax.legend(loc="best")

    clipped_force_values = clip_values(
        results["equilibrium_forces"],
        equilibrium_force_plot_limit,
    )
    force_ax.plot(
        results["angles"],
        clipped_force_values,
        color="orange",
        label="Applied Force for Equilibrium",
    )
    saturated_angles = []
    saturated_force_values = []
    for angle, force in zip(results["angles"], results["equilibrium_forces"]):
        if math.isnan(force):
            continue
        if abs(force) > equilibrium_force_plot_limit:
            saturated_angles.append(angle)
            saturated_force_values.append(
                math.copysign(equilibrium_force_plot_limit, force)
            )

    if saturated_angles:
        force_ax.scatter(
            saturated_angles,
            saturated_force_values,
            color="darkred",
            s=18,
            label=f"Clipped Beyond +/-{equilibrium_force_plot_limit:.0f} lbf",
        )

    force_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    force_ax.set_xlabel("Lid Angle (degrees)")
    force_ax.set_ylabel("Applied Force (lbf)")
    force_ax.set_ylim(-equilibrium_force_plot_limit, equilibrium_force_plot_limit)
    force_ax.grid(True, alpha=0.3)
    force_ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig("lid_torque.png", dpi=150, bbox_inches="tight")


def integrate_lid_dynamics(
    angles,
    net_torques,
    lid_axis_inertia_lbm_in2,
    initial_angular_velocity=0.0,
):
    moment_of_inertia = lid_moment_of_inertia(lid_axis_inertia_lbm_in2)
    delta_theta = math.radians(angles[1] - angles[0])
    angular_velocity = [initial_angular_velocity]
    elapsed_time = [0.0]

    for index in range(1, len(angles)):
        average_torque = 0.5 * (net_torques[index - 1] + net_torques[index])
        omega_previous = angular_velocity[-1]

        # Work-energy integration across each angle step.
        omega_squared = omega_previous**2 + (2 * average_torque * delta_theta) / moment_of_inertia
        omega_next = math.sqrt(max(0.0, omega_squared))
        angular_velocity.append(omega_next)

        omega_average = 0.5 * (omega_previous + omega_next)
        if omega_average > 0:
            elapsed_time.append(elapsed_time[-1] + delta_theta / omega_average)
        else:
            elapsed_time.append(elapsed_time[-1])

    return angular_velocity, elapsed_time


def lid_moment_of_inertia(lid_axis_inertia_lbm_in2):
    return lid_axis_inertia_lbm_in2 / GRAVITY_IN_PER_S2


def apply_hinge_friction(net_torque, hinge_friction_torque_lb_in):
    if net_torque > 0:
        return max(0.0, net_torque - hinge_friction_torque_lb_in)
    if net_torque < 0:
        return min(0.0, net_torque + hinge_friction_torque_lb_in)
    return 0.0


def equilibrium_force(degree, net_torque, applied_force_y, applied_force_z):
    rad = math.radians(degree)
    applied_y_pos = applied_force_y * math.cos(rad) - applied_force_z * math.sin(rad)
    if abs(applied_y_pos) < 1e-9:
        return math.nan
    return -net_torque / applied_y_pos


def clip_values(values, limit):
    clipped_values = []
    for value in values:
        if math.isnan(value):
            clipped_values.append(math.nan)
        else:
            clipped_values.append(max(-limit, min(limit, value)))
    return clipped_values


def weight_torque(degree, lid_weight, lid_cg_y, lid_cg_z):
    rad = math.radians(degree)
    cg_y_pos = lid_cg_y * math.cos(rad) - lid_cg_z * math.sin(rad)
    return lid_weight * cg_y_pos


def strut_torque(
    degree,
    gas_strut_force_per_length,
    gas_strut_base_to_axis,
    gas_strut_top_to_axis,
    gas_strut_compression_force,
):
    alpha = math.radians(degree)
    strut_length = math.sqrt(
        gas_strut_base_to_axis**2
        + gas_strut_top_to_axis**2
        - 2 * gas_strut_base_to_axis * gas_strut_top_to_axis * math.cos(alpha)
    )  # law of cosines
    strut_angle = math.acos(
        (gas_strut_base_to_axis**2 - strut_length**2 - gas_strut_top_to_axis**2)
        / (-2 * gas_strut_top_to_axis * strut_length)
    )
    perpendicular_force = (
        gas_strut_force_per_length * (strut_length - gas_strut_stroke)
        + gas_strut_compression_force
    ) * math.sin(strut_angle)
    return 2 * perpendicular_force * gas_strut_top_to_axis


def main():
    results = build_lid_response(
        lid_weight,
        lid_cg_y,
        lid_cg_z,
        lid_axis_inertia_lbm_in2,
        gas_strut_force_per_length,
        gas_strut_base_to_axis,
        gas_strut_top_to_axis,
        hinge_friction_torque_lb_in,
        INITIAL_ANGULAR_VELOCITY,
    )
    plot_lid_response(results)

    max_velocity = max(results["angular_velocity_deg_s"])
    max_velocity_angle = results["angles"][
        results["angular_velocity_deg_s"].index(max_velocity)
    ]
    print(f"Axis inertia used: {results['moment_of_inertia']:.2f} lbf-in-s^2")
    print(f"Hinge friction used: {results['hinge_friction_torque_lb_in']:.1f} lb-in")
    print(f"Peak angular velocity: {max_velocity:.1f} deg/s at {max_velocity_angle} deg")


if __name__ == "__main__":
    main()

