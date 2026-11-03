import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import contextlib
    import io
    from unittest.mock import patch

    import marimo as mo
    import matplotlib.pyplot as plt

    import beam_stress_sim as sim
    return contextlib, io, mo, patch, plt, sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Beam stress analysis

    This notebook walks a shaft case through each stage of `beam_stress_sim.py`: loads,
    reactions, internal loads, deflection, stress and, when the case asks for it, fatigue.
    All the calculation happens in that script. This notebook only displays its results,
    so the numbers here are the ones the script prints.

    Pick a case below and everything updates. If nothing appears when you first open the
    notebook, click the run button at the bottom right.
    """)
    return


@app.cell
def _(mo, sim):
    _case_files = sorted((sim.SCRIPT_DIR / "beam_cases").glob("*.toml"))
    case_picker = mo.ui.dropdown(
        options={path.name: path for path in _case_files},
        value=sim.DEFAULT_CASE.name,
        allow_select_none=False,
        label="Case file",
    )
    case_picker
    return (case_picker,)


@app.cell
def _(case_picker, contextlib, io, patch, plt, sim):
    # run_case is the one place the solver stages are wired together, so this cell
    # drives it instead of repeating them. Two stand-ins observe it without changing it:
    # plt.show records each finished figure, and the print_report spy keeps the solved
    # arrays that run_case doesn't return.
    #
    # matplotlib's state is per process and `marimo run` serves every viewer from one
    # process, so viewers running this cell at the same moment can collide.
    _figures = []
    _stdout = io.StringIO()
    case = sim.load_case(case_picker.value)
    with (
        patch.object(sim.plt, "show", lambda: _figures.append(plt.gcf())),
        patch.object(sim, "print_report", wraps=sim.print_report) as _report_spy,
        contextlib.redirect_stdout(_stdout),
    ):
        sim.run_case(case)
    report_text = _stdout.getvalue()
    _, x_arr, _, _, stresses, fos_msst, critical_idx, _, fatigue = _report_spy.call_args.args

    # In the order run_case shows them. Cases without [fatigue] stop after "section".
    figures = dict(zip(
        ("preview", "loads", "deflection", "mohr", "section", "fatigue_fos", "fatigue_diagram"),
        _figures,
    ))
    return case, critical_idx, fatigue, figures, fos_msst, report_text, stresses, x_arr


@app.cell
def _(case, sim):
    units = sim.ReportUnits(case["unit_system"])

    def quantity(value, kind):
        """A value with its unit, in the case's unit system, safe to use in Markdown."""
        return f"{units.number(value, kind)} {units.labels[kind]}".replace("*", "·")
    return (quantity,)


@app.cell(hide_code=True)
def _(case, critical_idx, fatigue, fos_msst, mo, quantity, sim, x_arr):
    _target = case["target_factor_of_safety"]
    _rows = [
        ("Unit system", case["unit_system"]),
        ("Beam length", quantity(case["length"], "length")),
        ("Shaft segments", len(case["geometry"])),
        ("Supports", len(case["supports"])),
        ("Applied loads", len(case["loads"])),
        ("Target factor of safety", sim.format_number(_target)),
        (
            "Minimum static factor of safety",
            f"{sim.format_number(fos_msst[critical_idx])} "
            f"at x = {quantity(x_arr[critical_idx], 'length')}",
        ),
    ]
    if fatigue is not None:
        _idx = fatigue["critical_idx"]
        _rows.append((
            "Minimum fatigue factor of safety",
            f"{sim.format_number(fatigue['fos_fatigue'][_idx])} "
            f"at x = {quantity(x_arr[_idx], 'length')}",
        ))
    _table = "\n".join(f"| {_name} | {_value} |" for _name, _value in _rows)
    mo.md(f"""
    ## Results at a glance

    | Quantity | Value |
    |---|---|
    {_table}
    """)
    return


@app.cell
def _(mo, report_text):
    mo.accordion({"Full text report": mo.plain_text(report_text)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The model

    The shaft is a line along $x$ made of round segments, each with its own diameter. For
    each segment the solver computes

    $$I = \frac{\pi d^4}{64}, \qquad J = \frac{\pi d^4}{32}, \qquad
      A = \frac{\pi d^2}{4}, \qquad c = \frac{d}{2}$$

    Stations are evenly spaced along the shaft, plus one at every stress raiser so that
    shoulders are always evaluated.

    A spur gear is converted to forces on the centerline. Its power $P$ (W) and speed $N$
    (RPM) come from the driving gear, and with pitch diameter $d$ and pressure angle
    $\phi$:

    $$W_t = \frac{60 P}{\pi d N}, \qquad W_r = W_t \tan\phi, \qquad
      M_x = \pm W_t \frac{d}{2}$$

    $M_x$ is positive at the driving gear and negative at a driven gear. In the preview,
    $x$ is to scale but gears are schematic symbols.
    """)
    return


@app.cell
def _(figures):
    figures["preview"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reactions

    Bearings react $F_y$ and $F_z$. A fixed support reacts all six components. The
    unknowns come from global equilibrium, with moments taken about $x = 0$:

    $$\sum F_x = \sum F_y = \sum F_z = 0, \qquad \sum M_x = 0$$

    $$\sum \left(M_y - F_z x\right) = 0, \qquad \sum \left(M_z + F_y x\right) = 0$$

    The solver rejects statically indeterminate or unstable support layouts, then checks
    these residuals again before reporting. The values are under **SUPPORT REACTIONS** in
    the text report.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Internal loads

    With the reactions known, every force and couple is a point load at some position
    $a$, and singularity functions sum them along the shaft. $\langle x-a\rangle^n$ is
    $(x-a)^n$ for $x \ge a$ and zero before it:

    $$V_y = \sum F_y \langle x-a\rangle^0, \qquad V_z = \sum F_z \langle x-a\rangle^0$$

    $$M_z = \sum F_y \langle x-a\rangle^1 - \sum M_{z} \langle x-a\rangle^0, \qquad
      M_y = \sum F_z \langle x-a\rangle^1 + \sum M_{y} \langle x-a\rangle^0$$

    $$T = \sum M_x \langle x-a\rangle^0, \qquad N = \sum F_x \langle x-a\rangle^0$$

    Shear steps at each point load, and moment changes slope there. The red dashed line
    marks the critical section from stage 5.
    """)
    return


@app.cell
def _(figures):
    figures["loads"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Slope and deflection

    Slope and deflection come from integrating curvature twice with the trapezoid rule,
    using the local $I$ so that diameter steps count. For the $y$ direction (the $z$
    direction uses $M_y$ the same way):

    $$\theta_y(x) = \int_0^x \frac{M_z}{E I}\,dx + C_1, \qquad
      \delta_y(x) = \int_0^x \theta_y\,dx + C_2$$

    The constants come from the supports: $\delta = 0$ at every support, and also
    $\theta = 0$ at a fixed support. This is Euler-Bernoulli bending, so shear deflection
    isn't included.
    """)
    return


@app.cell
def _(figures):
    figures["deflection"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Stress and the critical section

    At the outer fiber of each station, with resultant bending
    $M = \sqrt{M_y^2 + M_z^2}$:

    $$\sigma = \frac{M c}{I} + \frac{N}{A}, \qquad \tau = \frac{T c}{J}, \qquad
      \sigma_{vm} = \sqrt{\sigma^2 + 3\tau^2}$$

    The **critical section** is the station with the lowest factor of safety under the
    maximum shear stress (Tresca) theory. The stresses are nominal here, with no stress
    concentration:

    $$n = \frac{S_y}{2\tau_{max}}, \qquad \tau_{max} = \sqrt{\left(\frac{\sigma}{2}\right)^2 + \tau^2}$$

    The report sizes the shaft from the same theory, ignoring axial load:

    $$d_{req} = \left(\frac{32 n}{\pi S_y}\sqrt{M^2 + T^2}\right)^{1/3}$$
    """)
    return


@app.cell(hide_code=True)
def _(case, critical_idx, mo, quantity, sim, stresses, x_arr):
    _sigma, _tau, _sigma_vm = stresses
    _diameter = sim.section_at(x_arr[critical_idx], case["geometry"])["diameter"]
    mo.md(f"""
    In this case the critical section is at **x = {quantity(x_arr[critical_idx], "length")}**,
    where d = {quantity(_diameter, "diameter")}. There,
    σ = {quantity(_sigma[critical_idx], "stress")},
    τ = {quantity(_tau[critical_idx], "stress")} and
    σ<sub>vm</sub> = {quantity(_sigma_vm[critical_idx], "stress")}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mohr's circle

    Mohr's circle shows the plane stress state at that outer fiber, with $\sigma_y = 0$:

    $$C = \frac{\sigma}{2}, \qquad R = \sqrt{C^2 + \tau^2} = \tau_{max}, \qquad
      \sigma_{1,2} = C \pm R, \qquad \theta_p = \tfrac{1}{2}\operatorname{atan2}(\tau, C)$$
    """)
    return


@app.cell
def _(figures, mo):
    # Narrower than the page: at full width this tall figure would scroll in its cell.
    mo.as_html(figures["mohr"]).style({"max-width": "580px"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-section heatmaps

    The heatmaps evaluate the stress across the critical section instead of only at the
    worst fiber. Bending varies linearly across the section and torsional shear grows with
    radius $r = \sqrt{y^2 + z^2}$:

    $$\sigma(y, z) = \frac{N}{A} + \frac{M_z\,y}{I} + \frac{M_y\,z}{I}, \qquad
      \tau(r) = \frac{T r}{J}$$

    The peak von Mises stress sits on the rim, on the line of the resultant bending
    moment.
    """)
    return


@app.cell
def _(figures):
    figures["section"]
    return


@app.cell(hide_code=True)
def _(fatigue, mo):
    mo.stop(
        fatigue is None,
        mo.md("""
    ## 6. Fatigue

    This case has no `[fatigue]` section, so the analysis stops at static stress.
    """),
    )
    mo.md(r"""
    ## 6. Fatigue

    A rotating shaft sees fully reversed bending, so static yield isn't the whole story.
    This stage checks for infinite life, following Shigley chapters 6 and 7.

    **Endurance limit.** The specimen limit $S_e'$ is $0.5\,S_{ut}$ up to 1400 MPa and
    700 MPa above that, unless the case sets one. Marin factors for surface, size, load,
    temperature and reliability then reduce it:

    $$S_e = k_a k_b k_c k_d k_e S_e'$$

    **Stress concentration.** Each shoulder gets a $K_t$ from fits to Peterson's charts,
    reduced by the notch sensitivity $q$. The solver applies $K_f$ and $K_{fs}$ only at the
    shoulder stations, which is why the factor of safety dips sharply at those points:

    $$K_f = 1 + q\,(K_t - 1), \qquad K_{fs} = 1 + q_s\,(K_{ts} - 1)$$

    **Alternating and mean stress.** Bending alternates, while axial load and torque are
    steady. $f$ is the case's `torque_alternating_fraction`:

    $$\sigma_a = K_f \frac{32 M}{\pi d^3}, \qquad
      \sigma_m = K_f \frac{4 |N|}{\pi d^2}, \qquad
      \tau_m = K_{fs} \frac{16 |T|}{\pi d^3}, \qquad \tau_a = f\,\tau_m$$

    $$\sigma_a' = \sqrt{\sigma_a^2 + 3\tau_a^2}, \qquad
      \sigma_m' = \sqrt{\sigma_m^2 + 3\tau_m^2}$$
    """)
    return


@app.cell(hide_code=True)
def _(case, fatigue, mo, quantity, sim, x_arr):
    mo.stop(fatigue is None)
    _idx = fatigue["critical_idx"]
    _name = sim.FATIGUE_CRITERIA[case["fatigue"]["criterion"]]
    mo.md(f"""
    In this case the fatigue critical section is at **x = {quantity(x_arr[_idx], "length")}**
    under the selected {_name} criterion, where
    S<sub>e</sub> = {quantity(fatigue["Se"][_idx], "stress")},
    K<sub>f</sub> = {sim.format_number(fatigue["Kf"][_idx])} and
    n = {sim.format_number(fatigue["fos_fatigue"][_idx])}. It can differ from the static
    critical section because K<sub>f</sub> and the size factor vary along the shaft.
    """)
    return


@app.cell(hide_code=True)
def _(fatigue, mo):
    mo.stop(fatigue is None)
    mo.md(r"""
    ### Factor of safety along the shaft

    The fatigue criteria differ in how they trade alternating stress against mean stress:

    | Criterion | Factor of safety |
    |---|---|
    | Modified Goodman | $\dfrac{1}{n} = \dfrac{\sigma_a'}{S_e} + \dfrac{\sigma_m'}{S_{ut}}$ |
    | Soderberg | $\dfrac{1}{n} = \dfrac{\sigma_a'}{S_e} + \dfrac{\sigma_m'}{S_y}$ |
    | ASME elliptic | $\dfrac{1}{n} = \sqrt{\left(\dfrac{\sigma_a'}{S_e}\right)^2 + \left(\dfrac{\sigma_m'}{S_y}\right)^2}$ |
    | Gerber | $n = \dfrac{2 S_e}{\sigma_a' + \sqrt{\sigma_a'^2 + \left(2 \sigma_m' S_e / S_{ut}\right)^2}}$ |

    The plot also shows first-cycle yield (Langer), using the peak stress with $K_f$
    included:

    $$n_y = \frac{S_y}{\sqrt{(\sigma_a + \sigma_m)^2 + 3(\tau_a + \tau_m)^2}}$$

    Curves are capped at the value on the $y$ axis label, so unloaded stations with an
    infinite factor of safety don't flatten the plot.
    """)
    return


@app.cell
def _(fatigue, figures, mo):
    mo.stop(fatigue is None)
    figures["fatigue_fos"]
    return


@app.cell(hide_code=True)
def _(fatigue, mo):
    mo.stop(fatigue is None)
    mo.md(r"""
    ### Fatigue diagram

    This is the fatigue critical section on mean-alternating axes. Each curve is a failure
    envelope, and the operating point has coordinates $(\sigma_m', \sigma_a')$. The load
    line runs from the origin through that point, so the factor of safety is how far the
    point could move along the line before it reaches the envelope. A point inside an
    envelope has $n > 1$. A point far outside the envelopes is drawn as a red arrow at the
    edge of the plot, and its actual stresses are in the legend.

    The green star shows the same section at the diameter that just meets the target
    factor of safety, with $K_f$ held fixed. $S_e$ changes with diameter through $k_b$, so
    the required diameter is found by bisection.
    """)
    return


@app.cell
def _(fatigue, figures, mo):
    mo.stop(fatigue is None)
    # Narrower than the page: at full width this tall figure would scroll in its cell.
    mo.as_html(figures["fatigue_diagram"]).style({"max-width": "640px"})
    return


if __name__ == "__main__":
    app.run()
