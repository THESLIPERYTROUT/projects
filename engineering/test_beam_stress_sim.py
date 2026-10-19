"""Run with: python -m unittest discover -s engineering_shit -p 'test_*.py'."""

import io
import re
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import beam_stress_sim as sim


def case_text(system="Imperial", *, explicit_power_unit=None):
    # The same physical case in each system; constants are independent of the
    # production conversion table so tests also catch incorrect unit factors.
    imperial = system.casefold() == "imperial"
    length = 1.0 if imperial else 0.0254
    force = 1.0 if imperial else 4.4482216152605
    moment = 1.0 if imperial else 0.1129848290276167
    stress = 1.0 if imperial else 6894.757293168361
    density = 1.0 if imperial else 27679.904710203125
    power = 1.0 if imperial else 745.69987
    power_setting = "" if explicit_power_unit is None else f'power_unit = "{explicit_power_unit}"'
    return f'''
[units]
system = "{system}"
[beam]
length = {20 * length}
mesh_density_factor = {0.125 * length}
[material]
young_modulus = {29e6 * stress}
yield_strength = {36000 * stress}
density = {0.284 * density}
poisson_ratio = 0.29
[design]
target_factor_of_safety = 2.5
[[supports]]
position = {2 * length}
type = "fixed"
[[geometry]]
start = 0.0
end = {10 * length}
diameter = {1 * length}
[[geometry]]
start = {10 * length}
end = {20 * length}
diameter = {2 * length}
[[loads]]
type = "point load"
position = {20 * length}
force = [{10 * force}, {-20 * force}, {30 * force}]
[[loads]]
type = "point moment"
position = {15 * length}
moment = [{-40 * moment}, {50 * moment}, {-60 * moment}]
[[loads]]
type = "driving gear"
"meshing orientation" = "top"
position = {5 * length}
diameter = {4 * length}
power = {power}
{power_setting}
speed = 600
tooth_angle = 25
[[loads]]
type = "driven gear"
"meshing orientation" = "top"
position = {12 * length}
diameter = {6 * length}
tooth_angle = 20
'''


def fatigue_case_text(system="Imperial", criterion="goodman"):
    """case_text plus Sut, a [fatigue] section and a fillet at the x = 10 in shoulder."""
    imperial = system.casefold() == "imperial"
    stress = 1.0 if imperial else 6894.757293168361
    length = 1.0 if imperial else 0.0254
    temperature = 300 if imperial else (300 - 32) * 5 / 9
    text = case_text(system)
    text = text.replace(f"yield_strength = {36000 * stress}\n",
                        f"yield_strength = {36000 * stress}\nultimate_strength = {58000 * stress}\n")
    text = text.replace("[design]", f'''[fatigue]
criterion = "{criterion}"
surface_finish = "machined"
reliability = 0.99
temperature = {temperature}
torque_alternating_fraction = 0.1
[design]''')
    return text.replace(f"start = {10 * length}\n",
                        f"start = {10 * length}\nfillet_radius = {0.05 * length}\n")


def load_text(content):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "case.toml"
        path.write_text(content, encoding="utf-8")
        return sim.load_case(path)


def solve(case):
    sim.prepare_geometry(case["geometry"])
    x = sim.build_stations(case["length"], case["simulated_points"], case["geometry"])
    loads = sim.build_load_list(case["loads"], case["supports"])
    results = sim.compute_internal_loads(x, loads)
    return x, results


class InputUnitsTests(unittest.TestCase):
    def load_text(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.toml"
            path.write_text(content, encoding="utf-8")
            return sim.load_case(path)

    def test_all_inputs_normalize_to_same_si_values(self):
        si = self.load_text(case_text("SI"))
        imperial = self.load_text(case_text())
        for key in ("length", "mesh_density_factor", "simulated_points",
                    "yield_strength", "target_factor_of_safety"):
            self.assertAlmostEqual(si[key], imperial[key])
        for key in si["material_properties"]:
            self.assertAlmostEqual(si["material_properties"][key], imperial["material_properties"][key])
        for group in ("supports", "geometry", "loads"):
            for a, b in zip(si[group], imperial[group]):
                for key in a:
                    if key in ("power_input", "power_unit"):
                        continue
                    if isinstance(a[key], str):
                        self.assertEqual(a[key], b[key])
                    else:
                        np.testing.assert_allclose(a[key], b[key], rtol=1e-12)
        self.assertEqual(imperial["unit_system"], "Imperial")
        self.assertEqual(imperial["loads"][2]["power_unit"], "hp")

    def test_equivalent_cases_produce_same_results(self):
        outputs = []
        for system in ("SI", "Imperial"):
            case = self.load_text(case_text(system))
            sim.prepare_geometry(case["geometry"])
            loads = sim.build_load_list(case["loads"], case["supports"])
            sim.validate_global_equilibrium(loads)
            x = np.linspace(0, case["length"], case["simulated_points"])
            results = sim.compute_internal_loads(x, loads)
            deflection = sim.compute_deflection(
                x, results, case["geometry"], case["material_properties"], case["supports"]
            )
            stresses = sim.compute_stress(x, results, case["geometry"])
            outputs.append([*results.values(), *deflection, *stresses])
        for a, b in zip(*outputs):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-8)

    def test_gear_torque_is_in_newton_meters(self):
        case = self.load_text(case_text())
        loads = sim.build_load_list(case["loads"], case["supports"])
        expected_torque = 745.69987 / (600 * 2 * np.pi / 60)
        self.assertAlmostEqual(loads[2]["moment"][0], expected_torque)
        self.assertAlmostEqual(loads[3]["moment"][0], -expected_torque)
        self.assertAlmostEqual(loads[2]["force"][2], expected_torque / (4 * 0.0254 / 2))

    def test_explicit_power_unit_overrides_system_default(self):
        for system in ("SI", "Imperial"):
            for unit, factor in sim.POWER_UNITS.items():
                with self.subTest(system=system, unit=unit):
                    case = self.load_text(case_text(system, explicit_power_unit=unit))
                    gear = case["loads"][2]
                    self.assertEqual(gear["power"], gear["power_input"] * factor)

    def test_mesh_orientation_is_required_and_valid_for_both_gears(self):
        for gear_type in ("driving gear", "driven gear"):
            marker = f'type = "{gear_type}"\n"meshing orientation" = "top"'
            with self.subTest(gear_type=gear_type):
                with self.assertRaisesRegex(KeyError, "meshing orientation"):
                    self.load_text(case_text().replace(marker, f'type = "{gear_type}"'))
                with self.assertRaisesRegex(ValueError, "meshing orientation"):
                    self.load_text(case_text().replace(marker, marker.replace('"top"', '"diagonal"')))
        normalized = self.load_text(case_text().replace('"top"', '" TOP "'))
        self.assertEqual(normalized["loads"][2]["meshing orientation"], "top")

    def test_gear_force_directions_match_radial_separation_and_torque(self):
        # (radial direction toward mate, driving Fy/Fz coefficients, driven coefficients)
        # Each coefficient tuple is (coefficient of Wt, coefficient of Wr).
        expected = {
            "top": ((1, 0), ((0, -1), (1, 0)), ((0, -1), (-1, 0))),
            "right": ((0, -1), ((1, 0), (0, 1)), ((-1, 0), (0, 1))),
            "bottom": ((-1, 0), ((0, 1), (-1, 0)), ((0, 1), (1, 0))),
            "left": ((0, 1), ((-1, 0), (0, -1)), ((1, 0), (0, -1))),
        }
        for orientation, (direction, driving, driven) in expected.items():
            case = self.load_text(case_text().replace('"top"', f'"{orientation}"'))
            loads = sim.build_load_list(case["loads"], case["supports"])
            sim.validate_global_equilibrium(loads)
            torque = 745.69987 / (600 * 2 * np.pi / 60)
            for gear, coefficients, sign in ((loads[2], driving, 1), (loads[3], driven, -1)):
                with self.subTest(orientation=orientation, gear_type=gear["type"]):
                    radius = gear["diameter"] / 2
                    wt = torque / radius
                    wr = wt * np.tan(np.radians(gear["tooth_angle"]))
                    expected_force = [0, *(a * wt + b * wr for a, b in coefficients)]
                    np.testing.assert_allclose(gear["force"], expected_force)
                    contact = np.array([0, *direction]) * radius
                    np.testing.assert_allclose(np.cross(contact, gear["force"]), gear["moment"])
                    self.assertAlmostEqual(gear["moment"][0], sign * torque)
                    self.assertAlmostEqual(np.dot(gear["force"][1:], direction), -wr)

    def test_gear_magnitudes_and_pressure_angle_are_validated(self):
        for old, new in (("speed = 600", "speed = 0"), ("speed = 600", "speed = -600"),
                         ("power = 1.0", "power = -1.0"),
                         ("tooth_angle = 25", "tooth_angle = 90"),
                         ("diameter = 4.0", "diameter = 0.0")):
            with self.subTest(replacement=new), self.assertRaises(ValueError):
                self.load_text(case_text().replace(old, new))

    def test_omitted_units_preserve_si_defaults(self):
        text = case_text("SI").replace('[units]\nsystem = "SI"\n', "")
        case = self.load_text(text)
        self.assertEqual(case["unit_system"], "SI")
        self.assertAlmostEqual(case["length"], 0.508)
        self.assertEqual(case["loads"][2]["power_unit"], "W")

    def test_default_density_is_same_physical_density(self):
        for system in ("SI", "Imperial"):
            text = "\n".join(line for line in case_text(system).splitlines()
                             if not line.startswith("density ="))
            self.assertEqual(self.load_text(text)["material_properties"]["density"], 7850)

    def test_unit_system_validation(self):
        self.assertEqual(self.load_text(case_text("imperial"))["unit_system"], "Imperial")
        with self.assertRaisesRegex(ValueError, "system must be"):
            self.load_text(case_text("metric"))
        with self.assertRaisesRegex(ValueError, "Use \\[units\\]"):
            self.load_text(case_text().replace('[units]\nsystem = "Imperial"', 'units = "Imperial"'))

    def test_position_validation_after_conversion(self):
        text = case_text().replace("position = 20.0", "position = 21.0")
        with self.assertRaisesRegex(ValueError, "outside the beam"):
            self.load_text(text)


class ReportTests(unittest.TestCase):
    def report(self, system, *, unloaded=False, low_strength=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.toml"
            path.write_text(case_text(system), encoding="utf-8")
            case = sim.load_case(path)
        if unloaded:
            case["loads"] = []
        if low_strength:
            case["yield_strength"] = 1000.0
        output = io.StringIO()
        with redirect_stdout(output):
            sim.run_case(case, show_plots=False)
        return output.getvalue()

    def quantity_rows(self, report):
        rows = {}
        for line in report.splitlines():
            match = re.match(r"^  (.+?)\s{2,}([\d,.e+-]+|inf)\s+(m|mm|in|MPa|psi|N\*m|lbf\*in|rad|deg|-)\s*$", line)
            if match:
                name, value, unit = match.groups()
                rows.setdefault(name, (float(value.replace(",", "")), unit))
        return rows

    def test_imperial_report_converts_all_result_quantities(self):
        si = self.quantity_rows(self.report("SI"))
        imperial_text = self.report("Imperial")
        imperial = self.quantity_rows(imperial_text)
        scales = {"m": (1 / 0.0254, "in"), "mm": (1 / 25.4, "in"),
                  "MPa": (1e6 / 6894.757293168361, "psi"),
                  "N*m": (1 / 0.1129848290276167, "lbf*in"),
                  "-": (1, "-"), "rad": (1, "rad"), "deg": (1, "deg")}
        self.assertTrue({"Beam length", "Young's modulus", "Yield strength", "Current diameter",
                         "Von Mises stress", "Normal stress sigma_x", "Torsional shear tau_xy",
                         "Bending M_y", "Bending M_z", "Torque T", "Deflection y", "Deflection z",
                         "Slope about y", "Slope about z", "Principal stress sigma_1",
                         "Principal stress sigma_2", "Required diameter"}.issubset(si))
        for name, (value, unit) in si.items():
            with self.subTest(quantity=name):
                factor, expected_unit = scales[unit]
                converted, actual_unit = imperial[name]
                self.assertEqual(actual_unit, expected_unit)
                # Allow the final terminal rounding in both reports.
                np.testing.assert_allclose(converted, value * factor, rtol=0.002, atol=0.0001 * max(factor, 1))
        self.assertRegex(imperial_text, r"point load\s+20\s+10\s+-20\s+30\s+0\s+0\s+0")
        self.assertRegex(imperial_text, r"point moment\s+15\s+0\s+0\s+0\s+-40\s+50\s+-60")
        self.assertRegex(imperial_text, r"\n\s+5\s+1\s+600\n")  # driving gear: x, hp, RPM
        self.assertNotIn("np.float", imperial_text)
        self.assertNotIn("{'type'", imperial_text)

    def test_known_display_conversions(self):
        units = sim.ReportUnits("Imperial")
        for quantity, value in (
            ("length", 0.0254), ("diameter", 0.0254), ("deflection", 0.0254),
            ("force", 4.4482216152605), ("moment", 0.1129848290276167),
            ("stress", 6894.757293168361), ("power", 745.69987),
        ):
            with self.subTest(quantity=quantity):
                self.assertEqual(units.number(value, quantity), "1")

    def test_recommendations_follow_selected_units(self):
        for system, length_unit, diameter_unit in (("SI", "m", "mm"), ("Imperial", "in", "in")):
            report = self.report(system, low_strength=True)
            self.assertIn("DIAMETER RECOMMENDATIONS", report)
            self.assertIn(f"From [{length_unit}]", report)
            self.assertIn(f"Required [{diameter_unit}]", report)

    def test_unloaded_case_prints_infinite_fos_without_warnings(self):
        with np.errstate(all="raise"):
            report = self.report("Imperial", unloaded=True)
        self.assertRegex(report, r"Factor of safety\s+inf\s+-")
        self.assertIn("All sections meet", report)

    def test_number_format_preserves_small_values_and_removes_negative_zero(self):
        self.assertEqual(sim.format_number(-0.0), "0")
        self.assertEqual(sim.format_number(1200), "1,200")
        self.assertEqual(sim.format_number(1234567), "1,234,567")
        self.assertAlmostEqual(float(sim.format_number(-1.234e-9)), -1.234e-9)
        self.assertEqual(sim.format_number(np.inf), "inf")


class PlotUnitsTests(unittest.TestCase):
    def test_plot_data_labels_and_annotations_follow_units_without_mutating_solver(self):
        sim.plt.switch_backend("Agg")
        for system in ("SI", "Imperial"):
            with self.subTest(system=system), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "case.toml"
                path.write_text(case_text(system), encoding="utf-8")
                case = sim.load_case(path)
                sim.prepare_geometry(case["geometry"])
                loads = sim.build_load_list(case["loads"], case["supports"])
                x = np.linspace(0, case["length"], 40)
                results = sim.compute_internal_loads(x, loads)
                results.update(zip(("theta_y", "theta_z", "delta_y", "delta_z"),
                    sim.compute_deflection(x, results, case["geometry"], case["material_properties"], case["supports"])))
                original_case, original_results = deepcopy(case), deepcopy(results)
                units = sim.ReportUnits(system)
                sim.plt.close("all")
                try:
                    with patch.object(sim.plt, "show"):
                        sim.plot_beam_preview(case)
                        preview = sim.plt.gcf()
                        ax = preview.axes[0]
                        self.assertEqual(ax.get_xlabel(), f"x ({units.labels['length']})")
                        self.assertAlmostEqual(ax.patches[0].get_width(), case["geometry"][0]["end"] / units.scales["length"])
                        text = "\n".join(t.get_text() for t in ax.texts)
                        self.assertIn(units.labels["moment"], text)
                        self.assertIn(units.labels["power"], text)
                        sim.plot_diagrams(x, results, x[10], system)
                        figures = [sim.plt.figure(n) for n in sim.plt.get_fignums()]
                        force_ax = figures[1].axes[0]
                        np.testing.assert_allclose(force_ax.lines[0].get_xdata(), x / units.scales["length"])
                        np.testing.assert_allclose(force_ax.lines[0].get_ydata(), results["V_y"] / units.scales["force"])
                        np.testing.assert_allclose(force_ax.lines[2].get_xdata(), x[10] / units.scales["length"])
                        np.testing.assert_allclose(figures[1].axes[2].lines[0].get_ydata(), results["M_y"] / units.scales["moment"])
                        np.testing.assert_allclose(figures[2].axes[2].lines[0].get_ydata(), results["delta_y"] / units.scales["deflection"])
                        sim.plot_mohrs_circle(10, x, results, case["geometry"], system)
                        mohr = sim.plt.gcf()
                        sigma, tau, _ = sim.compute_stress(x, results, case["geometry"])
                        np.testing.assert_allclose(mohr.axes[0].lines[2].get_xdata(), sigma[10] / units.scales["stress"])
                        np.testing.assert_allclose(mohr.axes[0].lines[2].get_ydata(), -tau[10] / units.scales["stress"])
                        self.assertIn(units.labels["stress"], mohr.axes[0].get_xlabel())
                        Y, Z, *stresses = sim.critical_section_heatmap(10, x, results, case["geometry"])
                        sim.plot_section_heatmap(Y, Z, stresses, x[10], system)
                        heatmap = sim.plt.gcf()
                        self.assertEqual(heatmap.axes[0].get_xlabel(), f"y ({units.labels['diameter']})")
                        contour = heatmap.axes[0].collections[0]
                        self.assertLessEqual(contour.levels[0], np.nanmin(stresses[0]) / units.scales["stress"])
                        self.assertGreaterEqual(contour.levels[-1], np.nanmax(stresses[0]) / units.scales["stress"])
                        for number in sim.plt.get_fignums():
                            sim.plt.figure(number).canvas.draw()
                    self.assertEqual(case, original_case)
                    for key in results:
                        np.testing.assert_array_equal(results[key], original_results[key])
                finally:
                    sim.plt.close("all")


class BeamPreviewTests(unittest.TestCase):
    def test_gears_are_schematic_and_axes_fit_the_shaft(self):
        sim.plt.switch_backend("Agg")
        # A 40 in gear on a 2 in shaft: drawn to scale it would flatten the shaft.
        text = case_text().replace("diameter = 6.0", "diameter = 40.0")
        case = load_text(text)
        sim.prepare_geometry(case["geometry"])
        sim.plt.close("all")
        try:
            with patch.object(sim.plt, "show"):
                sim.plot_beam_preview(case)
            ax = sim.plt.gcf().axes[0]
            h = 1.0   # largest shaft radius, in inches
            self.assertLessEqual(ax.get_ylim()[1], 8 * h)
            self.assertGreaterEqual(ax.get_ylim()[0], -8 * h)
            text = "\n".join(t.get_text() for t in ax.texts)
            self.assertIn("D = 40 in", text)
            self.assertIn("mesh: top (+y)", text)
            self.assertIn("not to scale", ax.get_title())
            self.assertIn("Ø2", text)   # diameter labels moved below the shaft
            # Gear blanks stay within a few shaft radii, larger gear drawn taller.
            blanks = [p for p in ax.patches if isinstance(p, sim.plt.Rectangle)
                      and p.get_alpha() == 0.25]
            heights = sorted(p.get_height() for p in blanks)
            self.assertEqual(len(heights), 2)
            self.assertLess(heights[0], heights[1])
            self.assertLessEqual(heights[1], 7 * h)
        finally:
            sim.plt.close("all")


class SolverStationTests(unittest.TestCase):
    def test_cumulative_trapezoid_matches_running_trapezoid(self):
        x = np.sort(np.random.default_rng(1).uniform(0, 3, 50))
        y = np.sin(x) * x
        expected = [np.trapezoid(y[:i + 1], x[:i + 1]) for i in range(len(x))]
        np.testing.assert_allclose(sim.cumulative_trapezoid(y, x), expected, atol=1e-14)

    def test_vectorized_internal_loads_match_pointwise_singularity_sum(self):
        case = load_text(case_text())
        x, results = solve(case)
        loads = sim.build_load_list(case["loads"], case["supports"])
        for i in (0, 17, len(x) // 2, len(x) - 1):
            m_z = sum(l["force"][1] * sim.S(x[i], l["position"], 1)
                      - l["moment"][2] * sim.S(x[i], l["position"], 0) for l in loads)
            torque = sum(l["moment"][0] * sim.S(x[i], l["position"], 0) for l in loads)
            self.assertAlmostEqual(results["M_z"][i], m_z)
            self.assertAlmostEqual(results["T"][i], torque)
        self.assertIsInstance(sim.S(1.0, 0.5, 1), float)

    def test_shoulders_are_stations_and_use_the_smaller_section(self):
        case = load_text(case_text())
        sim.prepare_geometry(case["geometry"])
        shoulder = 10 * 0.0254
        x = sim.build_stations(case["length"], case["simulated_points"], case["geometry"])
        self.assertIn(shoulder, x)
        self.assertEqual(len(x), case["simulated_points"] + 1)
        self.assertAlmostEqual(sim.section_at(shoulder, case["geometry"])["diameter"], 0.0254)
        # Reversed step: the smaller segment is now on the left of the boundary.
        flipped = [dict(case["geometry"][0], diameter=0.0508), dict(case["geometry"][1], diameter=0.0254)]
        self.assertAlmostEqual(sim.section_at(shoulder, flipped)["diameter"], 0.0254)
        with self.assertRaisesRegex(ValueError, "outside the beam geometry"):
            sim.section_at(-0.1, case["geometry"])


class FatigueTests(unittest.TestCase):
    PSI = 6894.757293168361

    def fatigue(self, system="Imperial", criterion="goodman"):
        case = load_text(fatigue_case_text(system, criterion))
        x, results = solve(case)
        return case, x, results, sim.compute_fatigue(x, results, case["geometry"], case)

    def test_kt_fits_match_peterson_chart_readings(self):
        # (D/d, r/d, bending Kt, torsion Kts) read from Shigley Figs A-15-9 / A-15-8.
        for D_d, r_d, kt, kts in ((1.5, 0.1, 1.68, 1.42), (2.0, 0.05, 2.1, 1.75), (1.2, 0.1, 1.6, 1.35)):
            with self.subTest(D_d=D_d, r_d=r_d):
                self.assertAlmostEqual(sim.shoulder_kt(D_d, 1, r_d, sim.KT_BENDING), kt, delta=0.06 * kt)
                self.assertAlmostEqual(sim.shoulder_kt(D_d, 1, r_d, sim.KT_TORSION), kts, delta=0.06 * kts)
        self.assertEqual(sim.shoulder_kt(1.0, 1, 0.05, sim.KT_BENDING), 1.0)
        self.assertEqual(sim.shoulder_kt(1.0, 1, 0.05, sim.KT_TORSION), 1.0)

    def test_notch_sensitivity_and_marin_factors(self):
        # Shigley Fig 6-20: Sut = 100 kpsi, r = 0.1 in -> q ~ 0.84.
        self.assertAlmostEqual(sim.notch_sensitivity(0.1 * 0.0254, 100e3 * self.PSI), 0.84, delta=0.01)
        settings = {"surface_finish": "machined", "temperature_f": 500,
                    "reliability": 0.99, "endurance_limit": None}
        marin = sim.marin_factors(0.025, 690e6, settings)
        self.assertAlmostEqual(marin["ka"], 4.51 * 690**-0.265)
        self.assertAlmostEqual(float(marin["kb"]), 1.24 * 25**-0.107)
        self.assertAlmostEqual(marin["kd"], 0.995, delta=0.002)  # Shigley Table 6-4 (vs. its polynomial fit)
        self.assertAlmostEqual(marin["ke"], 0.814, places=3)    # Shigley Table 6-5
        self.assertAlmostEqual(marin["Se_prime"], 345e6)
        self.assertAlmostEqual(sim.marin_factors(0.025, 1600e6, settings)["Se_prime"], 700e6)
        self.assertAlmostEqual(float(sim.marin_factors(0.1, 690e6, settings)["kb"]), 1.51 * 100**-0.157)

    def test_criteria_limits_and_ordering(self):
        Se, Sut, Sy = 200e6, 600e6, 400e6
        for criterion in sim.FATIGUE_CRITERIA:
            with self.subTest(criterion=criterion):
                # Fully reversed only: every criterion reduces to Se / sigma_a.
                self.assertAlmostEqual(float(sim.fatigue_fos(50e6, 0, Se, Sut, Sy, criterion)), 4)
                self.assertEqual(float(sim.fatigue_fos(0, 0, Se, Sut, Sy, criterion)), np.inf)
        n = {c: float(sim.fatigue_fos(80e6, 150e6, Se, Sut, Sy, c)) for c in sim.FATIGUE_CRITERIA}
        self.assertAlmostEqual(n["goodman"], 1 / (80 / 200 + 150 / 600))
        self.assertAlmostEqual(n["soderberg"], 1 / (80 / 200 + 150 / 400))
        self.assertAlmostEqual(n["asme-elliptic"], 1 / np.hypot(80 / 200, 150 / 400))
        self.assertGreater(n["gerber"], n["goodman"])
        self.assertGreater(n["goodman"], n["soderberg"])
        gerber = n["gerber"]   # the operating point scaled by n lies on the Gerber parabola
        self.assertAlmostEqual(gerber * 80e6 / Se + (gerber * 150e6 / Sut)**2, 1)
        self.assertAlmostEqual(float(sim.fatigue_fos(0, 150e6, Se, Sut, Sy, "gerber")), 4)

    def test_fatigue_matches_hand_calculation_at_critical_and_shoulder(self):
        case, x, results, fatigue = self.fatigue()
        shoulder = int(np.flatnonzero(x == 10 * 0.0254)[0])
        raiser = fatigue["raisers"][0]
        self.assertFalse(raiser["r_assumed"])
        q = sim.notch_sensitivity(0.05 * 0.0254, 58e3 * self.PSI)
        kt = sim.shoulder_kt(2, 1, 0.05, sim.KT_BENDING)
        self.assertAlmostEqual(fatigue["Kf"][shoulder], 1 + q * (kt - 1))
        self.assertEqual(fatigue["Kf"][shoulder - 1], 1)
        for i in (fatigue["critical_idx"], shoulder):
            with self.subTest(station=i):
                d = fatigue["diameter"][i]
                M = np.hypot(results["M_y"][i], results["M_z"][i])
                sigma_a = fatigue["Kf"][i] * 32 * M / (np.pi * d**3)
                tau_m = fatigue["Kfs"][i] * 16 * abs(results["T"][i]) / (np.pi * d**3)
                sigma_m = fatigue["Kf"][i] * 4 * abs(results["N"][i]) / (np.pi * d**2)
                vm_a = np.hypot(sigma_a, np.sqrt(3) * 0.1 * tau_m)
                vm_m = np.hypot(sigma_m, np.sqrt(3) * tau_m)
                se = (4.51 * (58e3 * self.PSI / 1e6)**-0.265 * 1.24 * (d * 1e3)**-0.107
                      * (0.975 + 0.432e-3 * 300 - 0.115e-5 * 300**2 + 0.104e-8 * 300**3 - 0.595e-12 * 300**4)
                      * (1 - 0.08 * 2.3263478740408408) * 29e3 * self.PSI)
                self.assertAlmostEqual(fatigue["Se"][i] / se, 1)
                self.assertAlmostEqual(fatigue["fos_fatigue"][i] * (vm_a / se + vm_m / (58e3 * self.PSI)), 1)
                vm_max = np.hypot(sigma_a + sigma_m, np.sqrt(3) * 1.1 * tau_m)
                self.assertAlmostEqual(fatigue["fos_yield"][i] * vm_max / (36e3 * self.PSI), 1)

    def test_required_diameter_meets_target(self):
        case, x, results, fatigue = self.fatigue()
        target = case["target_factor_of_safety"]
        i = int(np.argmax(fatigue["d_required"]))
        d = np.array([fatigue["d_required"][i]])
        M = np.hypot(results["M_y"][i:i + 1], results["M_z"][i:i + 1])
        _, _, n_f, n_y = sim._fatigue_and_yield_fos(d, M, results["T"][i:i + 1], results["N"][i:i + 1],
                                                    fatigue["Kf"][i:i + 1], fatigue["Kfs"][i:i + 1], case)
        self.assertAlmostEqual(min(n_f[0], n_y[0]), target, places=6)

    def test_si_and_imperial_fatigue_agree(self):
        for criterion in sim.FATIGUE_CRITERIA:
            with self.subTest(criterion=criterion):
                *_, si = self.fatigue("SI", criterion)
                *_, imperial = self.fatigue("Imperial", criterion)
                for key in ("fos_fatigue", "fos_yield", "Se", "Kf", "Kfs", "d_required"):
                    np.testing.assert_allclose(si[key], imperial[key], rtol=1e-6)

    def test_every_criterion_is_evaluated_and_selected_one_governs(self):
        for selected in ("gerber", "asme-elliptic"):
            with self.subTest(selected=selected):
                case, x, results, fatigue = self.fatigue(criterion=selected)
                self.assertEqual(set(fatigue["fos_by_criterion"]), set(sim.FATIGUE_CRITERIA))
                np.testing.assert_array_equal(fatigue["fos_fatigue"], fatigue["fos_by_criterion"][selected])
                np.testing.assert_array_equal(fatigue["d_required"], fatigue["d_required_by_criterion"][selected])
                self.assertEqual(fatigue["critical_idx"], int(np.argmin(fatigue["fos_by_criterion"][selected])))
                for criterion, fos in fatigue["fos_by_criterion"].items():
                    np.testing.assert_allclose(fos, sim.fatigue_fos(
                        fatigue["vm_a"], fatigue["vm_m"], fatigue["Se"],
                        case["ultimate_strength"], case["yield_strength"], criterion))
        # With a mean stress present Gerber is the least and Soderberg the most conservative.
        loaded = fatigue["vm_m"] > 0
        by = fatigue["fos_by_criterion"]
        self.assertTrue(np.all(by["gerber"][loaded] >= by["goodman"][loaded]))
        self.assertTrue(np.all(by["goodman"][loaded] >= by["soderberg"][loaded]))
        self.assertTrue(np.all(by["asme-elliptic"][loaded] >= by["soderberg"][loaded]))
        required = fatigue["d_required_by_criterion"]
        self.assertTrue(np.all(required["gerber"] <= required["goodman"] * (1 + 1e-9)))
        self.assertTrue(np.all(required["goodman"] <= required["soderberg"] * (1 + 1e-9)))

    def test_report_includes_fatigue_only_when_enabled(self):
        output = io.StringIO()
        with redirect_stdout(output):
            sim.run_case(load_text(fatigue_case_text()), show_plots=False)
        report = output.getvalue()
        for title in ("FATIGUE ANALYSIS - Modified Goodman", "ENDURANCE LIMIT", "STRESS RAISERS",
                      "FATIGUE CRITICAL SECTION", "FATIGUE DIAMETER SIZING"):
            self.assertIn(title, report)
        self.assertRegex(report, r"Temperature factor kd\s+[\d.]+\s+- \(at 300 F\)")
        self.assertIn("FATIGUE CRITERIA COMPARISON", report)
        for name in sim.FATIGUE_CRITERIA.values():
            self.assertRegex(report, rf"\n  {re.escape(name)}(?: \(selected\))?\s+[\d.]+\s+[\d.]+")
        self.assertIn("Modified Goodman (selected)", report)
        output = io.StringIO()
        with redirect_stdout(output):
            sim.run_case(load_text(case_text()), show_plots=False)
        self.assertNotIn("FATIGUE", output.getvalue())

    def test_assumed_sharp_fillet_and_overrides(self):
        text = fatigue_case_text().replace("fillet_radius = 0.05\n", "")
        case = load_text(text)
        sim.prepare_geometry(case["geometry"])
        raiser = sim.stress_raisers(case["geometry"], case["ultimate_strength"])[0]
        self.assertTrue(raiser["r_assumed"])
        self.assertAlmostEqual(raiser["r"], 0.02 * 0.0254)
        case = load_text(fatigue_case_text().replace("fillet_radius = 0.05\n", "kf = 2.2\nkfs = 1.8\n"))
        raiser = sim.stress_raisers(case["geometry"], case["ultimate_strength"])[0]
        self.assertEqual((raiser["Kf"], raiser["Kfs"], raiser["override"]), (2.2, 1.8, True))

    def test_fillet_outside_kt_fit_range_is_warned(self):
        # 0.63 in on a 1 in shaft (r/d = 0.63): a likely in-vs-mm slip.
        for fillet, extra, warned in (("0.05", "", False), ("0.63", "", True), ("0.005", "", True),
                                      ("0.63", "kf = 2.0\nkfs = 1.6\n", False)):
            with self.subTest(fillet=fillet, override=bool(extra)):
                text = fatigue_case_text().replace("fillet_radius = 0.05\n", f"fillet_radius = {fillet}\n{extra}")
                case = load_text(text)
                sim.prepare_geometry(case["geometry"])
                raiser = sim.stress_raisers(case["geometry"], case["ultimate_strength"])[0]
                self.assertEqual(bool(raiser["warnings"]), warned)
                output = io.StringIO()
                with redirect_stdout(output):
                    sim.run_case(load_text(text), show_plots=False)
                report = output.getvalue()
                self.assertEqual("WARNING at x = 10 in: r/d" in report, warned)
                self.assertEqual("WARNING: r/d out of range" in report, warned)

    def test_fatigue_input_validation(self):
        good = fatigue_case_text()
        cases = {
            "ultimate_strength": (KeyError, good.replace("ultimate_strength = 58000.0\n", "")),
            "criterion must be": (ValueError, good.replace('"goodman"', '"walker"')),
            "surface_finish must be": (ValueError, good.replace('"machined"', '"polished"')),
            "reliability": (ValueError, good.replace("reliability = 0.99", "reliability = 1.0")),
            "unknown key": (KeyError, good.replace("[fatigue]\n", "[fatigue]\nlife = 1e6\n")),
            "at least yield_strength": (ValueError, good.replace("ultimate_strength = 58000.0", "ultimate_strength = 30000.0")),
            "first segment": (ValueError, good.replace("start = 0.0\n", "start = 0.0\nkf = 2.0\n")),
            "temperature": (ValueError, good.replace("temperature = 300", "temperature = 1200")),
        }
        for message, (error, text) in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                load_text(text)

    def test_fatigue_requires_shoulder_stations(self):
        case = load_text(fatigue_case_text())
        sim.prepare_geometry(case["geometry"])
        x = np.linspace(0, case["length"], 40)
        results = sim.compute_internal_loads(x, sim.build_load_list(case["loads"], case["supports"]))
        with self.assertRaisesRegex(ValueError, "build_stations"):
            sim.compute_fatigue(x, results, case["geometry"], case)

    def test_fatigue_diagram_keeps_operating_point_visible(self):
        sim.plt.switch_backend("Agg")
        case, x, results, fatigue = self.fatigue()
        i = fatigue["critical_idx"]
        units = sim.ReportUnits("Imperial")
        Sut, Sy = case["ultimate_strength"], case["yield_strength"]
        # How far the point is beyond the envelope axes, on its binding axis.
        reach = max(fatigue["vm_m"][i] / (1.05 * max(Sut, Sy)),
                    fatigue["vm_a"][i] / (1.05 * max(fatigue["Se"][i], Sy)))
        self.assertLess(reach, 1)
        for scale_up, off_scale in ((1, False), (2 / reach, False), (30 / reach, True)):
            with self.subTest(scale_up=scale_up):
                data = dict(fatigue, vm_a=fatigue["vm_a"] * scale_up, vm_m=fatigue["vm_m"] * scale_up)
                sim.plt.close("all")
                try:
                    with patch.object(sim.plt, "show"):
                        sim.plot_fatigue_diagram(x, data, case)
                    ax = sim.plt.gcf().axes[0]
                    px, py = data["vm_m"][i] / units.scales["stress"], data["vm_a"][i] / units.scales["stress"]
                    labels = "\n".join(line.get_label() for line in ax.lines)
                    self.assertEqual("off scale" in labels, off_scale)
                    if off_scale:
                        # The load line ends inside the axes, pointing at the real point.
                        load_line = next(line for line in ax.lines if line.get_label() == "load line")
                        end = load_line.get_xydata()[-1]
                        self.assertLess(end[0], ax.get_xlim()[1])
                        self.assertLess(end[1], ax.get_ylim()[1])
                        self.assertAlmostEqual(end[1] / end[0], py / px)
                    else:
                        self.assertLessEqual(px, ax.get_xlim()[1])
                        self.assertLessEqual(py, ax.get_ylim()[1])
                finally:
                    sim.plt.close("all")

    def test_fatigue_diagram_marks_target_fos_point_with_required_diameter(self):
        sim.plt.switch_backend("Agg")
        units = sim.ReportUnits("Imperial")
        for criterion in sim.FATIGUE_CRITERIA:
            with self.subTest(criterion=criterion):
                case, x, results, fatigue = self.fatigue(criterion=criterion)
                i = fatigue["critical_idx"]
                d_req = fatigue["d_required"][i]
                sim.plt.close("all")
                try:
                    with patch.object(sim.plt, "show"):
                        sim.plot_fatigue_diagram(x, fatigue, case)
                    ax = sim.plt.gcf().axes[0]
                    star = next(line for line in ax.lines if line.get_marker() == "*")
                    self.assertIn(f"required d = {units.number(d_req, 'diameter')} in", star.get_label())
                    self.assertTrue(any(f"required d = {units.number(d_req, 'diameter')} in" in t.get_text()
                                        for t in ax.texts))
                    # At the required diameter the governing FOS (fatigue or Langer) equals the target.
                    vm_m, vm_a = (v * units.scales["stress"] for v in star.get_xydata()[0])
                    Se = float(sim.marin_factors(d_req, case["ultimate_strength"], case["fatigue"])["Se"])
                    n_f = float(sim.fatigue_fos(vm_a, vm_m, Se, case["ultimate_strength"],
                                                case["yield_strength"], criterion))
                    state = sim.fatigue_stress_state(d_req, fatigue["M"][i], fatigue["T"][i], fatigue["N"][i],
                                                     fatigue["Kf"][i], fatigue["Kfs"][i], 0.1)
                    n_y = case["yield_strength"] / state["vm_max"]
                    self.assertAlmostEqual(min(n_f, n_y), case["target_factor_of_safety"], places=5)
                    # When fatigue governs, the star sits on the drawn design line.
                    design = next(line for line in ax.lines if "design line" in line.get_label())
                    if n_f < n_y:
                        sx, sy = star.get_xydata()[0]
                        line_x, line_y = design.get_xydata().T
                        self.assertAlmostEqual(np.interp(sx, line_x, line_y) / sy, 1, places=3)
                finally:
                    sim.plt.close("all")

    def test_fatigue_plots_render_in_selected_units(self):
        sim.plt.switch_backend("Agg")
        for system in ("SI", "Imperial"):
            with self.subTest(system=system):
                case, x, results, fatigue = self.fatigue(system)
                units = sim.ReportUnits(system)
                static = np.full(len(x), 5.0)
                sim.plt.close("all")
                try:
                    with patch.object(sim.plt, "show"):
                        sim.plot_fatigue_fos(x, static, fatigue, case)
                        ax = sim.plt.gcf().axes[0]
                        np.testing.assert_allclose(ax.lines[0].get_xdata(), x / units.scales["length"])
                        sim.plot_fatigue_diagram(x, fatigue, case)
                        ax = sim.plt.gcf().axes[0]
                        self.assertIn(units.labels["stress"], ax.get_xlabel())
                        i = fatigue["critical_idx"]
                        # Four criterion envelopes, then the Langer line, then the operating point.
                        self.assertEqual(sum("(selected)" in line.get_label() for line in ax.lines), 1)
                        point = ax.lines[len(sim.FATIGUE_CRITERIA) + 1]
                        self.assertAlmostEqual(point.get_ydata()[0], fatigue["vm_a"][i] / units.scales["stress"])
                        for number in sim.plt.get_fignums():
                            sim.plt.figure(number).canvas.draw()
                finally:
                    sim.plt.close("all")


if __name__ == "__main__":
    unittest.main()
