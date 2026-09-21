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


if __name__ == "__main__":
    unittest.main()
