import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from operatorbench.data.darcy_adapter import from_darcy_arrays
from operatorbench.data.schema import (load_operator_dataset, restore_grid_field,
                                      save_operator_dataset, select_split,
                                      validate_operator_dataset)


def darcy_fixture():
    data = {"x": np.linspace(0, 1, 3), "y": np.linspace(0, 1, 4)}
    for k, (split, count) in enumerate((("train", 2), ("val", 1), ("test", 1))):
        data[f"A_{split}"] = (np.arange(count * 12).reshape(count, 3, 4) + 1 + k * 100).astype("float32")
        data[f"F_{split}"] = np.ones((count, 3, 4), dtype="float32")
        u = np.zeros((count, 3, 4), dtype="float32")
        u[:, 1, 1:3] = np.array([2 + k, 7 + k])
        data[f"U_{split}"] = u
    meta = {"task": "a_to_u", "domain": [[0., 1.], [0., 1.]],
            "boundary": "homogeneous_dirichlet", "solver": {"method": "synthetic_layout_fixture_not_PDE"}}
    return data, from_darcy_arrays(data, meta, "layout-fixture")


class CommonSchemaTests(unittest.TestCase):
    def test_rectangular_grid_roundtrip_and_point_order(self):
        data, dataset = darcy_fixture()
        self.assertEqual(dataset["coordinates"]["grid_shape"], (3, 4))
        np.testing.assert_array_equal(dataset["coordinates"]["space"][6], [0.5, 2/3])
        for split in ("train", "val", "test"):
            for raw, name in (("A", "a"), ("F", "f"), ("U", "u")):
                values = select_split(dataset, split)[name]
                np.testing.assert_array_equal(values.reshape(data[f"{raw}_{split}"].shape), data[f"{raw}_{split}"])

    def test_save_load_preserves_every_field(self):
        _, dataset = darcy_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = save_operator_dataset(dataset, Path(folder)/"common.npz")
            loaded = load_operator_dataset(path)
            for name in dataset["fields"]:
                np.testing.assert_array_equal(loaded["fields"][name]["values"], dataset["fields"][name]["values"])
            with self.assertRaises(FileExistsError):
                save_operator_dataset(dataset, path)

    def test_fixed_field_must_be_fixed(self):
        _, dataset = darcy_fixture()
        dataset["fields"]["f"]["values"][1, 0, 1, 0] = 2
        with self.assertRaisesRegex(ValueError, "declared fixed"):
            validate_operator_dataset(dataset)

    def test_split_overlap_rejected(self):
        _, dataset = darcy_fixture()
        dataset["splits"]["test"] = dataset["splits"]["val"].copy()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_operator_dataset(dataset)

    def test_point_mismatch_rejected(self):
        _, dataset = darcy_fixture()
        dataset["coordinates"]["space"] = dataset["coordinates"]["space"][:-1]
        with self.assertRaises(ValueError):
            validate_operator_dataset(dataset)

    def test_unknown_input_rejected(self):
        _, dataset = darcy_fixture()
        dataset["task"]["inputs"] = ["missing"]
        with self.assertRaises(ValueError):
            validate_operator_dataset(dataset)

    def test_static_time_semantics_rejected(self):
        _, dataset = darcy_fixture()
        dataset["fields"]["u"]["values"] = np.repeat(dataset["fields"]["u"]["values"], 2, axis=1)
        with self.assertRaisesRegex(ValueError, "one time slot"):
            validate_operator_dataset(dataset)

    def test_nan_rejected(self):
        _, dataset = darcy_fixture()
        dataset["fields"]["u"]["values"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid field"):
            validate_operator_dataset(dataset)

    def test_unknown_version_rejected(self):
        _, dataset = darcy_fixture()
        dataset["schema_version"] = "99"
        with self.assertRaisesRegex(ValueError, "version"):
            validate_operator_dataset(dataset)

    def test_time_dependent_layout_not_burgers_solver(self):
        _, dataset = darcy_fixture()
        initial = dataset["fields"]["u"]
        initial["time"] = np.array([0.], dtype="float64")
        dataset["fields"] = {"u0": initial, "u": copy.deepcopy(initial)}
        dataset["fields"]["u"]["time"] = np.array([0., 0.1, 0.2])
        dataset["fields"]["u"]["values"] = np.repeat(initial["values"], 3, axis=1)
        dataset["task"] = {"name": "time_layout_fixture", "inputs": ["u0"], "targets": ["u"], "fixed_fields": []}
        dataset["conditions"]["initial_fields"] = ["u0"]
        validate_operator_dataset(dataset)
        with tempfile.TemporaryDirectory() as folder:
            loaded = load_operator_dataset(save_operator_dataset(dataset, Path(folder)/"time.npz"))
            np.testing.assert_array_equal(loaded["fields"]["u"]["time"], [0., 0.1, 0.2])
        dataset["fields"]["u"]["time"] = np.array([0., 0.2, 0.1])
        with self.assertRaisesRegex(ValueError, "time coordinates"):
            validate_operator_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
