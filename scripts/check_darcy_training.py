import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import (
    fit_normalization, normalize_coefficient, scale_solution,
)
from operatorbench.data.model_inputs import (
    make_deeponet_branch, make_deeponet_target, make_deeponet_coords,
    make_fno_input, make_fno_target,
)
from operatorbench.models.deeponet import DeepONet2d
from operatorbench.models.fno import FNO2d
from operatorbench.evaluation import evaluate_relative_l2
from operatorbench.training import train_model


CONFIG = {
    "num_steps": 6,
    "batch_size": 2,
    "validate_every": 4,
    "eval_batch_size": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "scheduler_step_size": 5000,
    "scheduler_gamma": 0.5,
    "sampling_seed": 2027,
}


def prepare_split(data, stats, split, count, name):
    A = normalize_coefficient(data[f"A_{split}"][:count], stats)
    U = scale_solution(data[f"U_{split}"][:count], stats)

    if name == "DeepONet":
        inputs = make_deeponet_branch(A)
        target = make_deeponet_target(U)
    elif name == "FNO":
        inputs = make_fno_input(A, data["x"], data["y"])
        target = make_fno_target(U)
    else:
        raise ValueError(f"Unknown model: {name}")

    return torch.from_numpy(inputs), torch.from_numpy(target)


def build_model(name, model_config, coords01):
    if name == "DeepONet":
        return DeepONet2d(coords01=coords01, **model_config)
    if name == "FNO":
        return FNO2d(**model_config)
    raise ValueError(f"Unknown model: {name}")


def check_training(data, metadata, dataset_path, output_root):
    # This is a CPU integration check, not a benchmark run.
    torch.set_num_threads(2)
    stats = fit_normalization(data["A_train"], data["U_train"])
    n_train, n_val = 8, 5
    assert len(data["A_train"]) >= n_train
    assert len(data["A_val"]) >= n_val
    assert (len(data["x"]), len(data["y"])) == (85, 85)

    coords01, trunk = make_deeponet_coords(data["x"], data["y"])
    coords01_t = torch.from_numpy(coords01)
    trunk_t = torch.from_numpy(trunk)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(
        prefix="darcy_training_check_", dir=output_root,
    ))

    model_configs = {
        "DeepONet": {
            "n_sensors": 85 * 85,
            "latent_dim": 128,
            "branch_hidden": (400, 400, 400),
            "trunk_hidden": (128, 128, 128),
            "use_hard_dirichlet": False,
        },
        "FNO": {"modes1": 20, "modes2": 20, "width": 32},
    }

    for name, model_config in model_configs.items():
        torch.manual_seed(2026)
        model = build_model(name, model_config, coords01_t)
        shared = (trunk_t,) if name == "DeepONet" else ()

        train_x, train_y = prepare_split(
            data, stats, "train", n_train, name,
        )
        val_x, val_y = prepare_split(
            data, stats, "val", n_val, name,
        )

        run_info = {
            "purpose": "smoke_test",
            "model_name": name,
            "model_config": model_config,
            "model_init_seed": 2026,
            "normalization": {key: float(value) for key, value in stats.items()},
            "x": torch.from_numpy(data["x"].copy()),
            "y": torch.from_numpy(data["y"].copy()),
            "dataset_path": str(dataset_path),
            "dataset_metadata": metadata,
            "train_indices": list(range(n_train)),
            "validation_indices": list(range(n_val)),
            "input_convention": "a_to_u; ij/C-order; DON trunk [-1,1]; FNO xy [0,1]",
        }
        run_dir = run_root / name.lower()

        history = train_model(
            model, train_x, train_y, val_x, val_y,
            shared_inputs=shared, config=CONFIG,
            run_info=run_info, run_dir=run_dir,
        )

        # Check selection among all evaluated validation steps.
        records = history["validation"]
        assert [row["step"] for row in records] == [1, 4, 6]
        best_record = min(records, key=lambda row: row["relative_l2"])
        assert history["best_step"] == best_record["step"]

        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=True,
        )
        assert checkpoint["best_step"] == best_record["step"]
        assert checkpoint["best_val_relative_l2"] == best_record["relative_l2"]
        with (run_dir / "history.json").open(encoding="utf-8") as file:
            assert json.load(file) == history

        # Rebuild from saved configuration, coordinates and normalization.
        saved = checkpoint["run_info"]
        saved_stats = {
            key: np.float32(value)
            for key, value in saved["normalization"].items()
        }
        restored_data = {
            "A_val": data["A_val"], "U_val": data["U_val"],
            "x": saved["x"].numpy(), "y": saved["y"].numpy(),
        }
        restored_x, restored_y = prepare_split(
            restored_data, saved_stats, "val", n_val, name,
        )
        torch.testing.assert_close(restored_x, val_x, rtol=0, atol=0)
        torch.testing.assert_close(restored_y, val_y, rtol=0, atol=0)

        saved_coords, saved_trunk = make_deeponet_coords(
            restored_data["x"], restored_data["y"],
        )
        restored = build_model(
            saved["model_name"], saved["model_config"],
            torch.from_numpy(saved_coords),
        )
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        restored.eval()
        restored_shared = (
            (torch.from_numpy(saved_trunk),) if name == "DeepONet" else ()
        )

        with torch.no_grad():
            before = model(val_x, *shared)
            after = restored(restored_x, *restored_shared)
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)

        # Five validation samples are evaluated as batches of 2, 2 and 1.
        restored_errors = evaluate_relative_l2(
            restored, restored_x, restored_y,
            shared_inputs=restored_shared,
            batch_size=CONFIG["eval_batch_size"],
        )
        assert restored_errors.shape == (n_val,)
        assert math.isclose(
            restored_errors.mean().item(),
            checkpoint["best_val_relative_l2"],
            rel_tol=1e-6, abs_tol=1e-7,
        )
        print(f"{name}: best selection and checkpoint reload passed.")

    print("Saved check outputs:", run_root)
    print("Darcy training/validation/checkpoint checks passed.")
    return run_root


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "darcy" / (
        "darcy_binary_f1_v1_full_N85"
        "_train1000_val200_test200_seeds2022-2024-2025.npz"
    )
    data, metadata = load_darcy_dataset(dataset_path)
    assert metadata["mode"] == "full"
    assert metadata["task"] == "a_to_u"
    assert metadata["source"] == {"kind": "constant", "value": 1.0}

    check_training(
        data, metadata, dataset_path, project_root / "outputs",
    )


if __name__ == "__main__":
    main()