import argparse
import hashlib
import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from operatorbench.data.darcy_adapter import validate_darcy_fixed_fields
from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import (
    fit_normalization, normalize_coefficient, scale_solution,
    fit_task_normalization, select_input_field, normalize_input,
)
from operatorbench.data.model_inputs import (
    make_deeponet_branch, make_deeponet_target, make_deeponet_coords,
    make_fno_input, make_fno_target,
    prepare_model_arrays,
)
from operatorbench.models.deeponet import DeepONet2d
from operatorbench.models.fno import FNO2d
from operatorbench.training import atomic_json_save, train_model


def file_sha256(path):
    # Compute a content hash to identify identical files.
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(data, stats, split, model_key, device, task="a_to_u"):
    # Use the entire split rather than a small prefix of samples.
    X_raw = select_input_field(data, task, split)
    X = (
        normalize_coefficient(X_raw, stats)
        if task == "a_to_u" else normalize_input(X_raw, stats)
    )
    U = scale_solution(data[f"U_{split}"], stats)

    inputs, target = prepare_model_arrays(X, U, data["x"], data["y"], model_key)

    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(target).to(device),
    )


def validate_dataset(data, metadata, settings):
    """Check the declared task, reference method, splits and physical inputs."""
    task = settings.get("task", "a_to_u")
    if task not in ("a_to_u", "f_to_u"):
        raise ValueError(f"Unsupported task: {task!r}.")
    source = settings.get("source", {"kind": "constant", "value": 1.0})
    reference = settings.get("reference", {"kind": "five_point_fd"})
    if reference not in ({"kind": "five_point_fd"}, {"kind": "analytic_sine"}):
        raise ValueError("Unsupported reference method in the training configuration.")
    expected_solver = {"method": reference["kind"], "compute_dtype": "float64"}
    if reference["kind"] == "five_point_fd":
        expected_solver["face_average"] = "arithmetic"
    if (metadata.get("mode") not in ("small", "full") or metadata.get("task") != task
            or metadata.get("N") != settings["N"]
            or metadata.get("source") != source
            or metadata.get("solver") != expected_solver
            or metadata.get("boundary") != "homogeneous_dirichlet"):
        raise ValueError("Dataset task, grid, source, reference, or boundary does not match the configuration.")

    if task == "a_to_u":
        expected_coefficient = {
            "generator": "sample_fno_darcy_coefficients",
            "n_modes": settings["N"], "values": [3.0, 12.0],
        }
        if source.get("kind") != "constant":
            raise ValueError("The current a_to_u adapter requires a constant source.")
    else:
        expected_coefficient = settings.get("coefficient")
        if (not isinstance(expected_coefficient, dict)
                or set(expected_coefficient) != {"kind", "value"}
                or expected_coefficient["kind"] != "constant"):
            raise ValueError("The current f_to_u adapter requires a positive constant coefficient.")
        a0 = expected_coefficient["value"]
        if type(a0) not in (int, float) or not np.isfinite(a0) or a0 <= 0:
            raise ValueError("The fixed coefficient must be finite and positive.")
        if source.get("kind") != "random_sine":
            raise ValueError("Unsupported f_to_u source distribution.")
    if metadata.get("coefficient") != expected_coefficient:
        raise ValueError("Dataset coefficient metadata does not match the configuration.")

    for split in ("train", "val", "test"):
        if data[f"A_{split}"].shape[0] != settings["split_sizes"][split]:
            raise ValueError(f"The {split} sample count does not match the configuration.")
        if metadata["splits"][split]["seed"] != settings["split_seeds"][split]:
            raise ValueError(f"The {split} generation seed does not match the configuration.")
    # Training consumes only train/validation fields, never test targets.
    if task == "a_to_u":
        for split in ("train", "val"):
            if not np.isin(data[f"A_{split}"], [3.0, 12.0]).all():
                raise ValueError("The binary-coefficient adapter requires values 3 and 12.")
    validate_darcy_fixed_fields(data, metadata, splits=("train", "val"))
    return task


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a Darcy baseline model")
    parser.add_argument("--model", choices=("deeponet", "fno"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--config", type=Path,
        default=project_root / "configs" / "darcy_baseline.json",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last.pt in the same run directory.",
    )
    parser.add_argument(
        "--stop-after", type=int,
        help="Save and exit at this absolute step number; useful for resume checks.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a two-step integration check using the full dataset splits.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Run names may contain only letters, digits, underscores, and hyphens.")
    run_dir = project_root / "outputs" / args.run_name
    if args.resume and not (run_dir / "last.pt").is_file():
        raise FileNotFoundError(f"No last.pt checkpoint to resume in: {run_dir}")
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the current Python environment. "
            "Check your environment or explicitly use --device cpu."
        )

    with args.config.open(encoding="utf-8") as file:
        settings = json.load(file)
    config = dict(settings["training"])
    if args.smoke:
        config["num_steps"] = 2
        config["validate_every"] = 1

    torch.set_num_threads(settings["cpu_threads"])
    seed = settings["model_init_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset_path = (
        args.dataset if args.dataset is not None
        else project_root / settings["dataset"]
    ).resolve()
    data, metadata = load_darcy_dataset(dataset_path)

    task = validate_dataset(data, metadata, settings)

    # Fit normalization once using the entire training split only.
    stats = (
        fit_normalization(data["A_train"], data["U_train"])
        if task == "a_to_u" else fit_task_normalization(data, task)
    )
    train_x, train_y = prepare_split(data, stats, "train", args.model, device, task)
    val_x, val_y = prepare_split(data, stats, "val", args.model, device, task)

    coords01, trunk = make_deeponet_coords(data["x"], data["y"])
    model_config = dict(settings["models"][args.model])
    if args.model == "deeponet":
        model_config.setdefault("n_sensors", train_x.shape[1])
        if model_config["n_sensors"] != train_x.shape[1]:
            raise ValueError("DeepONet n_sensors must match the number of input grid values.")
        model_name = "DeepONet"
        model = DeepONet2d(
            coords01=torch.from_numpy(coords01), **model_config,
        ).to(device)
        shared = (torch.from_numpy(trunk).to(device),)
    else:
        model_name = "FNO"
        N = settings["N"]
        if not (1 <= model_config["modes1"] <= N // 2
                and 1 <= model_config["modes2"] <= N // 2 + 1):
            raise ValueError("FNO modes must fit the selected grid; reduce the mode counts or increase N.")
        model = FNO2d(**model_config).to(device)
        shared = ()

    code_files = [Path(__file__), *sorted(
        (project_root / "operatorbench").rglob("*.py")
    )]
    run_info = {
        "purpose": "full_data_smoke_test" if args.smoke else "baseline_training",
        "model_name": model_name,
        "task": task,
        "input_field": "A" if task == "a_to_u" else "F",
        "model_config": model_config,
        "model_init_seed": seed,
        "normalization": {key: float(value) for key, value in stats.items()},
        "x": torch.from_numpy(data["x"].copy()),
        "y": torch.from_numpy(data["y"].copy()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_metadata": metadata,
        "train_indices": list(range(train_x.shape[0])),
        "validation_indices": list(range(val_x.shape[0])),
        "input_convention": f"{task}; ij/C-order; DON trunk [-1,1]; FNO xy [0,1]",
        "settings": settings,
        "code_sha256": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in code_files
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
        },
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "parameter_real_components": sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in model.parameters()
        ),
    }

    print("Model:", model_name, "| Device:", device)
    print("Task:", task, "| Input field:", run_info["input_field"])
    print("Reference method:", metadata["solver"]["method"])
    print("Training input and target shapes:", tuple(train_x.shape), tuple(train_y.shape))
    print("Validation input and target shapes:", tuple(val_x.shape), tuple(val_y.shape))
    print("Total optimization steps:", config["num_steps"])
    print("Dataset SHA-256:", run_info["dataset_sha256"])
    print("Run directory:", run_dir, flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train_model(
        model, train_x, train_y, val_x, val_y,
        shared_inputs=shared, config=config,
        run_info=run_info, run_dir=run_dir,
        resume=args.resume, stop_after=args.stop_after,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    summary = {
        "purpose": run_info["purpose"],
        "model_name": model_name,
        "task": task,
        "reference_method": metadata["solver"]["method"],
        "completed_steps": len(history["train"]),
        "completed": history["completed"],
        "resume_count": history["resume_count"],
        "best_step": history["best_step"],
        "best_val_relative_l2": history["best_val_relative_l2"],
        "training_validation_checkpoint_wall_seconds": history[
            "recorded_training_seconds"
        ],
        "current_session_wall_seconds": elapsed,
        "timing_note": (
            "Cumulative time based on saved progress. Excludes the previous "
            "session's final checkpoint write, lost work after that checkpoint, "
            "reload time, and downtime."
        ),
        "dataset_sha256": run_info["dataset_sha256"],
        "environment": run_info["environment"],
    }
    atomic_json_save(summary, run_dir / "summary.json")

    if not history["completed"]:
        print("Completed steps at the requested stopping point:", history["completed_steps"])
        print("Resume checkpoint:", run_dir / "last.pt")
        print("Use the same run name with --resume and omit --stop-after to continue.")
        return
    print("Best validation step:", history["best_step"])
    print("Best validation relative L2 error:", history["best_val_relative_l2"])
    print("Training, validation, and saving time this session (seconds):", round(elapsed, 2))
    print("Cumulative training time based on saved progress (seconds):",
          round(history["recorded_training_seconds"], 2))
    print("Saved best model:", run_dir / "best.pt")
    print("Smoke test completed." if args.smoke else "Training completed.")


if __name__ == "__main__":
    main()
