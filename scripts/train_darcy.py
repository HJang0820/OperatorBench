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
from operatorbench.training import atomic_json_save, train_model


def file_sha256(path):
    # Compute a content hash to identify identical files.
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(data, stats, split, model_key, device):
    # Use the entire split rather than a small prefix of samples.
    A = normalize_coefficient(data[f"A_{split}"], stats)
    U = scale_solution(data[f"U_{split}"], stats)

    if model_key == "deeponet":
        inputs = make_deeponet_branch(A)
        target = make_deeponet_target(U)
    else:
        inputs = make_fno_input(A, data["x"], data["y"])
        target = make_fno_target(U)

    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(target).to(device),
    )


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

    if (metadata.get("mode") != "full"
            or metadata.get("task") != "a_to_u"
            or metadata.get("N") != settings["N"]
            or metadata.get("source") != {"kind": "constant", "value": 1.0}):
        raise ValueError("Dataset task, grid, or source metadata does not match the configuration.")

    for split, count in settings["split_sizes"].items():
        if data[f"A_{split}"].shape[0] != count:
            raise ValueError(f"The {split} sample count does not match the configuration.")
        if metadata["splits"][split]["seed"] != settings["split_seeds"][split]:
            raise ValueError(f"The {split} generation seed does not match the configuration.")

    for split in ("train", "val"):
        if not np.isin(data[f"A_{split}"], [3.0, 12.0]).all():
            raise ValueError("This baseline requires coefficient values of 3 and 12 only.")
        if not np.all(data[f"F_{split}"] == 1.0):
            raise ValueError("This baseline requires a fixed source term of 1.")

    # Fit normalization once using the entire training split only.
    stats = fit_normalization(data["A_train"], data["U_train"])
    train_x, train_y = prepare_split(data, stats, "train", args.model, device)
    val_x, val_y = prepare_split(data, stats, "val", args.model, device)

    coords01, trunk = make_deeponet_coords(data["x"], data["y"])
    model_config = settings["models"][args.model]
    if args.model == "deeponet":
        model_name = "DeepONet"
        model = DeepONet2d(
            coords01=torch.from_numpy(coords01), **model_config,
        ).to(device)
        shared = (torch.from_numpy(trunk).to(device),)
    else:
        model_name = "FNO"
        model = FNO2d(**model_config).to(device)
        shared = ()

    code_files = [Path(__file__), *sorted(
        (project_root / "operatorbench").rglob("*.py")
    )]
    run_info = {
        "purpose": "full_data_smoke_test" if args.smoke else "baseline_training",
        "model_name": model_name,
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
        "input_convention": "a_to_u; ij/C-order; DON trunk [-1,1]; FNO xy [0,1]",
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