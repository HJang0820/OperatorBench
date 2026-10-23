import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import normalize_coefficient, unscale_solution
from operatorbench.data.model_inputs import (
    make_deeponet_branch, make_deeponet_coords, make_fno_input,
)
from operatorbench.inference import predict_batches, physical_solution_errors
from operatorbench.models.deeponet import DeepONet2d
from operatorbench.models.fno import FNO2d


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_best(path, model_name):
    # Only load your own trusted checkpoints; never disable weights_only here.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("schema_version") != 1
            or saved.get("checkpoint_kind") != "best_for_evaluation"):
        raise ValueError(f"Unsupported best checkpoint: {path}")
    info = saved["run_info"]
    if info["model_name"] != model_name or info["purpose"] != "baseline_training":
        raise ValueError(f"Wrong model or a smoke-test checkpoint: {path}")
    if not 1 <= saved["best_step"] <= saved["training_config"]["num_steps"]:
        raise ValueError("Invalid best_step in checkpoint.")
    return saved


def check_compatibility(saved, data, metadata, dataset_hash, root, model_key):
    info = saved["run_info"]
    if info["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset SHA-256 differs from the training dataset.")
    if info["dataset_metadata"] != metadata:
        raise ValueError("Dataset metadata differs from the training dataset.")
    if info["input_convention"] != "a_to_u; ij/C-order; DON trunk [-1,1]; FNO xy [0,1]":
        raise ValueError("Unsupported input convention.")
    for axis in ("x", "y"):
        if not np.array_equal(info[axis].cpu().numpy(), data[axis]):
            raise ValueError(f"Coordinate mismatch: {axis}")
    for split in ("train", "val", "test"):
        if len(data[f"A_{split}"]) != info["settings"]["split_sizes"][split]:
            raise ValueError(f"Sample-count mismatch: {split}")
    stats = info["normalization"]
    if (not np.isfinite([stats[k] for k in ("a_mean", "a_std", "u_scale")]).all()
            or stats["a_std"] <= 0 or stats["u_scale"] <= 0):
        raise ValueError("Invalid saved normalization statistics.")

    # Training/resume code may differ between runs; inference code must match.
    relevant = (
        "operatorbench/data/datasets.py",
        "operatorbench/data/preprocessing.py",
        "operatorbench/data/model_inputs.py",
        "operatorbench/losses.py",
        f"operatorbench/models/{model_key}.py",
    )
    saved_hashes = {k.replace("\\", "/"): v for k, v in info["code_sha256"].items()}
    for name in relevant:
        if file_sha256(root / name) != saved_hashes.get(name):
            raise ValueError(f"Inference source differs from training: {name}")


def predict_checkpoint(saved, data, model_key, device, batch_size):
    info = saved["run_info"]
    # Reuse the stored training statistics; do not fit anything on the test set.
    stats = {key: np.float32(value) for key, value in info["normalization"].items()}
    normalized_a = normalize_coefficient(data["A_test"], stats)
    coords01, trunk = make_deeponet_coords(data["x"], data["y"])
    if model_key == "deeponet":
        model = DeepONet2d(
            coords01=torch.from_numpy(coords01), **info["model_config"]
        )
        inputs = make_deeponet_branch(normalized_a)
        shared = (torch.from_numpy(trunk),)
        expected_shape = (len(inputs), len(data["x"]) * len(data["y"]))
    else:
        model = FNO2d(**info["model_config"])
        inputs = make_fno_input(normalized_a, data["x"], data["y"])
        shared = ()
        expected_shape = (*data["U_test"].shape, 1)

    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device)
    predicted = predict_batches(
        model, torch.from_numpy(inputs), shared_inputs=shared,
        batch_size=batch_size, device=device,
    )
    if tuple(predicted.shape) != expected_shape:
        raise ValueError(f"Unexpected model output shape: {tuple(predicted.shape)}")
    scaled = predicted.numpy().reshape(data["U_test"].shape, order="C")
    return unscale_solution(scaled, stats)


def save_sample_figure(data, predictions, sample_id, path):
    reference = data["U_test"][sample_id]
    don = predictions["deeponet"][sample_id]
    fno = predictions["fno"][sample_id]
    err_don = np.abs(don - reference)
    err_fno = np.abs(fno - reference)
    lower = min(float(field.min()) for field in (reference, don, fno))
    upper = max(float(field.max()) for field in (reference, don, fno))
    if upper <= lower:
        upper = lower + 1e-12
    error_upper = max(float(err_don.max()), float(err_fno.max()), 1e-12)
    panels = [
        (reference, "FDM reference", "viridis", lower, upper),
        (don, "DeepONet prediction", "viridis", lower, upper),
        (fno, "FNO prediction", "viridis", lower, upper),
        (data["A_test"][sample_id], "Coefficient a(x, y)", "viridis", 3, 12),
        (err_don, "DeepONet absolute error", "magma", 0, error_upper),
        (err_fno, "FNO absolute error", "magma", 0, error_upper),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        # ij-indexed arrays need a transpose for horizontal x and vertical y.
        image = ax.imshow(
            field.T, origin="lower", extent=(0, 1, 0, 1),
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(image, ax=ax)
    fig.suptitle(f"Darcy test sample {sample_id} | physical solution scale")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate two fixed Darcy checkpoints")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--deeponet-checkpoint", type=Path, default=(
        root / "outputs/deeponet_full_seed2026_01/best.pt"
    ))
    parser.add_argument("--fno-checkpoint", type=Path, default=(
        root / "outputs/fno_full_seed2026_resume_01/best.pt"
    ))
    parser.add_argument("--output-dir", type=Path, default=(
        root / "outputs/darcy_test_seed2026_01"
    ))
    parser.add_argument(
    "--sample-indices",
    type=int,
    nargs="+",
    default=None,
    help="Zero-based sample indices to visualize; all test samples are evaluated.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("Batch size and CPU thread count must be positive.")
    if args.output_dir.exists():
        raise FileExistsError("Output directory exists; choose a new --output-dir.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    torch.set_num_threads(args.cpu_threads)

    paths = {"deeponet": args.deeponet_checkpoint, "fno": args.fno_checkpoint}
    checkpoints = {
        "deeponet": load_best(paths["deeponet"], "DeepONet"),
        "fno": load_best(paths["fno"], "FNO"),
    }
    first = checkpoints["deeponet"]["run_info"]
    second = checkpoints["fno"]["run_info"]
    if first["normalization"] != second["normalization"]:
        raise ValueError("The two runs do not share the same normalization statistics.")
    for key in ("train_indices", "validation_indices"):
        if first[key] != second[key]:
            raise ValueError(f"The two runs use different {key}.")
    dataset_path = (args.dataset or (root / first["settings"]["dataset"])).resolve()
    dataset_hash = file_sha256(dataset_path)
    data, metadata = load_darcy_dataset(dataset_path)
    if (metadata.get("task") != "a_to_u" or metadata.get("mode") != "full"
            or metadata.get("source") != {"kind": "constant", "value": 1.0}):
        raise ValueError("This evaluator supports the full a_to_u, f=1 baseline.")
    if len(data["U_test"]) != 200:
        raise ValueError("This baseline comparison expects exactly 200 test samples.")
    if (not np.isin(data["A_test"], [3, 12]).all()
            or not np.all(data["F_test"] == 1)):
        raise ValueError("Unexpected test coefficient or source values.")
    for key, saved in checkpoints.items():
        check_compatibility(saved, data, metadata, dataset_hash, root, key)
    print("Checkpoint and dataset compatibility checks passed.", flush=True)
    print("Test samples:", len(data["U_test"]), "| Device:", device, flush=True)

    n_test = len(data["U_test"])
    if args.sample_indices is None:
        figure_indices = list(range(min(3, n_test)))
    else:
        figure_indices = args.sample_indices 
    if any(i < 0 or i >= n_test for i in figure_indices):
        parser.error(
        f"Sample indices must be between 0 and {n_test - 1}."
    )
    if len(set(figure_indices)) != len(figure_indices):
        parser.error("Sample indices must not contain duplicates.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions, errors, rows, provenance = {}, {}, [], {}
    for key, saved in checkpoints.items():
        info = saved["run_info"]
        print(f"Evaluating {info['model_name']} ...", flush=True)
        predictions[key] = predict_checkpoint(saved, data, key, device, args.batch_size)
        errors[key] = physical_solution_errors(predictions[key], data["U_test"])
        relative = errors[key]["relative_l2"]
        maximum = errors[key]["max_absolute_error"]
        row = {
            "model": info["model_name"], "n_test": len(relative),
            "best_step": saved["best_step"],
            "validation_relative_l2": saved["best_val_relative_l2"],
            "test_relative_l2_mean": float(relative.mean()),
            "test_relative_l2_std_across_samples": float(relative.std(ddof=1)),
            "test_relative_l2_mean_percent": float(100 * relative.mean()),
            "test_max_absolute_error_mean": float(maximum.mean()),
            "test_max_absolute_error_worst": float(maximum.max()),
        }
        rows.append(row)
        provenance[key] = {
            "checkpoint": str(paths[key].resolve()),
            "checkpoint_sha256": file_sha256(paths[key]),
            "model_config": info["model_config"],
            "normalization": info["normalization"],
            "training_config": saved["training_config"],
            "model_init_seed": info["model_init_seed"],
            "training_environment": info["environment"],
            "parameter_real_components": info.get("parameter_real_components"),
        }
        print(f"{info['model_name']}: test mean relative L2 = "
              f"{row['test_relative_l2_mean_percent']:.4f}%", flush=True)

    with (args.output_dir / "comparison.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "per_sample.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_index", "deeponet_relative_l2", "fno_relative_l2",
                         "deeponet_max_absolute_error", "fno_max_absolute_error"])
        for index in range(len(data["U_test"])):
            writer.writerow([index, errors["deeponet"]["relative_l2"][index],
                             errors["fno"]["relative_l2"][index],
                             errors["deeponet"]["max_absolute_error"][index],
                             errors["fno"]["max_absolute_error"][index]])
    np.savez_compressed(
        args.output_dir / "predictions.npz", test_indices=np.arange(len(data["U_test"])),
        x=data["x"], y=data["y"], A_test=data["A_test"], F_test=data["F_test"],
        U_reference=data["U_test"], U_deeponet=predictions["deeponet"], U_fno=predictions["fno"],
    )
    # Preselect examples by index, not by their prediction errors.
    for sample_id in figure_indices:
        save_sample_figure(data, predictions, sample_id,
                           args.output_dir / f"sample_{sample_id:03d}.png")
    summary = {
        "completed": True, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
        "split": "test", "grid": [len(data["x"]), len(data["y"])],
        "metric_domain": "all grid points, including boundaries; uniform discrete norms",
        "solution_scale": "physical", "relative_l2_denominator_epsilon": 1e-12,
        "std_note": "Variation across test samples, not across repeated training seeds.",
        "figure_sample_indices": figure_indices, "metrics": rows, "checkpoints": provenance,
        "evaluation_environment": {
            "python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "device": str(device),
            "cpu_threads": torch.get_num_threads(), "batch_size": args.batch_size,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "evaluation_code_sha256": {
            "scripts/evaluate_darcy.py": file_sha256(Path(__file__)),
            "operatorbench/inference.py": file_sha256(root / "operatorbench/inference.py"),
        },
    }
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    print("Darcy test evaluation completed.", flush=True)
    print("Saved results:", args.output_dir.resolve())


if __name__ == "__main__":
    main()