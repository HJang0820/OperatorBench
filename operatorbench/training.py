import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from operatorbench.evaluation import evaluate_relative_l2
from operatorbench.losses import relative_l2_per_sample


def atomic_torch_save(value, path):
    """Write a temporary file in the same directory, then replace the target."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def atomic_json_save(value, path):
    """Replace a JSON result only after the complete file has been written."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def cpu_copy(value):
    """Create independent CPU copies of model and optimizer state tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def same_value(left, right):
    """Compare nested settings and state objects, including tensors."""
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict) and left.keys() == right.keys()
            and all(same_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(same_value(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def capture_rng(generator, device):
    """Capture the batch-sampling generator and global random states."""
    numpy_state = np.random.get_state()
    return {
        "sampling": generator.get_state().cpu(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
        "python": random.getstate(),
        # Use built-in types instead of NumPy arrays for weights-only loading.
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(),
            int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]),
        ),
    }


def restore_rng(state, generator, device):
    generator.set_state(state["sampling"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("The GPU count differs from the saved environment.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((
        name, np.asarray(keys, dtype=np.uint32),
        position, has_gauss, cached_gauss,
    ))


def train_model(
    model, train_x, train_y, val_x, val_y, *,
    shared_inputs=(), config, run_info, run_dir,
    resume=False, stop_after=None,
):
    """Start a new run or continue after the last saved completed step."""
    config = dict(config)
    config.setdefault("checkpoint_every", config["validate_every"])
    for key in (
        "num_steps", "batch_size", "validate_every", "eval_batch_size",
        "scheduler_step_size", "checkpoint_every",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")

    for x, y in ((train_x, train_y), (val_x, val_y)):
        if x.shape[0] == 0 or x.shape[0] != y.shape[0]:
            raise ValueError("Inputs and targets must have matching, nonempty batches.")
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("Training and validation input shapes must match.")
    if train_y.shape[1:] != val_y.shape[1:]:
        raise ValueError("Training and validation target shapes must match.")
    device = train_x.device
    if any(t.device != device for t in (train_y, val_x, val_y, *shared_inputs)):
        raise ValueError("All inputs and targets must be on the same device.")

    run_dir = Path(run_dir)
    last_path = run_dir / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {last_path}. Do not substitute best.pt."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step_size"],
        gamma=config["scheduler_gamma"],
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config["sampling_seed"])

    history = {"train": [], "validation": []}
    best_value, best_step, best_state = float("inf"), None, None
    start_step, previous_seconds, resume_count = 0, 0.0, 0

    def save_best():
        atomic_torch_save({
            "schema_version": 1,
            "checkpoint_kind": "best_for_evaluation",
            "model_state_dict": best_state,
            "best_step": best_step,
            "best_val_relative_l2": best_value,
            "training_config": config,
            "run_info": run_info,
            "torch_version": str(torch.__version__),
        }, run_dir / "best.pt")

    if resume:
        # Load only checkpoints that you created and trust.
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if (saved.get("schema_version") != 2
                or saved.get("checkpoint_kind") != "last_for_resume"):
            raise ValueError("This file is not a supported resume checkpoint.")
        if not same_value(saved["training_config"], config):
            raise ValueError("Training settings differ from the saved checkpoint.")
        # Compare the dataset hash, normalization, model, code, and environment.
        for key in set(saved["run_info"]) | set(run_info):
            if key == "dataset_path":
                continue
            if not same_value(saved["run_info"].get(key), run_info.get(key)):
                raise ValueError(f"Run information differs from the checkpoint: {key}")
        if saved["device"] != str(device):
            raise ValueError("This resume implementation requires the same device.")

        start_step = saved["step"]
        history = saved["history"]
        if not 1 <= start_step <= config["num_steps"]:
            raise ValueError("The checkpoint contains an invalid completed step.")
        if len(history["train"]) != start_step:
            raise ValueError("The checkpoint step and training history length differ.")
        if history["train"][-1]["step"] != start_step:
            raise ValueError("The last history entry does not match the saved step.")

        model.load_state_dict(saved["model_state_dict"], strict=True)
        # Construct the scheduler first, then restore optimizer and scheduler states.
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        best_state = saved["best_model_state_dict"]
        best_step = saved["best_step"]
        best_value = saved["best_val_relative_l2"]
        previous_seconds = saved["recorded_training_seconds"]
        resume_count = saved["resume_count"] + 1
        restore_rng(saved["rng_state"], generator, device)
        # Treat last.pt as authoritative if an interrupted save left best.pt out of sync.
        save_best()
        print(f"Restored checkpoint after completed step {start_step}.", flush=True)

    end_step = config["num_steps"] if stop_after is None else stop_after
    if (not isinstance(end_step, int)
            or not start_step <= end_step <= config["num_steps"]):
        raise ValueError("The stopping step must be between the saved and total steps.")
    if stop_after is not None and end_step == start_step:
        raise ValueError("The stopping step must be greater than the saved step.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    segment_start = time.perf_counter()
    last_completed_step = start_step

    for step in range(start_step + 1, end_step + 1):
        indices = torch.randint(
            train_x.shape[0], (config["batch_size"],),
            generator=generator, device=device,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(train_x[indices], *shared_inputs)
        loss = relative_l2_per_sample(pred, train_y[indices]).mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at step {step}.")

        lr_used = float(optimizer.param_groups[0]["lr"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["train"].append({
            "step": step, "loss": float(loss.item()), "learning_rate": lr_used,
        })
        last_completed_step = step

        should_validate = (
            step == 1 or step % config["validate_every"] == 0
            or step == config["num_steps"]
        )
        improved = False
        if should_validate:
            errors = evaluate_relative_l2(
                model, val_x, val_y, shared_inputs=shared_inputs,
                batch_size=config["eval_batch_size"],
            )
            val_value = float(errors.mean().item())
            if not np.isfinite(val_value):
                raise FloatingPointError("The validation error is not finite.")
            history["validation"].append({
                "step": step, "relative_l2": val_value,
            })
            improved = val_value < best_value
            if improved:
                best_value, best_step = val_value, step
                best_state = cpu_copy(model.state_dict())

        # Save at a safe boundary after the update and any scheduled validation.
        should_save = (
            step == 1 or step % config["checkpoint_every"] == 0
            or step == end_step or improved
        )
        if should_save:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            snapshot = {
                "schema_version": 2,
                "checkpoint_kind": "last_for_resume",
                "step": step,
                "model_state_dict": cpu_copy(model.state_dict()),
                "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng(generator, device),
                "history": history,
                "best_model_state_dict": best_state,
                "best_step": best_step,
                "best_val_relative_l2": best_value,
                "training_config": config,
                "run_info": cpu_copy(run_info),
                "device": str(device),
                "resume_count": resume_count,
                "recorded_training_seconds": (
                    previous_seconds + time.perf_counter() - segment_start
                ),
            }
            atomic_torch_save(snapshot, last_path)
            save_best()

        if should_validate:
            print(
                f"{run_info['model_name']}: step {step:5d} | "
                f"train={loss.item():.6e} | val={val_value:.6e}"
                f"{' [best]' if improved else ''}",
                flush=True,
            )

    history.update({
        "best_step": best_step,
        "best_val_relative_l2": best_value,
        "completed_steps": last_completed_step,
        "completed": last_completed_step == config["num_steps"],
        "resume_count": resume_count,
    })
    if history["completed"]:
        # Keep the last training state in last.pt; restore the best model in memory only.
        model.load_state_dict(best_state, strict=True)
        model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    history["recorded_training_seconds"] = (
        previous_seconds + time.perf_counter() - segment_start
    )
    atomic_json_save(history, run_dir / "history.json")
    return history