import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from operatorbench.models.deeponet import DeepONet2d
from operatorbench.models.fno import FNO2d
from operatorbench.training import same_value, train_model


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # This checks checkpoint restoration, not physical solution accuracy.
    # It does not use production datasets or run directories.
    torch.set_num_threads(1)
    axis = torch.linspace(0, 1, 9)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    reset_seed(11)
    a = torch.randn(8, 9, 9)
    u = 1.0 + torch.rand(8, 9, 9)

    config = {
        "num_steps": 6,
        "batch_size": 2,
        "validate_every": 2,
        "eval_batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        # Change the learning rate at step 3 and test resuming immediately after.
        "scheduler_step_size": 3,
        "scheduler_gamma": 0.5,
        "sampling_seed": 2027,
        "checkpoint_every": 2,
    }

    def make_model(name):
        if name == "DeepONet":
            return DeepONet2d(
                n_sensors=81, latent_dim=8,
                branch_hidden=[16], trunk_hidden=[8],
                coords01=coords, use_hard_dirichlet=False,
            )
        return FNO2d(modes1=3, modes2=3, width=4)

    with tempfile.TemporaryDirectory(prefix="operatorbench-resume-check-") as tmp:
        root = Path(tmp)
        for name in ("DeepONet", "FNO"):
            if name == "DeepONet":
                inputs, targets = a.flatten(1), u.flatten(1)
                shared = (2.0 * coords - 1.0,)
            else:
                inputs = torch.stack((
                    a, xx.expand_as(a), yy.expand_as(a),
                ), dim=-1)
                targets = u[..., None]
                shared = ()
            arguments = (
                inputs[:6], targets[:6], inputs[6:], targets[6:],
            )
            info = {
                "model_name": name,
                "purpose": "resume_check",
                "dataset_sha256": "synthetic_inputs_for_resume_check_only",
            }
            full_dir = root / (name + "_full")
            resumed_dir = root / (name + "_resumed")

            reset_seed(2026)
            full_history = train_model(
                make_model(name), *arguments, shared_inputs=shared,
                config=config, run_info=info, run_dir=full_dir,
            )
            reset_seed(2026)
            partial = train_model(
                make_model(name), *arguments, shared_inputs=shared,
                config=config, run_info=info, run_dir=resumed_dir,
                stop_after=3,
            )
            assert partial["completed_steps"] == 3 and not partial["completed"]
            # Deliberately change the random states and initial weights before resuming.
            reset_seed(999)
            resumed_history = train_model(
                make_model(name), *arguments, shared_inputs=shared,
                config=config, run_info=info, run_dir=resumed_dir,
                resume=True,
            )
            assert resumed_history["completed_steps"] == 6
            assert resumed_history["completed"]
            assert same_value(full_history["train"], resumed_history["train"])
            assert same_value(
                full_history["validation"], resumed_history["validation"]
            )
            full = torch.load(
                full_dir / "last.pt", map_location="cpu", weights_only=True
            )
            resumed = torch.load(
                resumed_dir / "last.pt", map_location="cpu", weights_only=True
            )
            for key in (
                "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "rng_state",
                "best_model_state_dict", "best_step", "best_val_relative_l2",
            ):
                assert same_value(full[key], resumed[key]), f"{name}: {key}"
            assert full_history["train"][3]["learning_rate"] == 5e-4

            changed_config = dict(config, learning_rate=2e-3)
            try:
                train_model(
                    make_model(name), *arguments, shared_inputs=shared,
                    config=changed_config, run_info=info,
                    run_dir=resumed_dir, resume=True,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Resuming with different settings was not rejected.")
            print(f"{name}: uninterrupted and resumed training match.")

    print("Darcy checkpoint save/resume checks passed.")


if __name__ == "__main__":
    main()