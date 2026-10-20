from pathlib import Path

import torch

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import (
    fit_normalization,
    normalize_coefficient,
    scale_solution,
)
from operatorbench.data.model_inputs import (
    make_deeponet_branch,
    make_deeponet_target,
    make_deeponet_coords,
    make_fno_input,
    make_fno_target,
)
from operatorbench.models.deeponet import DeepONet2d
from operatorbench.models.fno import FNO2d
from operatorbench.losses import relative_l2_per_sample


NUM_STEPS = 300
PRINT_EVERY = 50
BATCH_SIZE = 2
REQUIRED_RATIO = 0.5


@torch.no_grad()
def fixed_batch_errors(model, inputs, target):
    model.eval()

    errors = relative_l2_per_sample(
        model(*inputs), target
    )

    assert torch.isfinite(errors).all().item()

    return errors.detach().cpu()

def train_fixed_batch(name, model, inputs, target):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5000,
        gamma=0.5,
    )

    initial_errors = fixed_batch_errors(
        model, inputs, target
    )
    initial_mean = initial_errors.mean().item()

    assert initial_mean > 0.0

    print(
        f"{name}: initial mean relative L2 = "
        f"{initial_mean:.6e}"
    )

    for step in range(1, NUM_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred = model(*inputs)
        loss = relative_l2_per_sample(
            pred, target
        ).mean()

        assert torch.isfinite(loss).item(), (
            f"{name}: non-finite loss at step {step}"
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == NUM_STEPS
        ):
            errors = fixed_batch_errors(
                model, inputs, target
            )

            per_sample_text = ", ".join(
                f"{100.0 * value:.3f}%"
                for value in errors.tolist()
            )

            print(
                f"{name}: step {step:4d} | "
                f"mean rel L2 = "
                f"{100.0 * errors.mean().item():.3f}% | "
                f"samples = [{per_sample_text}] | "
                f"lr = {optimizer.param_groups[0]['lr']:.2e}"
            )

    final_errors = fixed_batch_errors(
        model, inputs, target
    )
    final_mean = final_errors.mean().item()

    ratio = final_mean / initial_mean

    both_samples_improved = bool(
        torch.all(final_errors < initial_errors).item()
    )

    passed = (
        ratio < REQUIRED_RATIO
        and both_samples_improved
    )

    print(
        f"{name}: initial={initial_mean:.6e}, "
        f"final={final_mean:.6e}, "
        f"final/initial={ratio:.4f}"
    )
    print(
        f"{name}: both samples improved = "
        f"{both_samples_improved}"
    )
    print(
        f"{name}: fixed-batch reduction check = {passed}"
    )

    return passed

def check_fixed_batch(data):
    assert NUM_STEPS > 0 and PRINT_EVERY > 0
    assert 0.0 < REQUIRED_RATIO < 1.0

    torch.manual_seed(2026)
    device = torch.device("cpu")

    # Fit on the whole training split, not just two samples.
    stats = fit_normalization(
        data["A_train"], data["U_train"]
    )

    assert data["A_train"].shape[0] >= BATCH_SIZE

    A_normalized = normalize_coefficient(
        data["A_train"][:BATCH_SIZE], stats
    )
    U_scaled = scale_solution(
        data["U_train"][:BATCH_SIZE], stats
    )

    x, y = data["x"], data["y"]
    Nx, Ny = len(x), len(y)

    assert (Nx, Ny) == (85, 85)

    coords01, trunk = make_deeponet_coords(x, y)

    branch_t = torch.from_numpy(
        make_deeponet_branch(A_normalized)
    ).to(device)
    trunk_t = torch.from_numpy(trunk).to(device)
    coords01_t = torch.from_numpy(coords01).to(device)

    don_target_t = torch.from_numpy(
        make_deeponet_target(U_scaled)
    ).to(device)

    fno_input_t = torch.from_numpy(
        make_fno_input(A_normalized, x, y)
    ).to(device)
    fno_target_t = torch.from_numpy(
        make_fno_target(U_scaled)
    ).to(device)

    deeponet = DeepONet2d(
        n_sensors=Nx * Ny,
        latent_dim=128,
        branch_hidden=(400, 400, 400),
        trunk_hidden=(128, 128, 128),
        coords01=coords01_t,
        use_hard_dirichlet=False,
    ).to(device)

    fno = FNO2d(
        modes1=20,
        modes2=20,
        width=32,
    ).to(device)

    print("device:", device)
    print(
        "fixed training sample indices:",
        list(range(BATCH_SIZE)),
    )

    don_passed = train_fixed_batch(
        "DeepONet",
        deeponet,
        (branch_t, trunk_t),
        don_target_t,
    )

    fno_passed = train_fixed_batch(
        "FNO",
        fno,
        (fno_input_t,),
        fno_target_t,
    )

    assert don_passed and fno_passed, (
        "Fixed-batch reduction criterion was not met. "
        "Inspect the initial/final losses and per-sample logs."
    )

    print("Darcy fixed-batch learning checks passed.")


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "darcy" / (
        "darcy_binary_f1_v1_full_N85"
        "_train1000_val200_test200_seeds2022-2024-2025.npz"
    )

    data, metadata = load_darcy_dataset(dataset_path)

    assert metadata["mode"] == "full"
    assert metadata["task"] == "a_to_u"
    assert metadata["source"] == {
        "kind": "constant",
        "value": 1.0,
    }

    check_fixed_batch(data)


if __name__ == "__main__":
    main()