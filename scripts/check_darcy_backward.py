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


def check_loss_definition():
    target = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    )
    pred = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    expected = torch.tensor([0.0, 0.5])

    flat_errors = relative_l2_per_sample(pred, target)
    grid_errors = relative_l2_per_sample(
        pred.reshape(2, 2, 2, 1),
        target.reshape(2, 2, 2, 1),
    )

    torch.testing.assert_close(flat_errors, expected)
    torch.testing.assert_close(grid_errors, expected)
    torch.testing.assert_close(
        flat_errors.mean(), torch.tensor(0.25)
    )

    print("Relative L2 definition checks passed.")
 
def check_one_step(name, model, inputs, target):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6,
    )

    optimizer.zero_grad(set_to_none=True)

    pred = model(*inputs)
    per_sample = relative_l2_per_sample(pred, target)

    assert per_sample.shape == (target.shape[0],)
    assert torch.isfinite(per_sample).all().item()

    loss = per_sample.mean()

    assert loss.ndim == 0
    assert loss.requires_grad

    loss_before = loss.item()

    loss.backward()

    trainable = [
        (key, parameter)
        for key, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    # Every trainable parameter tensor must have a finite gradient.
    for key, parameter in trainable:
        assert parameter.grad is not None, (
            f"{name}: missing gradient: {key}"
        )
        assert torch.isfinite(parameter.grad).all().item(), (
            f"{name}: non-finite gradient: {key}"
        )

    # Each parameterized block must receive a nonzero loss gradient.
    for block_name, block in model.named_children():
        gradients = [
            parameter.grad
            for parameter in block.parameters()
            if parameter.requires_grad
        ]

        if gradients:
            assert any(
                torch.count_nonzero(gradient).item() > 0
                for gradient in gradients
            ), f"{name}: all gradients are zero in {block_name}"

    # Keep independent copies for checking the update.
    before = [
        parameter.detach().clone()
        for _, parameter in trainable
    ]

    optimizer.step()

    changed = sum(
        not torch.equal(old, parameter.detach())
        for old, (_, parameter) in zip(before, trainable)
    )

    assert changed > 0, f"{name}: no parameters were updated"

    for key, parameter in trainable:
        assert torch.isfinite(parameter).all().item(), (
            f"{name}: non-finite parameter after update: {key}"
        )

    # Observe the new loss; do not require it to decrease after one step.
    with torch.no_grad():
        loss_after = relative_l2_per_sample(
            model(*inputs), target
        ).mean()

    assert torch.isfinite(loss_after).item()

    print(f"{name}: loss before={loss_before:.6e}")
    print(
        f"{name}: {len(trainable)} gradient tensors are finite; "
        "nonzero gradients reached every parameterized block."
    )
    print(
        f"{name}: updated parameter tensors="
        f"{changed}/{len(trainable)}"
    )
    print(f"{name}: loss after one step={loss_after.item():.6e}")    

def check_backward(data):
    check_loss_definition()

    torch.manual_seed(2026)
    device = torch.device("cpu")

    # Statistics are fitted on the entire training split.
    stats = fit_normalization(data["A_train"], data["U_train"])

    B = 2
    assert data["A_train"].shape[0] >= B

    A_normalized = normalize_coefficient(
        data["A_train"][:B], stats
    )
    U_scaled = scale_solution(data["U_train"][:B], stats)

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

    check_one_step(
        "DeepONet",
        deeponet,
        (branch_t, trunk_t),
        don_target_t,
    )
    check_one_step(
        "FNO",
        fno,
        (fno_input_t,),
        fno_target_t,
    )

    print("Darcy loss/backward checks passed.")


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

    check_backward(data)


if __name__ == "__main__":
    main()