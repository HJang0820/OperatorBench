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


def parameter_counts(model):
    elements = 0
    real_scalars = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            elements += parameter.numel()

            multiplier = 2 if parameter.is_complex() else 1
            real_scalars += multiplier * parameter.numel()

    return elements, real_scalars


def check_models(data):
    torch.manual_seed(2026)
    device = torch.device("cpu")

    # Fit on the whole training split, not just the two check samples.
    stats = fit_normalization(data["A_train"], data["U_train"])

    B = 2
    assert data["A_train"].shape[0] >= B

    A_normalized = normalize_coefficient(
        data["A_train"][:B], stats
    )
    U_scaled = scale_solution(data["U_train"][:B], stats)

    x, y = data["x"], data["y"]
    Nx, Ny = len(x), len(y)
    M = Nx * Ny

    assert (Nx, Ny) == (85, 85)

    coords01, trunk = make_deeponet_coords(x, y)

    branch_t = torch.from_numpy(
        make_deeponet_branch(A_normalized)
    ).to(device)

    trunk_t = torch.from_numpy(trunk).to(device)
    coords01_t = torch.from_numpy(coords01).to(device)

    fno_input_t = torch.from_numpy(
        make_fno_input(A_normalized, x, y)
    ).to(device)

    don_target_t = torch.from_numpy(
        make_deeponet_target(U_scaled)
    ).to(device)

    fno_target_t = torch.from_numpy(
        make_fno_target(U_scaled)
    ).to(device)

    deeponet = DeepONet2d(
        n_sensors=M,
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

    deeponet.eval()
    fno.eval()

    with torch.no_grad():
        don_output = deeponet(branch_t, trunk_t)
        fno_output = fno(fno_input_t)

    assert don_output.shape == don_target_t.shape == (B, M)
    assert fno_output.shape == fno_target_t.shape == (B, Nx, Ny, 1)

    for output in (don_output, fno_output):
        assert output.dtype == torch.float32
        assert output.device == device
        assert torch.isfinite(output).all().item()

    # Reshaping does not convert scaled predictions to physical units.
    don_grid = don_output.reshape(B, Nx, Ny)
    fno_grid = fno_output[..., 0]

    assert don_grid.shape == fno_grid.shape == (B, Nx, Ny)

    don_counts = parameter_counts(deeponet)
    fno_counts = parameter_counts(fno)

    assert don_counts == (3_312_449, 3_312_449)
    assert fno_counts == (3_285_505, 6_562_305)

    print("device:", device)
    print("DeepONet output:", tuple(don_output.shape))
    print("FNO output:", tuple(fno_output.shape))
    print("DeepONet grid:", tuple(don_grid.shape))
    print("FNO grid:", tuple(fno_grid.shape))

    print(
        f"DeepONet parameters: elements={don_counts[0]:,}, "
        f"real scalars={don_counts[1]:,}"
    )
    print(
        f"FNO parameters: elements={fno_counts[0]:,}, "
        f"real scalars={fno_counts[1]:,}"
    )

    print("Darcy model forward checks passed.")


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

    check_models(data)


if __name__ == "__main__":
    main()