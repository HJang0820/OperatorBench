import numpy as np
import torch

from operatorbench.losses import relative_l2_per_sample


@torch.no_grad()
def predict_batches(model, inputs, *, shared_inputs=(), batch_size=16, device="cpu"):
    """Predict in sample order, transferring only one input batch at a time."""
    if inputs.shape[0] == 0 or batch_size < 1:
        raise ValueError("Inputs must be nonempty and batch_size must be positive.")

    device = torch.device(device)
    shared = tuple(value.to(device) for value in shared_inputs)
    was_training = model.training
    model.eval()
    predictions = []
    try:
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size].to(device)
            output = model(batch, *shared)
            if output.shape[0] != len(batch):
                raise ValueError("The output batch size does not match the input.")
            if not torch.isfinite(output).all().item():
                raise FloatingPointError("Predictions contain NaN or infinity.")
            predictions.append(output.detach().cpu())
    finally:
        model.train(was_training)

    return torch.cat(predictions, dim=0)


def physical_solution_errors(prediction, reference):
    """Return per-sample errors on full physical grids, including boundaries."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (prediction.shape != reference.shape or reference.ndim != 3
            or any(size == 0 for size in reference.shape)):
        raise ValueError("Expected matching, nonempty (samples, Nx, Ny) arrays.")
    if not (np.isfinite(prediction).all() and np.isfinite(reference).all()):
        raise ValueError("Prediction and reference must be finite.")
    if np.any(np.linalg.norm(reference.reshape(len(reference), -1), axis=1) == 0):
        raise ValueError("Relative L2 is undefined for a zero reference solution.")

    relative = relative_l2_per_sample(
        torch.from_numpy(prediction), torch.from_numpy(reference)
    ).numpy()
    maximum = np.abs(prediction - reference).max(axis=(1, 2))
    return {"relative_l2": relative, "max_absolute_error": maximum}