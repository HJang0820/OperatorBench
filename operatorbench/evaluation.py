import torch

from operatorbench.losses import relative_l2_per_sample


@torch.no_grad()
def evaluate_relative_l2(
    model, inputs, target, *, shared_inputs=(), batch_size=32
):
    """Return one relative L2 error per sample, in the original order."""
    n_samples = inputs.shape[0]
    if n_samples == 0 or target.shape[0] != n_samples:
        raise ValueError("Inputs and targets need matching, nonempty batches.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = model.training
    model.eval()
    all_errors = []

    try:
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            pred = model(inputs[start:stop], *shared_inputs)
            errors = relative_l2_per_sample(pred, target[start:stop])
            if not torch.isfinite(errors).all().item():
                raise FloatingPointError("Non-finite evaluation errors.")
            all_errors.append(errors.cpu())
    finally:
        model.train(was_training)

    return torch.cat(all_errors)