import torch


def relative_l2_per_sample(pred, target, eps=1e-12):
    """Return one relative L2 error for each sample."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={pred.shape}, target={target.shape}"
        )

    if pred.ndim < 2 or pred.numel() == 0:
        raise ValueError(
            "Expected nonempty tensors with a batch dimension."
        )

    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    error_norm = torch.linalg.vector_norm(
        pred_flat - target_flat, dim=1
    )
    target_norm = torch.linalg.vector_norm(target_flat, dim=1)

    return error_norm / (target_norm + eps)