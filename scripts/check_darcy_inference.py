import numpy as np
import torch
from torch import nn

from operatorbench.inference import predict_batches, physical_solution_errors


class CheckModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Inference must disable training mode and gradients.")
        return self.weight * inputs


def main():
    # Five samples with batch_size=2 also exercise the final partial batch.
    inputs = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2) + 1
    model = CheckModel()
    model.train()
    weight_before = model.weight.detach().clone()
    output = predict_batches(model, inputs, batch_size=2)
    torch.testing.assert_close(output, 2 * inputs)
    assert model.training
    assert model.weight.grad is None
    assert not output.requires_grad
    assert torch.equal(model.weight, weight_before)

    reference = inputs.numpy()
    exact = physical_solution_errors(reference, reference)
    np.testing.assert_array_equal(exact["relative_l2"], np.zeros(5))
    np.testing.assert_array_equal(exact["max_absolute_error"], np.zeros(5))

    doubled = physical_solution_errors(2 * reference, reference)
    np.testing.assert_allclose(doubled["relative_l2"], np.ones(5), atol=1e-10)
    np.testing.assert_array_equal(
        doubled["max_absolute_error"], reference.max(axis=(1, 2))
    )
    rescaled = physical_solution_errors(6 * reference, 3 * reference)
    np.testing.assert_allclose(rescaled["relative_l2"], doubled["relative_l2"])
    np.testing.assert_allclose(
        rescaled["max_absolute_error"], 3 * doubled["max_absolute_error"]
    )

    try:
        physical_solution_errors(reference, np.zeros_like(reference))
    except ValueError:
        pass
    else:
        raise AssertionError("A zero reference must be rejected.")

    print("Darcy inference and metric checks passed.")


if __name__ == "__main__":
    main()