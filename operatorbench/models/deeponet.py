import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.Tanh):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet2d(nn.Module):
    def __init__(
        self,
        n_sensors,
        latent_dim,
        branch_hidden,
        trunk_hidden,
        coords01,
        use_hard_dirichlet=False,
    ):
        super().__init__()

        self.branch = MLP(
            n_sensors, branch_hidden, latent_dim, activation=nn.Tanh
        )
        self.trunk = MLP(
            2, trunk_hidden, latent_dim, activation=nn.Tanh
        )

        self.bias = nn.Parameter(torch.zeros(1))
        self.use_hard_dirichlet = use_hard_dirichlet

        x, y = coords01[:, 0], coords01[:, 1]
        bc_factor = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        self.register_buffer("bc_factor", bc_factor)

    def forward(self, branch_input, trunk_coords):
        branch_features = self.branch(branch_input)
        trunk_features = torch.tanh(self.trunk(trunk_coords))

        output = torch.einsum(
            "bp,np->bn", branch_features, trunk_features
        )
        output = output + self.bias

        if self.use_hard_dirichlet:
            output = output * self.bc_factor[None, :]

        return output