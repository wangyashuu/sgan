from torch import nn

from sgan.models import make_mlp

# MARK: - Models


class QHead(nn.Module):
    def __init__(
        self,
        dims,
        n_disc_code,
        n_cont_code,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super().__init__()
        # [h_dim, mlp_dim, sum(n_disc_code)]
        self.disc_backbone = make_mlp(
            dims + [sum(n_disc_code)],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        # [h_dim, mlp_dim, n_cont_code*2]
        self.cont_backbone = make_mlp(
            dims + [n_cont_code * 2],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, x):
        disc_info = self.disc_backbone(x)
        cont_info = self.cont_backbone(x)
        return disc_info, cont_info


class DHead(nn.Module):
    def __init__(self, dims, activation="relu", batch_norm=True, dropout=0.0):
        super().__init__()
        # [h_dim, mlp_dim, 1]
        self.backbone = make_mlp(
            dims + [1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
