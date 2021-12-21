import torch
import torch.nn as nn
from sgan.models import make_mlp
from torch.distributions.normal import Normal


# MARK: - Generate Code


def get_disc_code(n_samples, n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        return torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.randint(low=0, high=dim, size=(n_samples,)), dim
                )
                for dim in n_disc_code
            ],
            dim=1,
        ).cuda()
    return None


def get_cont_code(n_samples, n_cont_code=0):
    if n_cont_code > 0:
        return torch.randn(n_samples, n_cont_code).cuda()
    return None


def get_fixed_disc_code(n_samples, disc_code=[], n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        return torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(disc_code[i]).repeat(n_samples),
                    n_disc_code[i],
                )
                for i in range(len(n_disc_code))
            ],
            dim=1,
        ).cuda()
    return None


def get_fixed_cont_code(n_samples, cont_code=[], n_cont_code=0):
    if n_cont_code > 0:
        return torch.tensor(cont_code).repeat(n_samples, 1).float().cuda()
    return None


def get_latent_code(disc_code, cont_code):
    var = []
    if disc_code is not None:
        var.append(disc_code)
    if cont_code is not None:
        var.append(cont_code)
    if len(var) > 0:
        return torch.cat(var, dim=1)
    return None


def expand_code(code, seq_start_end):
    if code is None:
        return None
    return torch.cat(
        [
            code[idx].view(1, -1).repeat(ed.item() - st.item(), 1)
            for idx, (st, ed) in enumerate(seq_start_end)
        ],
        dim=0,
    )


# MARK: - Loss
# TODO: infogan pytorch donot add qloss to discriminator
# TODO: predict logvar


def logli(x_var, mean, stddev):
    """
    TODO: why use sum in logli
    """
    # return Normal(mean, stddev).log_prob(x_var).sum(axis=1)
    TINY = torch.tensor(1e-8).float().cuda()
    pi = 2 * torch.acos(torch.zeros(1))
    epsilon = (x_var - mean) / (stddev + TINY)
    return torch.sum(
        -0.5 * torch.log(2 * pi)
        - torch.log(stddev + TINY)
        - 0.5 * (epsilon ** 2),
        axis=1,
    )


def original_normal_dist_loss():
    cont_log_q_c_given_x = logli(x_var, mean, stddev)
    cont_log_q_c = logli(x_var, mean=0, stddev=1)
    cont_cross_ent = torch.mean(-cont_log_q_c_given_x)
    cont_ent = torch.mean(-cont_log_q_c)
    cont_mi_est = cont_ent - cont_cross_ent
    return -cont_mi_est


def simplified_normal_dist_loss(x_var, mean, stddev):
    # maximize the prob
    return -Normal(mean, stddev).log_prob(x_var).mean()


def normal_dist_loss(x_var, mean, logvar):
    stddev = torch.sqrt(torch.exp(logvar))
    loss = simplified_normal_dist_loss(x_var, mean, stddev)
    return loss


def q_disc_loss_fn(disc_code, q_logits, n_disc_code):
    disc_loss = torch.tensor(0).float().cuda()
    if len(n_disc_code) > 0:
        st, ed = 0, 0
        for dim in n_disc_code:
            st, ed = ed, ed + dim
            cur_target = torch.argmax(disc_code[:, st:ed], 1)
            disc_loss += torch.nn.functional.cross_entropy(
                q_logits[:, st:ed], cur_target
            )
    return disc_loss


def q_cont_loss_fn(cont_code, q_cont, n_cont_code):
    cont_loss = torch.tensor(0).float().cuda()
    if n_cont_code > 0:
        q_mu, q_logvar = q_cont[:, :n_cont_code], q_cont[:, n_cont_code:]
        cont_loss = normal_dist_loss(cont_code, q_mu, q_logvar)
    return cont_loss


def q_loss_fn(q_info, code_tuple, n_disc_code, n_cont_code):
    disc_code, cont_code = code_tuple
    q_logits, q_cont = q_info
    disc_loss = q_disc_loss_fn(disc_code, q_logits, n_disc_code)
    cont_loss = q_cont_loss_fn(cont_code, q_cont, n_cont_code)
    loss = disc_loss + cont_loss
    return loss


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
