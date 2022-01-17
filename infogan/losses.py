import torch
from torch.distributions.normal import Normal

# MARK: - Loss


def logli(x_var, mean, stddev):
    """
    TODO: why use sum in logli
    """
    # return Normal(mean, stddev).log_prob(x_var).sum(axis=1)
    TINY = torch.tensor(1e-8).float()
    TINY = TINY.cuda() if torch.cuda.is_available() else TINY
    pi = 2 * torch.acos(torch.zeros(1))
    epsilon = (x_var - mean) / (stddev + TINY)
    return torch.sum(
        -0.5 * torch.log(2 * pi)
        - torch.log(stddev + TINY)
        - 0.5 * (epsilon ** 2),
        axis=1,
    )


def original_normal_dist_loss(x_var, mean, stddev):
    cont_log_q_c_given_x = logli(x_var, mean, stddev)
    cont_log_q_c = logli(x_var, mean=0, stddev=1)
    cont_cross_ent = torch.mean(-cont_log_q_c_given_x)
    cont_ent = torch.mean(-cont_log_q_c)
    cont_mi_est = cont_ent - cont_cross_ent
    return -cont_mi_est


def simplified_nll_loss(x_var, mean, stddev):
    # maximize the prob
    return -Normal(mean, stddev).log_prob(x_var).mean()


def nll_loss(x_var, mean, logvar):
    stddev = torch.sqrt(torch.exp(logvar))
    loss = simplified_nll_loss(x_var, mean, stddev)
    return loss


def q_disc_loss_fn(disc_code, q_logits, n_disc_code):
    disc_loss = torch.tensor(0).float()
    disc_loss = disc_loss.cuda() if torch.cuda.is_available() else disc_loss
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
    cont_loss = torch.tensor(0).float()
    cont_loss = cont_loss.cuda() if torch.cuda.is_available() else cont_loss

    if n_cont_code > 0:
        q_mu, q_logvar = q_cont[:, :n_cont_code], q_cont[:, n_cont_code:]
        cont_loss = nll_loss(cont_code, q_mu, q_logvar)
    return cont_loss


def q_loss_fn(q_info, code_tuple, n_disc_code, n_cont_code):
    disc_code, cont_code = code_tuple
    q_logits, q_cont = q_info
    disc_loss = q_disc_loss_fn(disc_code, q_logits, n_disc_code)
    cont_loss = q_cont_loss_fn(cont_code, q_cont, n_cont_code)
    return disc_loss, cont_loss
