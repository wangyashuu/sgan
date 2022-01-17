import torch

# MARK: - Generate Code

def get_disc_code(n_samples, n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        result = torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.randint(low=0, high=dim, size=(n_samples,)), dim
                )
                for dim in n_disc_code
            ],
            dim=1,
        )
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_cont_code(n_samples, n_cont_code=0):
    if n_cont_code > 0:
        result = torch.randn(n_samples, n_cont_code)
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_fixed_disc_code(n_samples, disc_code=[], n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        result = torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(disc_code[i]).repeat(n_samples),
                    n_disc_code[i],
                )
                for i in range(len(n_disc_code))
            ],
            dim=1,
        )
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_fixed_cont_code(n_samples, cont_code=[], n_cont_code=0):
    if n_cont_code > 0:
        result = torch.tensor(cont_code).repeat(n_samples, 1).float()
        return result.cuda() if torch.cuda.is_available() else result


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