"""Data generation matching ironjr/grokfast main.py."""

import torch


def multiplication_mod_p_data(p, eq_token, op_token):
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    return torch.stack([x, op, y, eq, result])
