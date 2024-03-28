import numpy as np
import torch
import torch.nn as nn


def init_param(param, init_type):
    if init_type is None:
        return param

    if isinstance(init_type, str):
        dim_recurrent = param.shape[0]

        # Naive
        if init_type == 'zero' or init_type == 'Zero':
            param = torch.nn.Parameter(torch.zeros_like(param))

        elif init_type == 'medium' or init_type == 'Medium':
            param = torch.nn.Parameter(torch.randn(param.shape[0], param.shape[1]) / np.sqrt(param.shape[1]))

        elif init_type == 'default' or init_type == 'PyTorch':
            pass

        elif init_type == 'EdgeofChaos' or init_type == 'orthonormal':
            W = np.random.randn(dim_recurrent, dim_recurrent)
            u, s, vT = np.linalg.svd(W)
            W = u @ np.diag(1.0*np.ones(dim_recurrent)) @ vT
            W = torch.tensor(W, dtype=torch.float32)
            param = torch.nn.Parameter(W)

        elif init_type == 'Xavier2010':
            param = torch.nn.Parameter(torch.zeros_like(param))
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))

        elif init_type == 'Xavier2010_norm':
            param = torch.nn.Parameter(torch.zeros_like(param))
            nn.init.xavier_normal_(param)

        elif init_type == 'Le2015':
            param = torch.nn.Parameter(torch.eye(dim_recurrent))
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

        elif init_type == 'Le2015_norm':
            param = torch.nn.Parameter(torch.eye(dim_recurrent))
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

        elif init_type == 'He2015':
            param = torch.nn.Parameter(torch.normal(0, 2./dim_recurrent, size=(dim_recurrent, dim_recurrent)))

        elif init_type == 'Sussillo2015':
            param = 1.5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
            param = torch.nn.Parameter(param)

        else:
            raise ValueError("Unknown init type {}".format(init_type))

    elif isinstance(init_type, dict):
        assert 'type' in init_type
        if init_type['type'] == 'sussillo' or init_type['type'] == 'Sussillo2015g':
            assert 'g' in init_type
            assert param.shape[0] == param.shape[1]
            g = init_type['g']
            dim_recurrent = param.shape[0]
            param = g * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
        else:
            raise ValueError("Unknown init type {}".format(init_type['type']))
    else:
        raise TypeError("Expected type str or dict, got {} for {}".format(type(init_type), init_type))

    return param
