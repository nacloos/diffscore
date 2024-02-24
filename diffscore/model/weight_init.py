import numpy as np
import torch
import torch.nn as nn

# from config_utils.dict_utils import dict_get, dict_in


registry = {}

def weight_initialization(name):
    def weight_init_wrapper(func):
        registry[name] = func
        return func
    return weight_init_wrapper


def init_param_from_registry(param, init_name, init_kwargs):
    print("Init {}, kwargs {}".format(init_name, init_kwargs))
    param = registry[init_name](param, **init_kwargs)
    return param


# TODO: use regsiter/make
def init_param(param, init_type):
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


def init_weights(model, init_type, debug=False):
    """
    Args:
        model: pytorch model
        init_type: dict specifying the initialization types of the model parameters
            e.g. {"fc_rec": {"weight": "zero"}}
    """
    state_dict = model.state_dict()
    for param_name, param in model.named_parameters():
        print("Param: {}".format(param_name)) if debug else None
        if param_name.startswith('model.'):
            # remove 'model.' from param_name
            param_name = param_name.split('model.')[1]
            # TODO: temporary
            starts_with_model = True
        else:
            starts_with_model = False

        # param_name uses dot dict indexing
        if dict_in(init_type, param_name):
            _init_type = dict_get(init_type, param_name)
            print("Init: {}, type: {}".format(param_name, _init_type)) if debug else None

            if isinstance(_init_type, dict) and 'name' in _init_type:
                init_name = _init_type.pop('name')
                param = init_param_from_registry(param, init_name, _init_type)
            else:
                # backward compatibility
                param = init_param(param, _init_type)

            if starts_with_model:
                param_name = 'model.{}'.format(param_name)

            state_dict[param_name] = param

        model.load_state_dict(state_dict)
    return model