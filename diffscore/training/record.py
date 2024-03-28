import torch
from torch.utils.data import DataLoader

from diffscore import Env
from diffscore.training.trainer import NGymOnlineDataset


def data_aligned_model_inputs(env, conditions):
    """
    Generate model inputs and targets aligned with the conditions.

    Args:
        env: id or object to generate model observations and targets
        conditions: list of dicts, each dict is a condition to generate a trial

    Returns:
        x: model inputs, pytorch tensor of shape (seq_len x batch x input_size)
        y: model targets, pytorch tensor of shape (seq_len x batch x output_size)
    """
    env = Env(env) if isinstance(env, str) else env

    env_dataset = NGymOnlineDataset(env, n_samples=len(conditions), condition=conditions)
    dataloader = DataLoader(env_dataset, batch_size=len(conditions))
    x, y = next(iter(dataloader))
    x, y = x.float(), y.float()
    # x: batch x seq_len x input_size
    x, y = torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)
    # x: seq_len x batch x input_size
    return x, y


def record_ngym_lit(
    model,
    x=None,
    y=None,
    env=None,
    conditions=None,
    record_window=None,
    dt=None
):
    """
    Record model activity and predictions on a given environment and conditions.
    Assume that the model has a states attribute that stores the model activity.

    Args:
        model: Model object
        x: Model inputs, pytorch tensor of shape (seq_len x batch x input_size)
        y: Model targets, pytorch tensor of shape (seq_len x batch x output_size)
        env: Environment object
        conditions: list of dicts, each dict is a condition to generate a trial
        record_window: tuple of start and stop time to record model activity
        dt: time step of the environment

    Returns:
        dict:
            x: Model inputs
            y: Model targets
            output: Model predictions
            activity: Model activity
            conditions: Conditions used to generate the model inputs and targets
    """
    if x is not None:
        assert y is not None, "Must provide y if x is provided"
        if record_window is not None:
            assert dt is not None, "Must provide dt if record_window is provided"
    else:
        assert env is not None, "Must provide env if x is not provided"
        assert conditions is not None, "Must provide conditions if x is not provided"

        env = Env(env) if isinstance(env, str) else env
        x, y = data_aligned_model_inputs(env=env, conditions=conditions)
        if record_window is not None and dt is None:
            assert hasattr(env, "dt"), "If dt is not provided, env must have dt attribute"
            dt = env.dt

    model.eval()
    model = model.to(x.device)
    with torch.no_grad():
        pred = model(x)

    assert hasattr(model, "states"), "Model must have states attribute"
    model_act = model.states.detach().numpy()
    model_act = model_act.astype(float)

    if record_window is not None:
        assert dt is not None, "Must provide dt if record_window is provided"
        start_time, stop_time = record_window
        if stop_time is not None:
            model_act = model_act[:int(stop_time/dt)]
        if start_time is not None:
            model_act = model_act[int(start_time/dt):]

    return {
        "x": x,
        "y": y,
        "output": pred,
        "activity": model_act,
        "conditions": conditions,
        # backward compatibility
        "model_output": pred,
        "model_activity": model_act,
    }
