import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            # first try to call __getattr__ from nn.Module
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith('_'):
                raise AttributeError("attempted to get missing private attribute '{}'".format(name))
            return getattr(self.model, name)


class IterateInput(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, inputs):
        assert len(inputs.shape) == 3, "inputs should be of shape (seq_len, batch_size, n_inputs)"
        seq_len, batch_size, n_inputs = inputs.shape

        state = self.model.init_state(batch_size)

        outputs = []
        states = []
        for inp in inputs:
            states.append(state.clone())
            out, state = self.model(inp, state)
            outputs.append(out)

        outputs = torch.stack(outputs)
        self.states = torch.stack(states)
        return outputs
