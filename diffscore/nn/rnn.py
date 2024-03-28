import math
import numpy as np
import torch
import torch.nn as nn

from .weight_init import init_param
from .activation_fn import get_activation_fn


class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_std=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std

        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def init_state(self, batch_size):
        raise NotImplementedError

    def recurrence(self, inp, h):
        raise NotImplementedError

    def forward(self, inp, state):
        """
        Args:
            inp: batch_size x n_inputs
            state: same as init_state
        Return:
            out: batch_size x n_outputs
            next_state: same as state
        """
        next_state = self.recurrence(inp, state)
        out = self.fc_out(next_state)
        return out, next_state


class RNN(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, act_fn='Tanh', noise_std=0, learn_init_state=True):
        super().__init__(input_size, hidden_size, output_size)
        self.noise_std = noise_std
        self.learn_init_state = learn_init_state

        self.act_fn = get_activation_fn(act_fn)
        self.fc_rec = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_in = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.h0 = torch.zeros(self.hidden_size)
        if self.learn_init_state:
            self.h0 = torch.nn.Parameter(self.h0, requires_grad=True)

    def init_state(self, batch_size):
        init_state = self.h0.repeat(batch_size, 1)
        return init_state

    def recurrence(self, inp, h):
        noise = self.noise_std*torch.randn(h.size()).to(inp.device)
        next_h = self.act_fn(self.fc_rec(h) + self.fc_in(inp) + noise)
        return next_h


class CTRNN(BaseRNN):
    def __init__(self, input_size, hidden_size: int, output_size, tau: float, dt, act_fn='Tanh', noise_std=0, learn_init_state=True, rec_init=None, in_init=None):
        super().__init__(input_size, hidden_size, output_size)
        self.tau = tau  # neuronal time constants (ms)
        self.dt = dt
        self.noise_std = noise_std
        self.learn_init_state = learn_init_state
        self.act_fn = get_activation_fn(act_fn)

        self.fc_rec = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_in = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.fc_rec.weight = init_param(self.fc_rec.weight, rec_init)
        self.fc_in.weight = init_param(self.fc_in.weight, in_init)

        self.h0 = torch.zeros(self.hidden_size)
        if self.learn_init_state:
            self.h0 = torch.nn.Parameter(self.h0, requires_grad=True)

    def init_state(self, batch_size):
        init_state = self.h0.repeat(batch_size, 1)
        # self.ah = init_state
        return init_state

    def recurrence(self, inp, h):
        self.alpha = self.dt/self.tau
        # print(self.tau)
        noise = self.noise_std*torch.randn(h.size()).to(inp.device)
        # self.ah = (1 - self.alpha) * self.ah + self.alpha * (
        #             self.fc_rec(h) + self.fc_in(inp)) + noise
        # next_h = self.act_fn(self.ah)
        r = self.act_fn(h)
        next_h = (1-self.alpha)*h + self.alpha*(self.fc_rec(r) + self.fc_in(inp)) + noise
        return next_h

    def forward(self, inp, state):
        next_state = self.recurrence(inp, state)
        # different from BaseRNN
        r = self.act_fn(next_state)
        out = self.fc_out(r)
        return out, next_state


class LowPassCTRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, tau, dt, act_fn='Tanh', noise_std=0, learn_init_state=True, rec_init=None):
        super().__init__(input_size, hidden_size, output_size)
        self.tau = tau  # neuronal time constants (ms)
        self.dt = dt
        self.alpha = self.dt / self.tau
        self.noise_std = noise_std
        self.learn_init_state = learn_init_state

        self.act_fn = get_activation_fn(act_fn)

        self.fc_rec = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_in = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.fc_rec.weight = init_param(self.fc_rec.weight, rec_init)

        self.h0 = torch.zeros(self.hidden_size)
        if self.learn_init_state:
            self.h0 = torch.nn.Parameter(self.h0, requires_grad=True)


    def init_state(self, batch_size):
        init_state = self.h0.repeat(batch_size, 1)
        return init_state

    def recurrence(self, inp, h):
        noise = self.noise_std*torch.randn(h.size()).to(inp.device)
        next_h = (1-self.alpha)*h + self.alpha*self.act_fn(self.fc_rec(h) + self.fc_in(inp) + noise)
        return next_h


class LSTM(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, act_fn):
        super().__init__(input_size, hidden_size, output_size)
        self.act_fn = get_activation_fn(act_fn)

        self.fc_i_x2i = nn.Linear(input_size, hidden_size)  # Wix @ x + bi
        self.fc_i_h2i = nn.Linear(hidden_size, hidden_size, bias=False)  # Wih @ h
        self.fc_o_x2o = nn.Linear(input_size, hidden_size)  # Wox @ x + bo
        self.fc_o_h2o = nn.Linear(hidden_size, hidden_size, bias=False)  # Woh @ h
        self.fc_f_x2f = nn.Linear(input_size, hidden_size)  # Wfx @ x + bf
        self.fc_f_h2f = nn.Linear(hidden_size, hidden_size, bias=False)  # Wfh @ h
        self.fc_z_x2z = nn.Linear(input_size, hidden_size)  # Wzx @ x + bz
        self.fc_z_h2z = nn.Linear(hidden_size, hidden_size, bias=False)  # Wzh @ h
        self.fc_y_h2y = nn.Linear(hidden_size, output_size)  # y = Wyh @ h + by
        # initialize the biases to be 0
        self.fc_o_x2o.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.fc_z_x2z.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.fc_y_h2y.bias = torch.nn.Parameter(torch.zeros(output_size))
        self.fc_f_x2f.bias = torch.nn.Parameter(torch.linspace(1, 10, hidden_size))
        self.fc_i_x2i.bias = torch.nn.Parameter(torch.linspace(-1, -10, hidden_size))

        self.c0 = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.h0 = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_z_h2z.weight = torch.nn.Parameter(torch.tensor(np.random.randn(self.hidden_size, self.hidden_size), dtype=torch.float32))

    def init_state(self, batch_size):
        self.c = self.c0.repeat(batch_size, 1)
        h = self.h0.repeat(batch_size, 1)
        return h

    def recurrence(self, inp, h):
        inp = inp
        h = h
        i = torch.sigmoid(self.fc_i_x2i(inp) + self.fc_i_h2i(h))
        o = torch.sigmoid(self.fc_o_x2o(inp) + self.fc_o_h2o(h))
        f = torch.sigmoid(self.fc_f_x2f(inp) + self.fc_f_h2f(h))
        z = torch.tanh(self.fc_z_x2z(inp) + self.fc_z_h2z(h))
        self.c = i*z + f*self.c
        h = o*self.act_fn(self.c)

        if self.noise_std is not None:
            h += (self.noise_std*torch.randn(h.shape))
        return h
