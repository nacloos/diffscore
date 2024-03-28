from functools import partial
import torch

from diffscore.nn import CTRNN, LSTM, LowPassCTRNN, IterateInput


def test_rnn():
    input_size = 10
    hidden_size = 15
    output_size = 5
    act_fn_values = ["Tanh", "ReTanh", "ReLU"]

    rnns = [
        partial(CTRNN, input_size, hidden_size, output_size, tau=10, dt=1),
        partial(LowPassCTRNN, input_size, hidden_size, output_size, tau=10, dt=1),
        partial(LSTM, input_size, hidden_size, output_size),
    ]

    inputs = torch.randn(3, 4, input_size)
    for rnn in rnns:
        for act_fn in act_fn_values:
            print(rnn)
            model = IterateInput(
                rnn(act_fn=act_fn)
            )
            outputs = model(inputs)
            assert outputs.shape == (3, 4, output_size), "Output shape mismatch"


if __name__ == "__main__":
    test_rnn()
