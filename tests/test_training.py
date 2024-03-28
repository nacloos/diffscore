import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from diffscore.training import NGymLitTrainer
from diffscore.training.trainer import LitModule
from diffscore import Env
from diffscore.nn import CTRNN, IterateInput


def test_lit_module():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, seq_len, n_samples, ob_size, tgt_size):
            super().__init__()
            self.n_samples = n_samples
            self.seq_len = seq_len

            self.x = torch.ones((seq_len, self.n_samples, ob_size))
            self.y = torch.ones((seq_len, self.n_samples, tgt_size))

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return self.x[:, idx], self.y[:, idx]

    dataset = DummyDataset(50, 1000, 5, 3)
    dataloader = DataLoader(dataset=dataset, batch_size=64)

    module = IterateInput(CTRNN(5, 100, 3, 100, 50))
    lit_module = LitModule(module, loss_fn=torch.nn.MSELoss(), optimizer=torch.optim.Adam, accuracy=None)
    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(lit_module, train_dataloaders=dataloader)


def test_trainer_env():
    env = Env("mante")
    model = CTRNN(
        input_size=env.observation_space.shape[0],
        hidden_size=100,
        output_size=env.action_space.n,
        tau=100,
        dt=env.dt
    )
    model = IterateInput(model)

    trainer = NGymLitTrainer(
        loss_fn="CrossEntropyLoss",
        batch_size=10,
        n_samples=10,
        max_epochs=1,
    )
    trainer.fit(model, env)


if __name__ == "__main__":
    test_lit_module()
    test_trainer_env()
