from copy import deepcopy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import neurogym as ngym


class NGymLitTrainer:
    """
    Trainer for neurogym environments using pytorch lightning.

    Args:
        loss_fn: loss function, e.g. MSELoss, CrossEntropyLoss, etc.
        optimizer: optimizer, e.g. Adam, SGD, etc.
        accuracy: accuracy function, e.g. accuracy, f1_score, etc.
        batch_first: True if batch is the first dim in data
        batch_size: batch size
        n_samples: number of samples in the dataset
        target_as_input: some models requires target as additional input
        loss_model_arg: add model input when calling loss function
        checkpoint_path: path to save checkpoints
        save_every_n_train_steps: save checkpoint every n train steps
        state_norm_reg: regularize the norm of the state
        max_reward: use max reward as loss
        online_dataset: generate dataset online
        num_workers: number of workers for dataloader
        **kwargs: additional arguments for pytorch lightning trainer

    """
    def __init__(
        self,
        loss_fn: str = "MSELoss",
        optimizer=optim.Adam,
        accuracy=None,
        batch_first: bool = False,
        batch_size: int = 64,
        n_samples: int = int(10e3),
        target_as_input=False,
        loss_model_arg=False,
        checkpoint_path=None,
        save_every_n_train_steps=None,
        state_norm_reg=None,
        max_reward=False,
        online_dataset=True,
        num_workers=0,
        **kwargs
    ):
        # raises a rather noninformative error if not an int
        assert isinstance(n_samples, int)
        assert isinstance(batch_size, int)

        if isinstance(loss_fn, str):
            loss_fn = getattr(nn, loss_fn)()

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.batch_first = batch_first
        self.target_as_input = target_as_input
        self.loss_model_arg = loss_model_arg
        self.state_norm_reg = state_norm_reg
        self.max_reward = max_reward
        self.online_dataset = online_dataset
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path

        # change model checkpoint path
        callbacks = kwargs.get('callbacks', [])
        if checkpoint_path is not None:
            # change log dir
            kwargs["default_root_dir"] = checkpoint_path

            for callback in callbacks:
                assert not isinstance(callback, ModelCheckpoint), "Cannot change checkpoint_point while providing ModelCheckpoint callback."

            # save init checkpoint callback
            callbacks.append(
                InitialModelCheckpoint(
                    checkpoint_path / "ckpt" / "step=0.ckpt"
                )
            )

            # save best checkpoint callback
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_path,
                    filename='best',
                    monitor='val_accuracy',
                    mode='max',
                    save_top_k=1,
                )
            )
            if save_every_n_train_steps is not None:
                # save every n train steps callback
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=checkpoint_path / 'ckpt',
                        filename='{step}',
                        every_n_train_steps=save_every_n_train_steps,
                        save_top_k=-1,
                    )
                )

        kwargs['callbacks'] = callbacks

        self.trainer = pl.Trainer(**kwargs)

    def _make_lit_module(self, model):
        lit_module = LitModule(
            model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            accuracy=self.accuracy,
            batch_first=self.batch_first,
            target_as_input=self.target_as_input,
            loss_model_arg=self.loss_model_arg,
            state_norm_reg=self.state_norm_reg,
            max_reward=self.max_reward
        )
        return lit_module

    def _make_dataloader(self, env):
        if self.online_dataset:
            dataset = NGymOnlineDataset(env, n_samples=self.n_samples)
        else:
            dataset = NGymDataset(env, n_samples=self.n_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        return dataloader

    def fit(self, model, env, val_env=None):
        self.lit_module = self._make_lit_module(model)
        self.dataloader = self._make_dataloader(env)
        if val_env is not None:
            self.val_dataloader = self._make_dataloader(val_env)
        else:
            self.val_dataloader = None

        self.trainer.fit(
            self.lit_module,
            train_dataloaders=self.dataloader,
            val_dataloaders=self.val_dataloader
        )

    def load(self, model, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_dir():
            ckpt_files = list(ckpt_path.glob("*.ckpt"))
            if len(ckpt_files) == 1:
                ckpt_path = ckpt_files[0]
            elif len(ckpt_files) == 0:
                raise ValueError(f"No .ckpt file found in folder: {ckpt_path}")
            else:
                raise ValueError(f"Found more than one .ckpt file in folder: {ckpt_path}. Please specify which one to load.")

        print(f"Load {ckpt_path}")
        self.lit_module = LitModule.load_from_checkpoint(ckpt_path, module=model, loss_fn=self.loss_fn, optimizer=self.optimizer, accuracy=self.accuracy, batch_first=self.batch_first)

    def load_checkpoints(self, model, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        ckpt_modules = []
        ckpt_names = []
        ckpt_steps = []
        for ckpt_file in ckpt_files:
            print(f"Load {ckpt_file}")
            ckpt_name = ckpt_file.stem
            ckpt_step = int(ckpt_name.split("=")[-1])

            ckpt_module = LitModule.load_from_checkpoint(ckpt_file, module=model, loss_fn=self.loss_fn, optimizer=self.optimizer, accuracy=self.accuracy, batch_first=self.batch_first)
            # have to deepcopy because otherwise the model is modified when loading the next checkpoint
            ckpt_module = deepcopy(ckpt_module)

            # alternative way to load ckpt
            # _model = deepcopy(model)
            # state_dict = torch.load(ckpt_file)
            # have to rename state dict keys to remove module. prefix
            # state_dict["state_dict"] = {k.replace("module.", ""): v for k, v in state_dict["state_dict"].items()}
            # _model.load_state_dict(state_dict["state_dict"])

            ckpt_modules.append(ckpt_module)
            ckpt_names.append(ckpt_name)
            ckpt_steps.append(ckpt_step)

        # sort by step
        ckpt_modules = [x for _, x in sorted(zip(ckpt_steps, ckpt_modules))]
        ckpt_names = [x for _, x in sorted(zip(ckpt_steps, ckpt_names))]
        ckpt_steps = sorted(ckpt_steps)

        return {
            "modules": ckpt_modules,
            "names": ckpt_names,
            "steps": ckpt_steps,
        }

    def predict(self, model, inputs=None, targets=None, env=None, n_samples=None, condition=None, return_data=False):
        """
        Args:
            inputs: optional, (seq_len, batch_size, input_size) if batch_first is False, (batch_size, seq_len, input_size) otherwise
            targets: optional, (seq_len, batch_size, output_size) if batch_first is False, (batch_size, seq_len, output_size) otherwise
        """
        if not isinstance(model, LitModule):
            model = self._make_lit_module(model)

        if inputs is None:
            assert env is not None, "Please specify either data or env"
            # assert n_samples is not None, "Please specify n_samples"

            # TODO: use ligthning trainer.predict? (e.g. to handle gpu)
            # self.dataset = NGymDataset(env, n_samples=n_samples, condition=condition)
            # x, y = self.dataset.x, self.dataset.y
            dataloader = self._make_dataloader(env)
            # output of dataloader is batch first
            x, y = next(iter(dataloader))
        else:
            x = inputs
            y = targets
            if not self.batch_first:
                # when batch_first is False, inputs are (seq_len, batch_size, input_size)
                # but model expects (batch_size, seq_len, input_size)
                x = torch.transpose(x, 1, 0)
                y = torch.transpose(y, 1, 0)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.float)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).type(torch.float)
        x, y = x.float(), y.float()

        # TODO: move that in LITModule? because need this to compute accuracy
        # TODO: use lit module predict (otherwise error that model is on gpu and data on cpu)
        model.to('cpu')
        # if self.target_as_input:
        #     action_pred = model(x, y)
        # else:
        #     action_pred = model(x)

        # model expects (batch_size, seq_len, input_size) (what the dataloader outputs)
        action_pred = model.predict_step((x, y), 0)

        if return_data:
            if inputs is None and self.batch_first is False:
                # data generated by dataloader is (batch_size, seq_len, input_size)
                # if batch_first is False, need to transpose to (seq_len, batch_size, input_size)
                x = torch.transpose(x, 0, 1)
                y = torch.transpose(y, 0, 1)

            return x, y, action_pred
        else:
            return action_pred


class LitModule(pl.LightningModule):
    """
    Convert a torch.nn.Module to a pytorch lightning module.
    """
    def __init__(
            self,
            module: nn.Module,
            loss_fn,
            optimizer,
            accuracy=None,
            batch_first: bool = False,
            target_as_input: bool = False,
            loss_model_arg: bool = False,
            state_norm_reg: float = None,
            max_reward: bool = False,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.module = module
        self.loss_fn = loss_fn
        self.accuracy = accuracy
        self.optimizer = optimizer
        self.batch_first = batch_first  # True if batch is the first dim in data
        self.target_as_input = target_as_input  # some models requires target as additional input
        self.loss_model_arg = loss_model_arg  # add model input when calling loss function
        self.state_norm_reg = state_norm_reg
        self.max_reward = max_reward

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def forward(self, x, y=None):
        if y is not None:
            out = self.module(x, y)
        else:
            out = self.module(x)
        return out

    def _step(self, x, y, return_loss=True):
        """
        x: (batch_size, seq_len, input_size)
        y: (batch_size, seq_len, output_size)
        """
        x, y = x.float(), y.float()
        if not self.batch_first:
            # transpose to (seq_len, batch_size, input_size)
            x = torch.transpose(x, 1, 0)
            y = torch.transpose(y, 1, 0)

        if self.target_as_input:
            out = self.forward(x, y)
        else:
            out = self.forward(x)

        if not return_loss:
            return out

        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_ts = torch.as_tensor(y.flatten(), dtype=torch.long)
            loss = self.loss_fn(out.reshape(-1, out.shape[-1]), y_ts)
        else:
            if self.loss_model_arg:
                loss = self.loss_fn(out, y, model=self.module)
            else:
                loss = self.loss_fn(out, y)

        # TODO: use custom loss function
        if self.max_reward:
            assert hasattr(self.module, "rewards"), "Expecting module to have a 'rewards' attribute when using max_reward"
            loss = -torch.mean(self.module.rewards)

        if self.state_norm_reg is not None:
            assert hasattr(self.module, "states"), "Expecting module to have a 'states' attribute when using state_norm_reg"
            states = self.module.states
            loss += self.state_norm_reg * torch.mean(torch.norm(states, dim=-1))

        if self.accuracy is not None:
            pred = np.argmax(out.detach().cpu().numpy(), axis=-1).squeeze()
            acc = self.accuracy(x.cpu(), y.cpu(), pred)
        else:
            acc = None

        return out, loss, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, loss, acc = self._step(x, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        if acc is not None:
            self.log("train_accuracy", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, loss, acc = self._step(x, y)

        self.log("val_loss", loss, on_step=True, prog_bar=True)
        if acc is not None:
            self.log("val_accuracy", acc, on_step=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        out = self._step(x, y, return_loss=False)

        action_pred = out.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1).squeeze()
        return action_pred


class NGymDataLoader(DataLoader):
    def __init__(self, env, batch_size):
        dataset = NGymOnlineDataset(env, n_samples=batch_size)
        super().__init__(dataset, batch_size=batch_size)


class NGymDataset(torch.utils.data.Dataset):
    def __init__(self, env, n_samples, condition=None, seq_len=None):
        if seq_len is None:
            assert hasattr(env, "seq_len"), "Please specify the seq_len of a trial"
            seq_len = env.seq_len

        # the for loop over batch in ngym.Dataset could be parallelized
        # import neurogym as ngym
        print("Generating task dataset...")
        dataset = ngym.Dataset(env, batch_size=n_samples, seq_len=seq_len, cache_len=seq_len)

        if condition:
            self.x, self.y = dataset(**condition)
        else:
            self.x, self.y = dataset()

        self.n_samples = n_samples
        assert self.x.shape[1] == self.n_samples
        assert self.y.shape[1] == self.n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[:, idx], self.y[:, idx]


class NGymOnlineDataset(torch.utils.data.Dataset):
    def __init__(self, env, n_samples, condition=None, seq_len=None):
        if seq_len is None:
            assert hasattr(env, "seq_len"), "Please specify the seq_len of a trial"
            seq_len = env.seq_len
        self.seq_len = seq_len
        self.n_samples = n_samples  # size of the dataset (don't matter so much because can generate as many trials as needed)
        self.env = env
        self.condition = {} if condition is None else condition
        if not isinstance(self.condition, dict):
            assert len(self.condition) == self.__len__(), "If condition is not a dict, it should be a list of conditions with the same length as the dataset"

        # get ob and gt shapes
        env.new_trial()
        self.ob_shape = env.ob.shape
        self.gt_shape = env.gt.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = np.zeros((self.seq_len, self.ob_shape[-1]))
        if len(self.gt_shape) == 1:
            y = np.zeros((self.seq_len))
        else:
            y = np.zeros((self.seq_len, self.gt_shape[-1]))

        if isinstance(self.condition, dict):
            condition = self.condition
        else:
            condition = self.condition[idx]

        # generate a new sample everytime
        seq_start = 0
        seq_end = 0
        while seq_end < self.seq_len:
            self.env.new_trial(**condition)
            ob, gt = self.env.ob, self.env.gt
            seq_len = ob.shape[0]
            assert gt.shape[0] == seq_len
            seq_end = seq_start + seq_len
            if seq_end > self.seq_len:
                seq_end = self.seq_len
                seq_len = seq_end - seq_start

            x[seq_start:seq_end, ...] = ob[:seq_len]
            y[seq_start:seq_end, ...] = gt[:seq_len]

            seq_start = seq_end

        return x, y


class InitialModelCheckpoint(pl.Callback):
    """
    Small callback to save the initial model.
    """
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def on_train_start(self, trainer, pl_module):
        trainer.save_checkpoint(self.checkpoint_path)
