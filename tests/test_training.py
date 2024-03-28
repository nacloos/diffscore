from diffscore.training import NGymLitTrainer
from diffscore import Env
from diffscore.nn import CTRNN, IterateInput


def test_trainer():
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
    test_trainer()
