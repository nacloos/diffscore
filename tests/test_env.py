from diffscore import Env


def test_mante():
    env = Env("mante")
    # test new trial
    env.new_trial()

    test_env = Env("mante-test")
    # test new trial
    test_env.new_trial()


if __name__ == "__main__":
    test_mante()
