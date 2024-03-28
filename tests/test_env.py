from diffscore import Env


def test_mante13():
    env = Env("mante13")
    # test new trial
    env.new_trial()

    test_env = Env("mante13-test")
    # test new trial
    test_env.new_trial()


if __name__ == "__main__":
    test_mante13()
