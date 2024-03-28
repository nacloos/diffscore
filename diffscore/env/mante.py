import numpy as np

import neurogym as ngym
from neurogym.utils import spaces

from diffscore import register


def _get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


def _gaussianbump(loc, theta, strength):
    dist = _get_dist(loc - theta)  # periodic boundary
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2) * strength


class _DMFamily(ngym.TrialEnv):
    """Delay comparison.

    Two-alternative forced choice task in which the subject
    has to compare two stimuli separated by a delay to decide
    which one has a higher frequency.

    Modified from original neurogym task to include context in trial kwargs.
    """
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, cohs=None,
                 dim_ring=16, w_mod=(1, 1), stim_mod=(True, True),
                 delaycomparison=True):
        super().__init__(dt=dt)

        # trial conditions
        if cohs is None:
            self.cohs = np.array([0.08, 0.16, 0.32])
        else:
            self.cohs = cohs
        self.w_mod1, self.w_mod2 = w_mod
        self.stim_mod1, self.stim_mod2 = stim_mod
        self.delaycomparison = delaycomparison

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if self.delaycomparison:
            self.timing = {
                'fixation': lambda: self.rng.uniform(200, 500),
                'stim1': 500,
                'delay': 1000,
                'stim2': 500,
                'decision': 200}
        else:
            self.timing = {
                'fixation': lambda: self.rng.uniform(200, 500),
                'stimulus': 500,
                'delay': 0,
                'decision': 200}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation space
        self.dim_ring = dim_ring
        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        if dim_ring < 2:
            raise ValueError('dim ring can not be smaller than 2')

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _add_singlemod(self, trial, mod=1, **kwargs):
        """Add stimulus to modality."""
        mod = '_mod' + str(mod)

        if self.delaycomparison:
            period1, period2 = 'stim1', 'stim2'
            coh1, coh2 = self.rng.choice(self.cohs, 2, replace=False)
            trial['coh1' + mod] = coh1
            trial['coh2' + mod] = coh2
        else:
            period1, period2 = 'stimulus', 'stimulus'
            if 'coh' + mod in kwargs:
                coh = kwargs['coh' + mod]
            else:
                coh = self.rng.choice(self.cohs) * self.rng.choice([-1, +1])
            # coh = coh1 - coh2
            trial['coh1' + mod] = coh1 = 0.5 + coh / 2
            trial['coh2' + mod] = coh2 = 0.5 - coh / 2

        # recompute coh1, coh2 in case they are specified in kwargs, it then overwrites kwargs['coh' + mod]
        # if 'coh1' + mod in kwargs or 'coh2' + mod in kwargs:
        trial.update(kwargs)
        coh1 = trial['coh1' + mod]
        coh2 = trial['coh2' + mod]
        # compute the coherence
        trial['coh' + mod] = coh1 - coh2

        stim = _gaussianbump(trial['theta1'], self.theta, coh1)
        self.add_ob(stim, period1, where='stimulus' + mod)
        stim = _gaussianbump(trial['theta2'], self.theta, coh2)
        self.add_ob(stim, period2, where='stimulus' + mod)

    def _new_trial(self, **kwargs):
        # if context is given in kwargs overwrite self.w_mod1 and self.w_mod2, i.e. which modality is relevant
        if 'context' in kwargs:
            if kwargs['context'] == 1:
                self.w_mod1, self.w_mod2 = 1, 0
            else:
                self.w_mod1, self.w_mod2 = 0, 1

        trial = {}
        i_theta1 = self.rng.choice(self.choices)
        while True:
            i_theta2 = self.rng.choice(self.choices)
            if i_theta2 != i_theta1:
                break
        trial['theta1'] = self.theta[i_theta1]
        trial['theta2'] = self.theta[i_theta2]

        # recompute i_theta1, i_theta2 in case kwargs specifies theta1, theta2
        trial.update(kwargs)
        i_theta1 = np.argwhere(np.abs(self.theta - trial['theta1']) < 1e-10)[0][0]
        i_theta2 = np.argwhere(np.abs(self.theta - trial['theta2']) < 1e-10)[0][0]

        # Periods
        if self.delaycomparison:
            periods = ['fixation', 'stim1', 'delay', 'stim2', 'decision']
        else:
            periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision')
        if self.delaycomparison:
            self.add_randn(0, self.sigma, ['stim1', 'stim2'])
        else:
            self.add_randn(0, self.sigma, ['stimulus'])

        coh1, coh2 = 0, 0
        if self.stim_mod1:
            self._add_singlemod(trial, mod=1, **kwargs)
            coh1 += self.w_mod1 * trial['coh1_mod1']
            coh2 += self.w_mod1 * trial['coh2_mod1']
        if self.stim_mod2:
            self._add_singlemod(trial, mod=2, **kwargs)
            coh1 += self.w_mod2 * trial['coh1_mod2']
            coh2 += self.w_mod2 * trial['coh2_mod2']

        # target = theta1 if coh = coh1 - coh2 > 0 else theta2
        i_target = i_theta1 if coh1 + self.rng.uniform(-1e-6, 1e-6) > coh2 else i_theta2
        self.set_groundtruth(i_target, period='decision', where='choice')

        trial['context'] = self.w_mod1 - self.w_mod2  # mod1 = context 1, mod2 = context -1
        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt, 'trial_info': self.trial}


class CtxdmEnv(_DMFamily):
    """
    Contextual decision making task. Randomly switch the context and add it as an additional input.
    """
    def __init__(self, dt=100, timing=None, sigma=1.0, cohs=None, dim_ring=2, rewards=None, abort=False, seq_len=None):
        super().__init__(dt=dt, timing=timing, sigma=sigma, cohs=cohs, dim_ring=dim_ring, delaycomparison=False,
                         rewards=rewards)
        self.abort = abort
        self.seq_len = seq_len
        # add an observation for the context
        obs_shape, obs_name = self.observation_space.shape, self.observation_space.name
        obs_name.update({"context": [obs_shape[0], obs_shape[0]+1]})
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_shape[0]+2,),
                                            dtype=self.observation_space.dtype,
                                            name=obs_name)

    def _new_trial(self, **kwargs):
        # choose the context randomly if not given
        if "context" not in kwargs:
            kwargs["context"] = self.rng.choice([1, -1])

        trial = super()._new_trial(**kwargs)

        if trial["context"] == 1:
            context = [1, 0]
        else:
            context = [0, 1]
        self.add_ob(context, where="context")
        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        done = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if not self.in_period('decision'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
                if self.abort:
                    done = True
        else:
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, done, {'new_trial': new_trial, 'gt': gt, 'trial_info': self.trial}


@register("env.mante")
def env_mante():
    dt = 50
    env = CtxdmEnv(
        dt=dt,
        dim_ring=2,
        seq_len=2000//dt,
        cohs=[0.06, 0.17, 0.5],
        timing={
            "fixation": 350,
            "stimulus": 750,
            "delay": ["truncated_exponential", [300, 0, 3000]],
            "decision": 300
        }
    )
    return env


@register("env.mante-test")
def env_mante_test():
    dt = 50
    env = CtxdmEnv(
        dt=dt,
        dim_ring=2,
        seq_len=2000//dt,
        cohs=[0.06, 0.17, 0.5],
        timing={
            "fixation": 350,
            "stimulus": 750,
            "delay": 300,
            "decision": 300
        }
    )
    return env
