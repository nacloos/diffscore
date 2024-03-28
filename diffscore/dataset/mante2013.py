from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d


file_path = Path(__file__).parent.resolve()
DOWNLOAD_DIR = file_path / Path("data/mante2013")
DATA_DIR1 = DOWNLOAD_DIR / Path("PFC data/PFC data 1")
DATA_DIR2 = DOWNLOAD_DIR / Path("PFC data/PFC data 2")

# use one session as the reference for the trial info (sessions can have different levels of coherence but map them to the same reference)
REF_FILE1 = "ar090313_1_a1_Vstim_100_850_ms.mat"
REF_FILE2 = "fe111019_2_a1_Vstim_100_850_ms.mat"


def download(save_path):
    def download_and_unzip(url, extract_to='.'):
        # https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
        print("Downloading: {}".format(url))
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)

    url = "https://www.ini.uzh.ch/dam/jcr:ca4213cf-1692-4c3d-8aeb-d4c5081d2fd1/PFC%20data.zip"
    download_and_unzip(url, extract_to=save_path)


def downsample(activity, n_samples):
    # activity: ... x n_timesteps x ...
    n_steps = activity.shape[1]
    assert n_samples <= n_steps

    window = int(n_steps / n_samples)
    resampled_act = np.zeros((activity.shape[0], n_samples, *activity.shape[2:]))
    for k in range(n_samples):
        # average within the temporal window
        resampled_act[:, k] = activity[:, k * window:(k + 1) * window].mean(axis=1)
    return resampled_act


def process_data(subject: str = "ar", data_dir: str | Path = None, dt: int = 50):
    def _process_unit(unit):
        """
        Args:
            unit: struct load from .mat file

        Returns:
            dt: sampling timestep
            cond_avg_activity: n_conditions x n_steps array, condition-averaged activity of the unit
            trials_info: list of dict containing the condition info for each row of activity
        """
        unit_dt = (unit['time'][1] - unit['time'][0])*1000  # ms
        task_var = unit['task_variable']
        activity = unit['response']  # n_trials x n_steps

        mante_var_names = ['targ_dir', 'stim_dir', 'stim_col2dir', 'context']
        new_names = ['gt_choice', 'coh_mod1', 'coh_mod2', 'context']

        trials_by_condition = defaultdict(list)  # trials sorted by condition
        # loop through all trials
        for trial in range(task_var['stim_trial'][-1]):
            condition = tuple([task_var[name][trial] for name in mante_var_names])
            trials_by_condition[condition].append(trial)

        conditions, cond_trials = list(trials_by_condition.keys()), list(trials_by_condition.values())
        # sort conditions to have the same order for all units
        k = np.empty(len(trials_by_condition), dtype=object)
        k[:] = conditions  # array of tuple
        indices = np.argsort(k, axis=0)

        cond_avg_activity = []  # n_conditions x n_steps
        trials_info = []
        for idx in indices:
            cond, trials = conditions[idx], cond_trials[idx]

            if not np.all(task_var['correct'][trials]):
                # skip incorrect trials (the task var 'correct' is the same for all trials with the same condition)
                continue
            cond_avg_activity.append(activity[trials].mean(axis=0))

            trial_info = {k: v for k, v in zip(new_names, cond)}
            trials_info.append(trial_info)

        cond_avg_activity = np.array(cond_avg_activity)

        # resampling
        n_samples = int(cond_avg_activity.shape[-1]*unit_dt/dt)
        cond_avg_activity = downsample(cond_avg_activity, n_samples)

        return dt, cond_avg_activity, trials_info


    if subject is not None:
        if subject == "ar":
            DATA_DIR = DATA_DIR1
        elif subject == "fe":
            DATA_DIR = DATA_DIR2
        else:
            raise ValueError

    # download the data if data_dir doesn't exist
    data_dir = DATA_DIR if data_dir is None else data_dir
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    if not data_dir.exists():
        download(DOWNLOAD_DIR)

    ref_file = REF_FILE1 if data_dir.name == "PFC data 1" else REF_FILE2

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    ref_file = data_dir / ref_file
    # not all units have the same levels of coh, use one session as the reference for the trial info
    ref_mat = spio.loadmat(str(ref_file), simplify_cells=True)
    trials_info = _process_unit(ref_mat['unit'])[-1]

    _dt = None
    activity = []
    for i, file_path in enumerate(data_dir.iterdir()):
        mat = spio.loadmat(str(file_path), simplify_cells=True)
        unit = mat['unit']
        unit_dt, unit_activity, unit_trials_info = _process_unit(unit)

        # skip when don't have all conditions recorded
        if unit_activity.shape[0] != 72:
            continue

        activity.append(unit_activity)

        # assert all units have the same dt
        _dt = unit_dt if _dt is None else _dt
        assert _dt == unit_dt
        assert _dt == dt

    activity = np.transpose(np.array(activity), (2, 1, 0))  # n_steps x n_trials x n_neurons

    for trial_info in trials_info:
        # remap choice
        if trial_info['gt_choice'] == -1:
            trial_info['gt_choice'] = 2

    mante_timing = {
        'fixation': 350,
        'stimulus': 330,
        'decision': 300
    }
    regions = ["PFC"]*activity.shape[-1]
    return {
        "dt": dt,
        "activity": activity,
        "trial_info": trials_info,  # backward compatibility
        "conditions": trials_info,
        "trial_timing": [mante_timing]*len(trials_info),
        "region": regions,
        "target": [trial_info['gt_choice'] for trial_info in trials_info]
    }


class TDR:
    def __init__(self, dt, smooth_std=40, pca_components=12, reg_time_windows=None):
        super().__init__()
        self.dt = dt
        self.smooth_std = smooth_std 
        self.reg_time_windows = reg_time_windows
        self.scaler = StandardScaler()
        self.lin_reg = LinearRegression()
        self.denoising_pca = PCA(n_components=pca_components)

    def _normalize(self, values, normalization_type, axis=0):
        """
        Args:
            values: 2-dim array
            normalization_type: str

        Returns:
            normalized values
        """
        if normalization_type == 'zscore':
            # some neurons have zero activity
            values_std = values.std(axis=axis)
            values_std[values_std == 0] = 1
            norm_values = (values - values.mean(axis=axis)) / values_std
        elif normalization_type == 'max_abs':
            norm_values = values / np.max(np.abs(values), axis=axis)
        else:
            raise NotImplementedError
        return norm_values

    def _agg_regression_vectors(self, Bs, agg_type, time_windows=None):
        """
        Args:
            Bs: (var, time, neuron) array
            agg_type: how the regression vectors are aggregated along the time component

        Returns:
            (var, neuron) array
        """
        n_vars, n_steps, n_neurons = Bs.shape
        if agg_type == 'max_norm':
            B = np.zeros((n_vars, n_neurons))
            for nu in range(n_vars):
                Bnorms = np.linalg.norm(Bs[nu], axis=-1)
                B[nu] = Bs[nu, np.argmax(Bnorms)]

        elif agg_type == 'average':
            if time_windows is not None:

                B = np.zeros((n_vars, n_neurons))
                masks = list(time_windows.values())
                for i in range(n_vars):
                    B[i] = np.mean(Bs[i, masks[i]], axis=0)
            else:
                B = Bs.mean(axis=1)
        else:
            raise NotImplementedError
        return B

    def get_activity_by_condition(self, activity, trial_conditions, conditions, average=False):
        """
        Sort the trials by condition

        Args:
            activity: n_steps x n_trials x n_neurons
            trial_conditions: list of dict containing the condition info for each trial
            conditions: list of dict, each dict contains the condition info to filter the trials
            average: whether to average the activity across trials in each condition

        Returns:
            act_by_cond: list of length len(conditions) of arrays of size
                n_steps x n_trials_in_cond x n_neurons if average, otherwise n_steps x n_neurons
        """
        def _get_condition_trials(condition: dict):
            n_timesteps, n_trials, n_neurons = activity.shape
            mask = np.ones(n_trials)
            for cond_key, cond_val in condition.items():
                info_val = [trial_info[cond_key] for trial_info in trial_conditions]

                if isinstance(cond_val, (np.ndarray, list)):
                    # the value of the condition is a list (e.g. one hot task rule)
                    cond_mask = (info_val == np.array(cond_val)).all(axis=1)
                else:
                    if np.array(cond_val).dtype == float:
                        # allow some negligible differences
                        cond_mask = np.abs(info_val - np.array(cond_val)) < 1e-8
                    else:
                        cond_mask = info_val == np.array(cond_val)

                mask = np.logical_and(mask, cond_mask)
            return np.where(mask)[0]

        act_by_cond = []
        for c in conditions:
            # n_steps x n_trials_in_cond x n_neurons
            trial_indices = _get_condition_trials(c)
            act = activity[:, trial_indices]

            assert act.shape[1] > 0
            if average:
                act = act.mean(axis=1)
            act_by_cond.append(act)
        return act_by_cond

    def fit(self, activity, trial_conditions, task_vars, var_names, conditions: list[dict] = None):
        """
        Args:
            activity: n_steps x n_trials x n_neurons
            trial_conditions: list of dict containing the condition info for each trial
            task_vars: n_steps x n_trials x n_vars
            var_names: list of str
            conditions: list of dict, each dict contains the condition info to filter the trials
        """
        self.trial_conditions = trial_conditions
        self.conditions = conditions
        self.var_names = np.array(var_names)

        r = activity  # n_steps x n_trials x n_neurons
        task_var = task_vars  # n_steps x n_trials x n_vars
        n_steps, n_trials, n_neurons = r.shape
        n_vars = task_var.shape[-1]

        # z-score neural response neuron by neuron
        r = self._normalize(r.reshape(n_steps*n_trials, -1), "zscore")
        r = r.reshape(n_steps, n_trials, n_neurons)

        # normalize task variables
        task_var = self._normalize(task_var.reshape(n_steps*n_trials, -1), "max_abs")
        task_var = task_var.reshape(n_steps, n_trials, n_vars)

        # linear regression of neural activity on task vars
        r = np.transpose(r, (1, 0, 2))  # n_trials x n_steps x n_neurons
        task_var = np.transpose(task_var, (1, 0, 2))  # n_trials x n_steps x n_vars
        B = np.zeros((n_vars, n_steps, n_neurons))
        for i in range(n_neurons):
            for t in range(n_steps):
                self.lin_reg.fit(task_var[:, t], r[:, t, i])
                B[:, t, i] = self.lin_reg.coef_

        # condition-averaged (optional), smoothed, z-score neural response
        if conditions is not None:
            self.n_traj = len(conditions)
            X = self.get_activity_by_condition(activity, trial_conditions, conditions, average=True)  # n_conditions x n_steps x n_neurons
            X = np.transpose(X, (1, 0, 2)).reshape(n_steps, -1)  # n_steps x n_conditions*n_neurons
        else:
            X = activity
            self.n_traj = X.shape[1]
            X = X.reshape(n_steps, -1)  # n_steps x n_trials*n_neurons

        X = gaussian_filter1d(X, self.smooth_std / self.dt, axis=0)
        X = X.reshape(n_steps * self.n_traj, n_neurons)
        X = self.scaler.fit_transform(X)

        # don't seem to be required
        # de-noising pca
        # X = self.denoising_pca.fit_transform(X)
        # B = self.denoising_pca.transform(B.reshape(-1, n_neurons)).reshape(n_vars, n_steps, -1)
        # B = gaussian_filter1d(B, self.smooth_std/self.dt, axis=1)

        self.B_temporal = B

        if self.reg_time_windows:
            B = self._agg_regression_vectors(B, agg_type='average', time_windows=self.reg_time_windows)
        else:
            # max norm time-independent regression vectors
            B = self._agg_regression_vectors(B, agg_type='max_norm')

        Borth, R = np.linalg.qr(B.T)
        self.B = Borth.T

        self.X = X.reshape(n_steps, self.n_traj, -1)
        self.proj_X = np.tensordot(self.X, self.B, axes=(2, 1))  # n_steps x n_conditions x n_vars

    def visualize(self, var1_label, var2_label, ax=None, cond_kwargs=None):
        if ax is None:
            plt.figure(figsize=(4, 3), dpi=130)
            ax = plt.gca()

        assert var1_label in self.var_names, "Expected {} to be in {}".format(var1_label, self.var_names)
        assert var2_label in self.var_names, "Expected {} to be in {}".format(var2_label, self.var_names)
        var1_i = np.argwhere(self.var_names == var1_label)[0]
        var2_i = np.argwhere(self.var_names == var2_label)[0]

        for c_idx, c in enumerate(self.conditions):
            kwargs = cond_kwargs[c_idx] if cond_kwargs is not None else {}
            plt.plot(self.proj_X[:, c_idx, var1_i], self.proj_X[:, c_idx, var2_i], label=c, marker=".",
                     **kwargs)

        plt.xlabel(var1_label)
        plt.ylabel(var2_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    def visualize_coef_norm(self, save_path=None):
        plt.figure(figsize=(4, 3), dpi=130)
        n_vars = self.B_temporal.shape[0]
        for i in range(n_vars):
            norm = np.linalg.norm(self.B_temporal[i], axis=-1)
            plt.plot(norm, label=self.var_names[i])
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)


def mante2013_tdr(activity, trial_conditions, dt):
    """
    Non-official implementation of the Targeted Dimensionality Reduction analysis from Mante et al. 2013.
    This is a simplified version of the original analysis. It is mainly used to make sure the data processing is correct.
    Use caution when using this implementation for other purposes.

    Args:
        activity: n_steps x n_trials x n_neurons
        trial_conditions: list of dict containing the condition info for each trial
        dt: sampling timestep
    """
    trial_info_labels = {
        "Choice": "gt_choice",
        "CohMod1": "coh_mod1",
        "CohMod2": "coh_mod2",
        "Context": "context"
    }

    # done in the original analysis but don't do it here for simplification
    # time_windows = {
    #     "Choice": [600, 900],
    #     "CohMod1": [250, 450],
    #     "CohMod2": [350, 550],
    #     "Context": [100, 900]
    # }
    # time = dt*np.arange(15) + 125
    # time_masks = {name: np.logical_and(lower < time, time < upper) for name, (lower, upper) in time_windows.items()}
    # tdr = TDR(reg_time_windows=time_masks)

    tdr = TDR(dt=dt)

    tdr_results = []
    n_steps = activity.shape[0]
    gt_choice_key, coh_mod1_key, coh_mod2_key, context_key = [
        trial_info_labels[name] for name in ["Choice", "CohMod1", "CohMod2", "Context"]
    ]

    # create task dataset
    task_vars_dict = {
        name: np.array([trial_info[label] for trial_info in trial_conditions]) for name, label in trial_info_labels.items()
    }
    var_names = list(task_vars_dict.keys())

    # original analysis also includes interaction terms
    task_vars_dict["Choice*Context"] = task_vars_dict["Choice"]*task_vars_dict["Context"]
    task_vars_dict["CohMod1*Context"] = task_vars_dict["CohMod1"]*task_vars_dict["Context"]
    task_vars_dict["CohMod2*Context"] = task_vars_dict["CohMod2"]*task_vars_dict["Context"]

    task_vars = np.stack(list(task_vars_dict.values()), axis=1)
    task_vars = np.repeat(task_vars[np.newaxis, :, :], n_steps, axis=0)

    # make the plots
    gt_choice = np.array([trial_info[gt_choice_key] for trial_info in trial_conditions])
    coh_mod1 = np.array([trial_info[coh_mod1_key] for trial_info in trial_conditions])
    coh_mod2 = np.array([trial_info[coh_mod2_key] for trial_info in trial_conditions])

    # colors
    if len(np.unique(coh_mod1)) == 6:
        coh_mod1_colors = ["black", "grey", "lightgrey", "lightgrey", "grey", "black"]
        coh_mod2_colors = ["#3d5a80", "#98c1d9", "#D1E9F0", "#D1E9F0", "#98c1d9", "#3d5a80"]
        coh_mfc = ["white", "white", "white", None, None, None]
    else:
        coh_mod1_colors = [None]*len(np.unique(coh_mod1))
        coh_mod2_colors = [None]*len(np.unique(coh_mod2))
        coh_mfc = [None]*len(np.unique(coh_mod1))

    kwargs_coh1 = []
    kwargs_coh2 = []
    for i, v in enumerate(np.unique(coh_mod1)):
        kwargs_coh1.append({"color": coh_mod1_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})
        kwargs_coh2.append({"color": coh_mod2_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})

    kwargs_coh1_choice = []
    kwargs_coh2_choice = []
    for i, v in enumerate(np.unique(coh_mod1)):
        for j, v2 in enumerate(np.unique(gt_choice)):
            kwargs_coh1_choice.append(
                {"color": coh_mod1_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})
            kwargs_coh2_choice.append(
                {"color": coh_mod2_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})

    def _context_plots(context):
        assert context == 1 or context == -1
        row = 0 if context == 1 else 1

        if context == 1:
            conditions = [{coh_mod1_key: v, context_key: context} for v in np.unique(coh_mod1)]
            cond_kwargs = kwargs_coh1
        else:
            conditions = [
                {coh_mod1_key: v, gt_choice_key: v2, context_key: context}
                for v in np.unique(coh_mod1) for v2 in np.unique(gt_choice)
            ]
            cond_kwargs = kwargs_coh1_choice

        tdr.fit(activity, trial_conditions, task_vars, var_names, conditions)

        ax = plt.subplot(2, 3, 3 * row + 1)
        tdr.visualize("Choice", "CohMod1", ax=ax, cond_kwargs=cond_kwargs)
        tdr_results.append(tdr.proj_X)

        ax = plt.subplot(2, 3, 3 * row + 2, sharey=ax, sharex=ax)
        plt.title("{} context ".format("Mod1" if context == 1 else "Mod2"))
        if context == 1:
            tdr.visualize("Choice", "CohMod2", ax=ax, cond_kwargs=cond_kwargs)
            tdr_results.append(tdr.proj_X)

            conditions = [
                {coh_mod2_key: v, gt_choice_key: v2, context_key: context}
                for v in np.unique(coh_mod2) for v2 in np.unique(gt_choice)
            ]
            cond_kwargs = kwargs_coh2_choice
        else:
            conditions = [{coh_mod2_key: v, context_key: context} for v in np.unique(coh_mod2)]
            cond_kwargs = kwargs_coh2
            tdr.fit(activity, trial_conditions, task_vars, var_names, conditions)
            tdr.visualize("Choice", "CohMod1", ax=ax, cond_kwargs=cond_kwargs)
            tdr_results.append(tdr.proj_X)

        tdr.fit(activity, trial_conditions, task_vars, var_names, conditions)
        ax = plt.subplot(2, 3, 3 * row + 3, sharey=ax, sharex=ax)
        tdr.visualize("Choice", "CohMod2", ax=ax, cond_kwargs=cond_kwargs)
        tdr_results.append(tdr.proj_X)

    plt.figure(figsize=(6, 4), dpi=180)
    _context_plots(1)
    _context_plots(-1)


if __name__ == "__main__":
    data = process_data()
    print("Activity shape:", data["activity"].shape)

    mante2013_tdr(data["activity"], data["trial_info"], data["dt"])
    plt.show()
