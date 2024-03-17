from pathlib import Path

import numpy as np
import scipy.io as spio

# dryad: https://datadryad.org/stash/dataset/doi:10.5061/dryad.xsj3tx9cm

# original links
# DATA_LINK = "https://datadryad.org/stash/downloads/file_stream/397190",
# other monkey data (not used)
# DATA_LINK = "https://datadryad.org/stash/downloads/file_stream/397193"
# original data is in mat v7.3 format, which is not supported by scipy.io.loadmat
# use converted copy instead
DATA_LINK = "https://www.dropbox.com/scl/fi/u7b6t3wnx7sxowzec28gt/Dataset5_Monkey4_Session1_ReachDataSimultaneousRecording-v7.mat?rlkey=x45tm5poo0jlo6m34jfuf48cs&dl=1"

SAVE_DIR = Path(__file__).parent.resolve() / Path("data/hatsopoulos2007")


def download_data(save_dir=None, save_name=None):
    save_dir = save_dir or SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_name:
        file_name = save_name
    else:
        file_name = DATA_LINK.split("/")[-1]
    file_path = save_dir / file_name
    if not file_path.exists():
        print(f"Downloading {file_name}...")
        import requests
        response = requests.get(DATA_LINK)
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"{file_name} already exists.")
    print("Data downloaded.")


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


def bin_spike_times(spike_times, dt, bin_range=None):
    """
    Args:
        spike_times: list (neurons) of lists (spikes times for each neuron)
        dt: bin width
    Returns:
        n_neurons x n_bins np.ndarray of binned spikes
    """
    # TODO: why 0 is the min?
    bin_range = (0, max([times[-1] for times in spike_times])) if bin_range is None else bin_range
    n_bins = np.ceil(bin_range[1]/dt).astype(int)

    binned_spikes = []  # list of binned spikes for each neuron
    for neuron_spike_times in spike_times:
        binned_spikes.append(
            np.histogram(neuron_spike_times, n_bins, bin_range)[0]
        )
    binned_spikes = np.array(binned_spikes)
    return binned_spikes


def process_data(
    data_dir=None,
    file_name="Dataset5_Monkey4_Session1_ReachDataSimultaneousRecording-v7.mat",
    dt=50,
    condition_average=True,
    smooth_std=25,
    align_event='mvt_onset',
    align_range=(-500, 500),
    condition_label='target'
):
    """
    Args:
        data_dir: Path to the .mat data file
        dt: Discretization step in ms
        condition_average: Average trials of the same condition if true
        smooth_std: Gaussian filter std in ms
        align_event: Event relative to which the trials are aligned
        align_range: Start and end time relative to align_event
        condition_label: Label used to match trial_info with models
    """
    data_dir = Path(data_dir) if data_dir else Path(SAVE_DIR)
    data_path = data_dir / file_name
    if not data_path.exists():
        download_data(save_dir=data_dir, save_name=file_name)

    n_targets = 8  # 8 reaching targets = 8 conditions

    print("Start processing Hatsopoulos2007 data...")
    # not working for mat v7.3
    mat = spio.loadmat(str(data_path), simplify_cells=True)

    raw_spike_times = mat['spikes']*1000  # s to ms
    spike_times = []  # list of spikes times for each neuron
    for channel in raw_spike_times:
        if len(channel) == 0:
            continue
        if channel.dtype == 'object':
            for unit in channel:
                if len(unit) > 0:
                    spike_times.append(unit)
        else:
            spike_times.append(channel)

    binned_spikes = bin_spike_times(spike_times, dt)

    if align_event == 'mvt_onset':
        go_times = mat['stmv_cell']*1000  # take mvt onset as the go to account for the delay not modelled
    else:
        raise NotImplementedError

    trial_start = go_times + align_range[0]
    trial_stop = go_times + align_range[1]

    activity_list = []
    trial_timing = []
    trial_info = []
    hand_pos = []
    joint_angles = []

    for cond in range(n_targets):
        for trial in range(len(trial_start[cond])):
            start, stop = trial_start[cond][trial], trial_stop[cond][trial]
            # the stimulus is directly shown (no fixation period without any stimulus shown)
            fixation = 0
            stimulus = go_times[cond][trial] - start - fixation
            decision = stop - start - stimulus - fixation
            timing = {'fixation': fixation, 'stimulus': stimulus, 'delay': 0, 'decision': decision}
            trial_timing.append(timing)

            info = {condition_label: cond}
            trial_info.append(info)

            # activity
            act = binned_spikes[:, int(start/dt):int(stop/dt)]

            activity_list.append(act)

            # hand position (sampled at 2ms)
            x_time, hand_x = 1000*mat["x"][:, 0], mat["x"][:, 1]
            y_time, hand_y = 1000*mat["y"][:, 0], mat["y"][:, 1]

            trial_hand_x = hand_x[np.logical_and(x_time > start, x_time < stop)]
            trial_hand_y = hand_y[np.logical_and(y_time > start, y_time < stop)]
            hand_pos.append(np.stack([trial_hand_x, trial_hand_y]).T)

            # joint angles
            x_time, theta1 = 1000*mat["th1"][:, 0], mat["th1"][:, 1]
            y_time, theta2 = 1000*mat["th2"][:, 0], mat["th2"][:, 1]

            trial_theta1 = theta1[np.logical_and(x_time > start, x_time < stop)]
            trial_theta2 = theta2[np.logical_and(y_time > start, y_time < stop)]
            joint_angles.append(np.stack([trial_theta1, trial_theta2]).T)

    # downsample hand pos
    hand_pos = np.array(hand_pos)
    n_samples = int(hand_pos.shape[1]*2/dt)
    hand_pos = downsample(hand_pos, n_samples)
    hand_pos = np.transpose(hand_pos, (1, 0, 2))  # time x trial x position

    joint_angles = np.array(joint_angles)
    n_samples = int(joint_angles.shape[1]*2/dt)
    joint_angles = downsample(joint_angles, n_samples)
    joint_angles = np.transpose(joint_angles, (1, 0, 2))  # time x trial x position

    activity = np.stack(activity_list)
    activity = np.transpose(activity, (2, 0, 1))
    activity = activity

    # average within conditions
    if condition_average:
        cond_activity = np.zeros((activity.shape[0], n_targets, activity.shape[-1]))
        cond_hand_pos = np.zeros((hand_pos.shape[0], n_targets, hand_pos.shape[-1]))
        cond_joint = np.zeros((joint_angles.shape[0], n_targets, joint_angles.shape[-1]))
        cond_trials_info = []
        cond_trials_timing = []

        trials_target = np.array([info[condition_label] for info in trial_info])
        for target in range(n_targets):
            trials = np.argwhere(trials_target == target).reshape(-1)
            cond_activity[:, target] = np.mean(activity[:, trials], axis=1)
            cond_trials_info.append(trial_info[trials[0]])
            cond_trials_timing.append(trial_timing[trials[0]])
            cond_hand_pos[:, target] = np.mean(hand_pos[:, trials], axis=1)
            cond_joint[:, target] = np.mean(joint_angles[:, trials], axis=1)

        activity = cond_activity
        hand_pos = cond_hand_pos
        joint_angles = cond_joint
        trial_timing = cond_trials_timing
        trial_info = cond_trials_info

    # temporal smoothing
    if smooth_std:
        def gaussian_filter(size, sigma):
            filter_range = np.linspace(-int(size/2),int(size/2),size)
            gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
            return np.array(gaussian_filter)

        filter = gaussian_filter(activity.shape[0], smooth_std / dt)
        activity = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=0, arr=activity)

    regions = ["M1"]*activity.shape[-1]
    print("Hatsopoulos2007 data processed. Activity shape:", activity.shape)
    return {
        "activity": activity,
        "trial_info": trial_info,
        "regions": regions,
        "hand_pos": hand_pos,
        "joint_angles": joint_angles,
        "trial_timing": trial_timing
    }



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # [Neural population dynamics in motor cortex are different for reach and grasp](https://elifesciences.org/articles/58848)
    # Suresh2020, fig1a
    data = process_data(condition_average=True)
    print("activity shape:", data["activity"].shape)
    print("trial_info:", data["trial_info"])
    print("trial_timing:", data["trial_timing"])
    # activity: time x trial x neuron
    neurons = [20, 100, 48, 30]
    for neuron in neurons:
        plt.figure()
        plt.plot(data["activity"][:, :, neuron])
    plt.show()
