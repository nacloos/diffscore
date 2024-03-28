import numpy as np


def trial_acc(ob, gt, action_pred):
    ob = np.asarray(ob)
    gt = np.asarray(gt)
    action_pred = np.asarray(action_pred)

    if len(ob.shape) == 3:
        # ob: (seq_len, batch_size, ob_dim)
        fix = ob[:, :, 0]
        assert len(gt.shape) == 2
    else:
        # ob: (seq_len, ob_dim)
        fix = ob[:, 0]
        assert len(gt.shape) == 1

    fix_acc = np.sum(action_pred[fix == 1] == gt[fix == 1])/np.sum(fix)
    act_acc = np.sum(action_pred[fix == 0] == gt[fix == 0])/np.sum(1-fix)
    return fix_acc, act_acc


def trial_choice_acc(x, y, pred):
    return trial_acc(x, y, pred)[1]
