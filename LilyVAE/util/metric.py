"""
used to recoder and cal metrics
"""
from typing import Union

import numpy as np
from torch import Tensor


class Metric:
    """
    recorde tp, fp, tn, fn of status prediction and mae of power prediction
    and then provide acc, pre, rec, f1, mae through `get_index()`
    """

    def __init__(self, threshold: int, window_size: int, window_stride: int):
        super(Metric, self).__init__()
        self.threshold = threshold
        self.window_size = window_size
        self.window_stride = window_stride
        self.apps = np.empty((0, window_size))
        self.apps_pred = np.empty((0, window_size))

    def add(self, apps: Union[Tensor, np.ndarray], apps_pred: Union[Tensor, np.ndarray]):
        """
        update the ground apps/status and the pred apps/status
        Parameters:
            apps of shape `(bs, L)` or `(bs, 1, L)`: the ground power of the app
            apps_pred of shape `(bs, L)` or `(bs, 1, L)`: the pred power of the app
        """
        if isinstance(apps, Tensor):
            apps = apps.squeeze().cpu().detach().numpy()
        if isinstance(apps_pred, Tensor):
            apps_pred = apps_pred.squeeze().cpu().detach().numpy()
        self.apps = np.concatenate([self.apps, apps])
        self.apps_pred = np.concatenate([self.apps_pred, apps_pred])

    def get_metrics(self):
        """ return metrics """
        apps, apps_pred = self.merge(self.apps), self.merge(self.apps_pred)
        status, status_pred = apps > self.threshold, apps_pred > self.threshold
        tp, fp = np.sum(status & status_pred), np.sum(~status & status_pred)
        tn, fn = np.sum(~status & ~status_pred), np.sum(status & ~status_pred)
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        pre = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0
        mae = np.fabs(apps - apps_pred).mean()
        mae_on = np.fabs(apps[status] - apps_pred[status]).mean()
        return mae, mae_on, acc, pre, rec, f1

    def merge(self, seqs, overlap=True):
        """
        merge the overlap seqs
        1. populate the seq stair by seqs, take `window_size = 4` and `window_stride=1` for example:
            1,2,3,4|5,6,7,8|x,
            x,2,3,4,5|6,7,8,9|
            x,x,3,4,5,6|x,x,x,
            x,x,x,4,5,6,7|x,x,
        """
        if overlap:
            return seqs.flatten()
        total_length = self.window_size + self.window_stride * (len(self.apps) - 1)
        overlops = self.window_size // self.window_stride
        seq_stair = np.full((overlops, total_length), np.nan)
        for idx, seq in enumerate(seqs):
            stair_idx = idx % overlops
            start = idx // overlops * self.window_size + stair_idx * self.window_stride
            seq_stair[stair_idx, start:start + self.window_size] = seq

        return np.nanmedian(seq_stair, axis=0)

# metric  = Metric(4, 6, 2)
# apps = np.array([1,2,3,4,5,6,7,8])
# apps_pred = np.array([2,3,5,7,6,2,4,6])
# metric.add(apps[None, 0:6], apps_pred[None, 0:6])
# metric.add(apps[None, 2:8], apps_pred[None, 2:8])
# print(metric.get_metrics())
