import numpy as np
import networkx as nx
import re

from pathlib import Path
from pgmpy.utils import get_example_model

class DAGMetrics():
    def __init__(self):
        self
     
    def confusion_counts(self, posterior: np.ndarray, ground_truth: np.ndarray):
        B, N, _ = posterior.shape
        tmp = (N * (N - 1)) // 2
        mask = ~np.eye(N, dtype=bool)            # (N, N)
        post_m = (posterior != 0)[:, mask]       # (B, M)
        gt_m = (ground_truth != 0)[mask][None]   # (1, M)
        M = mask.sum()

        tp = np.count_nonzero(post_m & gt_m, axis=1)
        fp = np.count_nonzero(post_m & ~gt_m, axis=1)
        fn = np.count_nonzero(~post_m & gt_m, axis=1)
        tn = M - (tp + fp + fn)
        
        a = np.count_nonzero(post_m, axis=1)
        i = tmp - a
        wa = np.divide(1.0, a, out=np.zeros_like(a, dtype=np.float64), where=(a != 0))
        wi = np.divide(1.0, i, out=np.zeros_like(i, dtype=np.float64), where=(i != 0))
        
        return tp, tn, fp, fn, wa, wi